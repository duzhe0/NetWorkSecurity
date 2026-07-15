#!/usr/bin/env python3
"""评估 XGBoost 模型在 train_test 数据集上的效果

新版 XGBoost 在训练阶段已通过零权重虚拟样本补齐到 23 类（0..22），
预测空间与编码空间直接一致，无需任何标签重编码。
本脚本与 eval_dnn_train_test.py 逻辑对齐，确保 app 与 eval 结果一致。
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# =============== 常量定义（与预处理模块一致）===============
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

def load_and_preprocess(base_dir, class_names):
    """加载 train_test 并预处理，返回 (X_test, y_test, feature_cols)"""

    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_standard.pkl')
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
    print(f"[预处理] 已加载 OneHotEncoder 和 StandardScaler")

    test_path = os.path.join(base_dir, 'Train', 'train_test')
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"[数据] train_test 原始数据: {df_test.shape}")

    # 生成标签：未知攻击(训练集 23 类之外)映射为 -1，评估时剔除
    df_test['label_binary'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    if class_names:
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        df_test['label_multiclass_encoded'] = df_test['label'].map(label_to_idx).fillna(-1).astype(int)
    else:
        df_test['label_multiclass_encoded'] = 0

    y_test = df_test['label_multiclass_encoded'].values
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    print(f"[数据] 测试集标签分布:")
    for lbl, cnt in zip(unique_labels, label_counts):
        lbl_name = class_names[lbl] if class_names and 0 <= lbl < len(class_names) else str(lbl)
        print(f"  {lbl_name}: {cnt}")

    # 从训练集样本获取特征列顺序
    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    df_train_sample = pd.read_csv(train_csv_path, nrows=1)
    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                    'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]

    # One-Hot 编码
    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
    df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
    df_test_enc = pd.concat([df_rest, df_ohe], axis=1)

    # 补齐训练集有的列
    for col in feature_cols:
        if col not in df_test_enc.columns:
            df_test_enc[col] = 0.0
    df_test_enc = df_test_enc[feature_cols + ['label_binary']]

    # 标准化
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_test_enc.columns]
    df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])

    X_test = df_test_enc[feature_cols].values.astype(np.float32)
    print(f"[数据] 预处理完成，特征矩阵: {X_test.shape}")

    return X_test, y_test, feature_cols


def print_eval_results(y_true, y_pred, y_prob, num_classes, class_names):
    """打印评估结果、分类报告、混淆矩阵"""

    # 剔除未知标签(-1)样本，与 app.py /test/model 及其他 eval 脚本一致
    valid_mask = y_true >= 0
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    y_prob_v = y_prob[valid_mask]
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[过滤] 剔除 {n_dropped} 个未知标签样本（训练集 23 类之外的新攻击）")

    print("\n" + "=" * 70)
    print("评估结果（多分类，weighted 平均）")
    print("=" * 70)

    accuracy = accuracy_score(y_true_v, y_pred_v)
    precision = precision_score(y_true_v, y_pred_v, average='weighted', zero_division=0)
    recall = recall_score(y_true_v, y_pred_v, average='weighted', zero_division=0)
    f1 = f1_score(y_true_v, y_pred_v, average='weighted', zero_division=0)

    print(f"准确率 (Accuracy):     {accuracy:.4f}")
    print(f"精确率 (Precision):    {precision:.4f}")
    print(f"召回率 (Recall):       {recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")

    try:
        auc = roc_auc_score(y_true_v, y_prob_v, multi_class='ovr',
                            average='weighted', labels=list(range(num_classes)))
        print(f"AUC (OvR weighted):   {auc:.4f}")
    except Exception as e:
        print(f"AUC 计算失败: {e}")

    print("\n" + "=" * 70)
    print("每个类别的详细表现")
    print("=" * 70)
    cr = classification_report(y_true_v, y_pred_v,
                               labels=list(range(num_classes)),
                               target_names=class_names if class_names else None,
                               zero_division=0)
    print(cr)

    print("\n" + "=" * 70)
    print("混淆矩阵（行=真实，列=预测）")
    print("=" * 70)
    cm = confusion_matrix(y_true_v, y_pred_v, labels=list(range(num_classes)))
    print("标签索引:")
    if class_names:
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    print("\n混淆矩阵:")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:5d}" for v in row)
        if class_names and i < len(class_names):
            print(f"  {class_names[i]:18s} | {row_str}")
        else:
            print(f"  {i:18d} | {row_str}")

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = "XGBoost"

    print("=" * 70)
    print(f"{model_name} 模型在 train_test 数据集上的效果评估")
    print("=" * 70)

    # 加载类别列表（原始 23 类，固定顺序）
    class_file = os.path.join(base_dir, 'Train', 'encoder_multiclass_23_classes.txt')
    class_names = []
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]

    # 加载模型
    model_path = os.path.join(base_dir, 'models', 'xgboost', 'model_xgboost.pkl')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return

    model = joblib.load(model_path)
    num_classes = model.n_classes_ if hasattr(model, 'n_classes_') else (len(class_names) or 23)
    print(f"[模型] 类别数: {num_classes}")
    print(f"[模型] 已加载 {model_path}")

    # 数据预处理
    X_test, y_test, feature_cols = load_and_preprocess(base_dir, class_names)

    # 预测：新版模型在 0..num_classes-1 空间直接训练，预测即为编码空间，无需重编码
    print("\n[预测] 开始预测...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print(f"[预测] 完成，y_prob shape: {y_prob.shape}")

    # 打印结果
    print_eval_results(y_test, y_pred, y_prob, num_classes, class_names)


if __name__ == '__main__':
    main()
