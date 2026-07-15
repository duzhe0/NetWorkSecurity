#!/usr/bin/env python3
"""评估 XGBoost 模型在 train_test 数据集上的效果"""

import pandas as pd
import numpy as np
import os
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

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
    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_robust.pkl')
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
    print(f"[预处理] 已加载 OneHotEncoder 和 StandardScaler")

    test_path = os.path.join(base_dir, 'Train', 'train_test')
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"[数据] train_test 原始数据: {df_test.shape}")

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

    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    df_train_sample = pd.read_csv(train_csv_path, nrows=1)
    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                    'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]

    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
    df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
    df_test_enc = pd.concat([df_rest, df_ohe], axis=1)

    for col in feature_cols:
        if col not in df_test_enc.columns:
            df_test_enc[col] = 0.0
    df_test_enc = df_test_enc[feature_cols + ['label_binary']]

    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_test_enc.columns]
    # Binary Indicator
    ZERO_INFLATED_COLS = ['src_bytes', 'dst_bytes', 'duration',
                           'num_failed_logins', 'num_shells', 'num_access_files',
                           'num_file_creations', 'num_root']
    BINARY_INDICATOR_COLS = []
    for col in ZERO_INFLATED_COLS:
        if col in df_test_enc.columns:
            indicator_name = f'is_zero_{col}'
            df_test_enc[indicator_name] = (df_test_enc[col] == 0).astype(int)
            BINARY_INDICATOR_COLS.append(indicator_name)
    # Log1p transform
    LOGP1_COLS = ['src_bytes', 'dst_bytes', 'duration', 'hot']
    for col in LOGP1_COLS:
        if col in df_test_enc.columns:
            df_test_enc[col] = np.log1p(df_test_enc[col])
    numeric_cols.extend(BINARY_INDICATOR_COLS)
    df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])

    X_test = df_test_enc[feature_cols].values.astype(np.float32)
    print(f"[数据] 预处理完成，特征矩阵: {X_test.shape}")

    return X_test, y_test, feature_cols


def get_xgb_label_map(base_dir):
    """获取 XGBoost 标签重编码映射。优先从 pkl 读取，没有则从训练集重建"""
    label_map_path = os.path.join(base_dir, 'models', 'xgboost', 'xgboost_label_map.pkl')
    if os.path.exists(label_map_path):
        with open(label_map_path, 'rb') as f:
            saved = pickle.load(f)
        label_map = saved['label_map']
        train_labels = saved['train_labels']
        num_class = saved['num_class']
        print(f"[映射] 从 xgboost_label_map.pkl 加载: {num_class} 个连续标签")
        return label_map, train_labels, num_class

    # 从训练集重建
    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    df_train = pd.read_csv(train_csv_path, usecols=['label_multiclass_encoded'])
    train_labels = sorted(df_train['label_multiclass_encoded'].unique())
    label_map = {old: i for i, old in enumerate(train_labels)}
    num_class = len(label_map)
    print(f"[映射] 从训练集重建标签映射: {num_class} 个连续标签")
    print(f"  原始标签 -> 模型标签:")
    for old, new in sorted(label_map.items()):
        print(f"    {old:3d} -> {new:2d}")
    return label_map, train_labels, num_class


def print_eval_results(y_true, y_pred, y_prob, num_classes, class_names, label_map, train_labels):
    """打印评估结果。y_true 已在模型空间。"""

    valid_mask = y_true >= 0
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    y_prob_v = y_prob[valid_mask]
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[过滤] 剔除 {n_dropped} 个训练集中未出现的标签样本")

    # 构造显示用的类别名（仅模型空间内有的类别）
    if class_names and label_map:
        # 反向映射：模型标签 -> 原始标签索引 -> 类别名
        idx_to_orig = {v: k for k, v in label_map.items()}
        display_names = []
        for i in range(num_classes):
            orig = idx_to_orig.get(i, -1)
            if 0 <= orig < len(class_names):
                display_names.append(class_names[orig])
            else:
                display_names.append(f"class_{i}")
    else:
        display_names = [f"class_{i}" for i in range(num_classes)]

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
                               target_names=display_names,
                               zero_division=0)
    print(cr)

    print("\n" + "=" * 70)
    print("混淆矩阵（行=真实，列=预测）")
    print("=" * 70)
    cm = confusion_matrix(y_true_v, y_pred_v, labels=list(range(num_classes)))
    print("标签索引（模型空间 -> 原始标签 -> 类别名）:")
    for i in range(num_classes):
        orig = idx_to_orig.get(i, -1) if 'idx_to_orig' in dir() else i
        name = display_names[i] if i < len(display_names) else f"class_{i}"
        print(f"  {i:2d} -> {orig:3d} -> {name}")
    print("\n混淆矩阵:")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:5d}" for v in row)
        name = display_names[i] if i < len(display_names) else f"class_{i}"
        print(f"  {name:22s} | {row_str}")

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = "XGBoost"

    print("=" * 70)
    print(f"{model_name} 模型在 train_test 数据集上的效果评估")
    print("=" * 70)

    # 加载类别名（原始23类）
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
    num_classes = model.n_classes_ if hasattr(model, 'n_classes_') else None
    print(f"[模型] 类别数: {num_classes}")
    print(f"[模型] 已加载 {model_path}")

    # 获取标签映射（原始编码 -> 模型编码）
    label_map, train_labels, num_class = get_xgb_label_map(base_dir)
    if num_classes is None:
        num_classes = num_class

    # 数据预处理
    X_test, y_test_orig, feature_cols = load_and_preprocess(base_dir, class_names)

    # 将 y_test 重编码到模型空间
    y_test = np.array([label_map.get(int(v), -1) for v in y_test_orig], dtype=np.int64)
    print(f"[映射] 测试标签已重编码到模型空间 (0..{num_class-1})")

    # 预测
    print("\n[预测] 开始预测...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print(f"[预测] 完成，y_prob shape: {y_prob.shape}")

    # 打印结果
    print_eval_results(y_test, y_pred, y_prob, num_classes, class_names, label_map, train_labels)


if __name__ == '__main__':
    main()
