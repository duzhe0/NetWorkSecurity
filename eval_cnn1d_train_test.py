#!/usr/bin/env python3
"""评估 CNN1D 模型在 train_test 数据集上的效果"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# =============== 模型定义（与训练代码一致）===============
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels=[64, 128, 256],
                 kernel_size=3, dropout_rate=0.3):
        super(CNN1D, self).__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)

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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("CNN1D 模型在 train_test 数据集上的效果评估")
    print("=" * 70)

    # 1. 加载模型
    model_path = os.path.join(base_dir, 'models', 'cnn1d', 'model_cnn1d.pth')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return

    state_dict = torch.load(model_path, map_location='cpu')
    # 从 state_dict 推断输入维度和类别数
    num_classes = state_dict['classifier.3.weight'].shape[0]
    # 训练时使用的 input_dim 从训练数据获取，这里先假设用标准预处理后的维度
    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    if os.path.exists(train_csv_path):
        df_train_sample = pd.read_csv(train_csv_path, nrows=1)
        exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                        'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
        feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]
        input_dim = len(feature_cols)
        print(f"[模型] 输入维度: {input_dim}, 类别数: {num_classes}")
    else:
        input_dim = 117
        print(f"[模型] 使用默认输入维度: {input_dim}, 类别数: {num_classes}")

    model = CNN1D(input_dim, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[模型] 已加载 {model_path}")

    # 2. 加载预处理器
    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_standard.pkl')
    if not os.path.exists(ohe_path) or not os.path.exists(scaler_path):
        print("错误: 预处理器文件不存在，请先运行数据预处理")
        return

    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
    print("[预处理] 已加载 OneHotEncoder 和 StandardScaler")

    # 3. 加载类别列表（用于显示标签名称）
    class_file = os.path.join(base_dir, 'Train', 'encoder_multiclass_23_classes.txt')
    class_names = []
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"[标签] 类别列表: {class_names}")
    else:
        print("[标签] 未找到类别列表文件")

    # 4. 加载并预处理 train_test 数据
    test_path = os.path.join(base_dir, 'Train', 'train_test')
    if not os.path.exists(test_path):
        print(f"错误: 测试数据不存在 {test_path}")
        return

    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"\n[数据] train_test 原始数据: {df_test.shape}")

    # 生成多分类标签和二分类标签
    df_test['label_binary'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    if class_names:
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        unknown_idx = len(class_names) - 1
        df_test['label_multiclass_encoded'] = df_test['label'].map(label_to_idx).fillna(unknown_idx).astype(int)
    else:
        df_test['label_multiclass_encoded'] = 0

    y_test = df_test['label_multiclass_encoded'].values
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    print(f"[数据] 测试集标签分布:")
    for lbl, cnt in zip(unique_labels, label_counts):
        lbl_name = class_names[lbl] if class_names and 0 <= lbl < len(class_names) else str(lbl)
        print(f"  {lbl_name}: {cnt}")

    # One-Hot 编码
    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
    df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
    df_test_enc = pd.concat([df_rest, df_ohe], axis=1)

    # 补齐训练集有的列
    if 'feature_cols' in dir():
        for col in feature_cols:
            if col not in df_test_enc.columns:
                df_test_enc[col] = 0.0
        df_test_enc = df_test_enc[feature_cols + ['label_binary']]

    # 标准化
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_test_enc.columns]
    df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])
    print(f"[数据] 预处理完成，特征矩阵: {df_test_enc[feature_cols].shape}")

    X_test = df_test_enc[feature_cols].values.astype(np.float32)

    # 5. 模型预测
    print("\n[预测] 开始预测...")
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        logits = model(X_tensor).numpy()

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)

    # 过滤掉标签不在类别列表中的样本
    valid_mask = y_test >= 0
    y_test_valid = y_test[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_prob_valid = y_prob[valid_mask]
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[过滤] 剔除 {n_dropped} 个非法标签样本（不应出现）")

    # 6. 计算指标
    print("\n" + "=" * 70)
    print("评估结果（多分类，weighted 平均）")
    print("=" * 70)

    accuracy = accuracy_score(y_test_valid, y_pred_valid)
    precision = precision_score(y_test_valid, y_pred_valid, average='weighted', zero_division=0)
    recall = recall_score(y_test_valid, y_pred_valid, average='weighted', zero_division=0)
    f1 = f1_score(y_test_valid, y_pred_valid, average='weighted', zero_division=0)

    print(f"准确率 (Accuracy):     {accuracy:.4f}")
    print(f"精确率 (Precision):    {precision:.4f}")
    print(f"召回率 (Recall):       {recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")

    try:
        auc = roc_auc_score(y_test_valid, y_prob_valid, multi_class='ovr',
                            average='weighted', labels=list(range(num_classes)))
        print(f"AUC (OvR weighted):   {auc:.4f}")
    except Exception as e:
        print(f"AUC 计算失败: {e}")

    # 7. 每个类别的详细表现
    print("\n" + "=" * 70)
    print("每个类别的详细表现")
    print("=" * 70)
    cr = classification_report(y_test_valid, y_pred_valid,
                               labels=list(range(num_classes)),
                               target_names=class_names if class_names else None,
                               zero_division=0)
    print(cr)

    # 8. 混淆矩阵
    print("\n" + "=" * 70)
    print("混淆矩阵（行=真实，列=预测）")
    print("=" * 70)
    cm = confusion_matrix(y_test_valid, y_pred_valid, labels=list(range(num_classes)))
    print("标签索引:")
    if class_names:
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    print("\n混淆矩阵:")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:4d}" for v in row)
        if class_names and i < len(class_names):
            print(f"  {class_names[i]:15s} | {row_str}")
        else:
            print(f"  {i:15d} | {row_str}")

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
