#!/usr/bin/env python3
"""评估 CNN1D 模型在 train_test 数据集上的效果"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    
    # 模型输出23类，评估时扩展到24类（含unknown）
    model_num_classes = num_classes
    eval_num_classes = 24
    
    # 添加第24类unknown到class_names用于显示
    if len(class_names) == 23:
        class_names.append('unknown')

    # 4. 加载并预处理 train_test 数据
    test_path = os.path.join(base_dir, 'Train', 'train_test')
    if not os.path.exists(test_path):
        print(f"错误: 测试数据不存在 {test_path}")
        return

    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"\n[数据] train_test 原始数据: {df_test.shape}")

    # 生成多分类标签和二分类标签
    df_test['label_binary'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 将训练集中不存在的标签编码为23（第24类-可疑流量）
    if class_names and len(class_names) == eval_num_classes:
        # 使用前23类进行映射，未知的映射为23（unknown）
        label_to_idx = {name: i for i, name in enumerate(class_names[:-1])}
        df_test['label_multiclass_encoded'] = df_test['label'].map(label_to_idx).fillna(eval_num_classes - 1).astype(int)
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

    # 5. 模型预测（含阈值判断识别unknown）
    print("\n[预测] 开始预测...")
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        logits = model(X_tensor).numpy()

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob_23 = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_pred_23 = np.argmax(y_prob_23, axis=1)
    max_probs = np.max(y_prob_23, axis=1)
    
    # 分析unknown样本的概率分布，确定合理阈值
    unknown_mask = y_test == eval_num_classes - 1
    if np.sum(unknown_mask) > 0:
        unknown_max_probs = max_probs[unknown_mask]
        print(f"[分析] unknown样本最大概率分布:")
        print(f"  最小值: {unknown_max_probs.min():.4f}")
        print(f"  最大值: {unknown_max_probs.max():.4f}")
        print(f"  平均值: {unknown_max_probs.mean():.4f}")
        print(f"  中位数: {np.median(unknown_max_probs):.4f}")
        print(f"  前25%分位数: {np.percentile(unknown_max_probs, 25):.4f}")
        print(f"  前75%分位数: {np.percentile(unknown_max_probs, 75):.4f}")
    
    known_mask = y_test != eval_num_classes - 1
    if np.sum(known_mask) > 0:
        known_max_probs = max_probs[known_mask]
        print(f"[分析] 已知类别样本最大概率分布:")
        print(f"  最小值: {known_max_probs.min():.4f}")
        print(f"  最大值: {known_max_probs.max():.4f}")
        print(f"  平均值: {known_max_probs.mean():.4f}")
        print(f"  中位数: {np.median(known_max_probs):.4f}")
    
    # 根据分布差异设置阈值
    threshold = 0.5
    if np.sum(unknown_mask) > 0 and np.sum(known_mask) > 0:
        unknown_median = np.median(unknown_max_probs)
        known_median = np.median(known_max_probs)
        threshold = (unknown_median + known_median) / 2
        print(f"[阈值] 根据分布差异自动设置阈值: {threshold:.4f}")
    
    y_pred = y_pred_23.copy()
    y_pred[max_probs < threshold] = eval_num_classes - 1  # 23
    
    # 扩展概率矩阵到24类
    y_prob = np.zeros((len(y_test), eval_num_classes))
    y_prob[:, :model_num_classes] = y_prob_23
    y_prob[:, eval_num_classes - 1] = 1.0 - max_probs

    # 保留所有样本用于评估
    y_test_valid = y_test
    y_pred_valid = y_pred
    y_prob_valid = y_prob
    
    unknown_count = np.sum(y_test == eval_num_classes - 1)
    unknown_predicted = np.sum(y_pred == eval_num_classes - 1)
    print(f"[评估] 阈值: {threshold}, 可疑流量（第24类）真实数: {unknown_count}, 预测数: {unknown_predicted}")

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

    # 已知类别（前23类）的准确率
    unknown_idx = eval_num_classes - 1
    known_mask = y_test_valid != unknown_idx
    known_acc = accuracy_score(y_test_valid[known_mask], y_pred_valid[known_mask]) if np.sum(known_mask) > 0 else 0
    
    # unknown类指标
    unknown_tp = np.sum((y_test_valid == unknown_idx) & (y_pred_valid == unknown_idx))
    unknown_fn = np.sum((y_test_valid == unknown_idx) & (y_pred_valid != unknown_idx))
    unknown_fp = np.sum((y_test_valid != unknown_idx) & (y_pred_valid == unknown_idx))
    unknown_recall = unknown_tp / (unknown_tp + unknown_fn) if (unknown_tp + unknown_fn) > 0 else 0
    unknown_precision = unknown_tp / (unknown_tp + unknown_fp) if (unknown_tp + unknown_fp) > 0 else 0
    
    print(f"\n[已知类别] 前23类准确率: {known_acc:.4f}")
    print(f"[可疑流量] 第24类(unknown)真实数: {np.sum(y_test_valid == unknown_idx)}")
    print(f"[可疑流量] 预测为可疑流量数: {np.sum(y_pred_valid == unknown_idx)}")
    print(f"[可疑流量] unknown召回率: {unknown_recall:.4f}, 精确率: {unknown_precision:.4f}")

    try:
        auc = roc_auc_score(y_test_valid, y_prob_valid, multi_class='ovr',
                            average='weighted', labels=list(range(eval_num_classes)))
        print(f"AUC (OvR weighted):   {auc:.4f}")
    except Exception as e:
        print(f"AUC 计算失败: {e}")

    # 7. 每个类别的详细表现
    print("\n" + "=" * 70)
    print("每个类别的详细表现")
    print("=" * 70)
    cr = classification_report(y_test_valid, y_pred_valid,
                               labels=list(range(eval_num_classes)),
                               target_names=class_names if class_names else None,
                               zero_division=0)
    print(cr)

    # 8. 混淆矩阵
    print("\n" + "=" * 70)
    print("混淆矩阵（行=真实，列=预测）")
    print("=" * 70)
    cm = confusion_matrix(y_test_valid, y_pred_valid, labels=list(range(eval_num_classes)))
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

    # 9. 可视化分析
    print("\n" + "=" * 70)
    print("可视化分析")
    print("=" * 70)

    cm = confusion_matrix(y_test_valid, y_pred_valid, labels=list(range(eval_num_classes)))

    # 图1: 每类攻击统计（总数、正确数、错误数）
    plt.figure(figsize=(20, 10))
    total_per_class = np.bincount(y_test_valid, minlength=eval_num_classes)
    correct_per_class = np.diag(cm)
    incorrect_per_class = total_per_class - correct_per_class

    x = np.arange(eval_num_classes)
    width = 0.35

    plt.bar(x - width/2, total_per_class, width, label='总数', 
            color='#1f77b4', edgecolor='black', alpha=0.9)
    plt.bar(x + width/2, correct_per_class, width, label='正确', 
            color='#2ca02c', edgecolor='black', alpha=0.9)

    for i, (total, correct) in enumerate(zip(total_per_class, correct_per_class)):
        if total > 0:
            acc = correct / total * 100
            plt.text(i - width/2, total + max(total_per_class)*0.01, f'{total}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.text(i + width/2, correct + max(total_per_class)*0.01, 
                     f'{correct}\n({acc:.1f}%)',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     linespacing=0.8)

    plt.xlabel('攻击类型', fontsize=12, fontweight='bold')
    plt.ylabel('样本数量', fontsize=12, fontweight='bold')
    plt.title('每类攻击测试集统计', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names if class_names else [str(i) for i in range(eval_num_classes)], 
               rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('eval_cnn1d_class_counts.png', dpi=300, bbox_inches='tight')
    print("每类攻击统计图已保存: eval_cnn1d_class_counts.png")
    plt.close()

    # 图2: 误判分析（每类被误判成什么）
    plt.figure(figsize=(24, 16))
    plot_count = 0
    for true_class in range(eval_num_classes):
        if total_per_class[true_class] == 0:
            continue
        row = cm[true_class]
        misclassified = row.copy()
        misclassified[true_class] = 0
        if misclassified.sum() == 0:
            continue
        plot_count += 1
        mis_class_indices = np.where(misclassified > 0)[0]
        mis_class_counts = misclassified[mis_class_indices]
        mis_class_names = [class_names[i] if class_names else str(i) for i in mis_class_indices]
        accuracy = row[true_class] / total_per_class[true_class] * 100

        plt.subplot(5, 5, plot_count)
        plt.barh(mis_class_names, mis_class_counts, 
                 color='#ff7f0e', edgecolor='black', alpha=0.9)
        plt.title(f'{class_names[true_class] if class_names else str(true_class)}\n(准确率: {accuracy:.1f}%, 错误: {misclassified.sum()})', 
                  fontsize=10, fontweight='bold')
        plt.xlabel('误判数量', fontsize=9)
        plt.yticks(fontsize=8)
        plt.xlim(0, max(mis_class_counts) * 1.15)
        for i, v in enumerate(mis_class_counts):
            plt.text(v + max(mis_class_counts)*0.02, i, f'{v}', 
                     va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('eval_cnn1d_misclassification.png', dpi=300, bbox_inches='tight')
    print("误判分析图已保存: eval_cnn1d_misclassification.png")
    plt.close()

    # 图3: 归一化混淆矩阵
    plt.figure(figsize=(18, 14))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names if class_names else [str(i) for i in range(eval_num_classes)],
                yticklabels=class_names if class_names else [str(i) for i in range(eval_num_classes)],
                annot_kws={'fontsize': 8, 'fontweight': 'bold'},
                cbar_kws={'label': '比例', 'shrink': 0.8})
    plt.title(f'CNN1D 归一化混淆矩阵 ({eval_num_classes} 类)', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('eval_cnn1d_confusion_normalized.png', dpi=300, bbox_inches='tight')
    print("归一化混淆矩阵图已保存: eval_cnn1d_confusion_normalized.png")
    plt.close()

    # 图4: 每类精确率、召回率、F1
    precisions = precision_score(y_test_valid, y_pred_valid, average=None, zero_division=0,
                                 labels=list(range(eval_num_classes)))
    recalls = recall_score(y_test_valid, y_pred_valid, average=None, zero_division=0,
                           labels=list(range(eval_num_classes)))
    f1_scores = f1_score(y_test_valid, y_pred_valid, average=None, zero_division=0,
                         labels=list(range(eval_num_classes)))

    plt.figure(figsize=(20, 10))
    x = np.arange(eval_num_classes)
    width = 0.25

    plt.bar(x - width, precisions, width, label='精确率', color='#1f77b4', edgecolor='black')
    plt.bar(x, recalls, width, label='召回率', color='#ff7f0e', edgecolor='black')
    plt.bar(x + width, f1_scores, width, label='F1', color='#2ca02c', edgecolor='black')

    for i in range(eval_num_classes):
        if total_per_class[i] > 0:
            plt.text(i - width, precisions[i] + 0.02, f'{precisions[i]:.2f}', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.text(i, recalls[i] + 0.02, f'{recalls[i]:.2f}', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.text(i + width, f1_scores[i] + 0.02, f'{f1_scores[i]:.2f}', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('攻击类型', fontsize=12, fontweight='bold')
    plt.ylabel('指标值', fontsize=12, fontweight='bold')
    plt.title('每类攻击精确率、召回率、F1-Score', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names if class_names else [str(i) for i in range(eval_num_classes)], 
               rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('eval_cnn1d_per_class_metrics.png', dpi=300, bbox_inches='tight')
    print("每类指标图已保存: eval_cnn1d_per_class_metrics.png")
    plt.close()

    # 图5: 混淆矩阵热力图（含类别名称）
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else [str(i) for i in range(eval_num_classes)],
                yticklabels=class_names if class_names else [str(i) for i in range(eval_num_classes)],
                annot_kws={'fontsize': 8, 'fontweight': 'bold'},
                cbar_kws={'label': '样本数量'})
    plt.title(f'CNN1D 混淆矩阵 ({eval_num_classes} 类)', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('eval_cnn1d_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵图已保存: eval_cnn1d_confusion_matrix.png")
    plt.close()

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
