# =============================================================================
# 文件名: cnn1d_model.py
# 描述: 基于PyTorch实现的1D卷积神经网络（CNN1D），用于KDD CUP 99数据集的多分类任务。
#       将117维特征视为长度为117的单通道时序信号，通过多层1D卷积+全局平均池化进行分类。
#       支持23个类别（normal + 22种攻击类型）。
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
warnings.filterwarnings('ignore')

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置matplotlib中文字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 定义1D-CNN模型类
# =============================================================================
class CNN1D(nn.Module):
    """
    1D卷积神经网络模型（多分类版，默认23类：normal + 22种攻击）
    将输入特征视为长度为 input_dim 的单通道序列，使用多层1D卷积、
    BatchNorm、ReLU、MaxPool和Dropout，最后通过全局平均池化+全连接层输出类别概率。
    参数:
        input_dim (int): 输入特征维度（序列长度） 117维
        num_classes (int): 输出类别数
        conv_channels (list): 各卷积层的输出通道数列表
        kernel_size (int): 卷积核大小，默认为3
        dropout_rate (float): Dropout比率，默认为0.3
    """
    def __init__(self, input_dim, num_classes, conv_channels=[64, 128, 256],
                 kernel_size=3, dropout_rate=0.3):
        super(CNN1D, self).__init__()

        # 构建卷积层序列：Conv1d -> BatchNorm -> ReLU -> MaxPool1d -> Dropout
        layers = []
        in_ch = 1  # 初始输入通道数为1（单通道序列）
        for out_ch in conv_channels:
            # padding=kernel_size//2 保持序列长度不变（除MaxPool下采样外）
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))  # 每层将序列长度减半
            layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # 自适应平均池化：将任意长度的序列压缩为每个通道一个值（输出形状: batch, channels, 1）
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 分类器：全连接层 -> ReLU -> Dropout -> 输出层（无softmax，因为CrossEntropyLoss自带）
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        前向传播。
        输入 x: shape (batch, input_dim) 或 (batch, 1, input_dim)
        输出: shape (batch, num_classes) 的logits
        """
        # 如果输入是二维 (batch, input_dim)，则增加通道维 -> (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # 通过卷积层序列
        x = self.conv(x)
        # 全局平均池化 -> (batch, channels, 1)，然后压缩最后一维 -> (batch, channels)
        x = self.gap(x).squeeze(-1)
        # 通过分类器得到最终logits
        return self.classifier(x)


# =============================================================================
# 2. 数据加载函数
# =============================================================================
def load_preprocessed_data():
    """
    从三个独立的CSV文件加载训练集、验证集和测试集。
    文件路径: "../../Train/KDDTrain_preprocessed_train.csv" 等。
    返回:
        X_train, X_val, X_test: 特征数组 (numpy float32)
        y_train, y_val, y_test: 标签数组 (numpy int64)
        feature_cols: 特征列名列表
        num_classes: 类别总数
        class_names: 类别名称列表
    """
    print("=" * 60)
    print("1D-CNN 卷积神经网络训练")
    print("=" * 60)

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'Train') + os.sep

    # 读取预处理后的数据
    df_train = pd.read_csv(data_dir + "KDDTrain_preprocessed_train.csv")
    df_val   = pd.read_csv(data_dir + "KDDTrain_preprocessed_val.csv")
    df_test  = pd.read_csv(data_dir + "KDDTrain_preprocessed_test.csv")

    print(f"\n[数据加载] 训练集形状: {df_train.shape}")
    print(f"[数据加载] 验证集形状: {df_val.shape}")
    print(f"[数据加载] 测试集形状: {df_test.shape}")

    # 排除非特征列（标签、难度、各种编码列等）
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded',
                    'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    # 提取特征和标签（使用多分类编码 label_multiclass_encoded）
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train['label_multiclass_encoded'].values.astype(np.int64)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val['label_multiclass_encoded'].values.astype(np.int64)

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test['label_multiclass_encoded'].values.astype(np.int64)

    # 模型输出24类（包含unknown类），与app.py保持一致
    model_num_classes = 24

    # 尝试从文本文件加载类别名称（用于信息显示）
    class_file = os.path.join(data_dir, "encoder_multiclass_23_classes.txt")
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        # 添加第24类'unknown'用于显示和评估
        class_names.append('unknown')
    num_classes = len(class_names)  # 用于评估的类别数（24）
    
    # 统计unknown样本数
    unknown_count_train = (df_train['label_multiclass_encoded'] == 23).sum()
    unknown_count_val = (df_val['label_multiclass_encoded'] == 23).sum()
    unknown_count_test = (df_test['label_multiclass_encoded'] == 23).sum()
    
    print(f"\n【{model_num_classes} 分类任务】模型输出 {model_num_classes} 类，评估时通过阈值识别第24类(unknown)")
    print(f"[unknown统计] 训练集: {unknown_count_train}, 验证集: {unknown_count_val}, 测试集: {unknown_count_test}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"测试集样本数: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes, class_names


# =============================================================================
# 3. 模型训练函数
# =============================================================================
def train_cnn1d(X_train, X_val, y_train, y_val, model_num_classes, epochs=50, batch_size=64):
    """
    训练1D-CNN模型（多分类），在验证集上监控性能，并记录损失和准确率历史。
    参数:
        X_train, y_train: 训练集特征和标签
        X_val, y_val: 验证集特征和标签
        model_num_classes: 模型输出类别数（23，只训练已知类别）
        epochs: 训练轮数
        batch_size: 批次大小
    返回:
        model: 训练好的模型
        train_time: 训练耗时（秒）
        history: 包含训练损失、验证损失、验证准确率的字典
        device: 使用的设备（CPU或CUDA）
    """
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    input_dim = X_train.shape[1]  # 特征维度（序列长度）

    # 超参数设置
    conv_channels = [64, 128, 256]
    kernel_size = 3
    dropout_rate = 0.3
    learning_rate = 0.001

    print(f"\n1D-CNN 网络结构（多分类）:")
    print(f"  输入层: {input_dim} 维（作为长度 {input_dim} 的单通道序列）")
    print(f"  卷积通道: {conv_channels}, kernel_size={kernel_size}")
    print(f"  Dropout率: {dropout_rate}")
    print(f"  输出层: {model_num_classes} 维 (softmax via CrossEntropyLoss)")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")

    # 选择设备（GPU优先）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 初始化模型
    model = CNN1D(input_dim, model_num_classes, conv_channels, kernel_size, dropout_rate).to(device)

    # 使用全部训练数据（包括unknown类）
    X_train_known = X_train
    y_train_known = y_train
    X_val_known = X_val
    y_val_known = y_val
    print(f"[训练数据] 样本数: {len(y_train_known)}")
    print(f"[验证数据] 样本数: {len(y_val_known)}")

    # 将numpy数据转为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_known)
    y_train_tensor = torch.LongTensor(y_train_known)
    X_val_tensor = torch.FloatTensor(X_val_known)
    y_val_tensor = torch.LongTensor(y_val_known)

    # 使用标准CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 创建DataLoader（使用标准shuffle，不使用WeightedRandomSampler避免训练不稳定）
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n开始训练...")
    start_time = time.time()

    # 记录训练历史
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # 训练模式
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 梯度清零、前向、损失、反向、优化
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 学习率调度器步进
        scheduler.step()

        # 验证模式（不计算梯度）
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val_tensor.to(device)
            y_val_dev = y_val_tensor.to(device)
            val_outputs = model(X_val_dev)
            val_loss = criterion(val_outputs, y_val_dev).item()
            val_losses.append(val_loss)

            # 预测标签并计算准确率
            y_val_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val_known, y_val_pred)
            val_accs.append(val_acc)

        # 每10个epoch打印一次日志
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.2f} 秒")

    # 返回训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }

    return model, train_time, history, device


# =============================================================================
# 4. 模型评估函数
# =============================================================================
def tune_threshold(model, X_val, y_val, device, num_classes):
    """
    在验证集上搜索最优阈值，平衡已知类别准确率和unknown召回率。
    如果验证集没有unknown样本，则根据已知类别的概率分布自动设置阈值。
    参数:
        model: 训练好的模型
        X_val, y_val: 验证集特征和标签
        device: 当前设备
        num_classes: 评估时的类别数（24，包含unknown）
    返回:
        best_threshold: 最优阈值
        best_score: 最优分数
    """
    print("\n" + "=" * 60)
    print("阈值调优 (验证集)")
    print("=" * 60)

    model.eval()
    X_val_tensor = torch.FloatTensor(X_val).to(device)

    with torch.no_grad():
        logits = model(X_val_tensor).cpu().numpy()
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob_23 = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        max_probs = np.max(y_prob_23, axis=1)

    unknown_idx = num_classes - 1
    unknown_count_val = np.sum(y_val == unknown_idx)

    if unknown_count_val > 0:
        known_mask_val = y_val != unknown_idx
        best_threshold = 0.5
        best_score = -1

        for threshold_candidate in np.arange(0.5, 0.999, 0.01):
            y_pred = np.argmax(y_prob_23, axis=1)
            y_pred[max_probs < threshold_candidate] = unknown_idx

            known_acc_val = accuracy_score(y_val[known_mask_val], y_pred[known_mask_val]) if np.sum(known_mask_val) > 0 else 0

            unknown_tp_val = np.sum((y_val == unknown_idx) & (y_pred == unknown_idx))
            unknown_fn_val = np.sum((y_val == unknown_idx) & (y_pred != unknown_idx))
            unknown_recall_val = unknown_tp_val / (unknown_tp_val + unknown_fn_val) if (unknown_tp_val + unknown_fn_val) > 0 else 0

            score = 0.6 * known_acc_val + 0.4 * unknown_recall_val

            if score > best_score:
                best_score = score
                best_threshold = threshold_candidate

        print(f"最优阈值: {best_threshold:.4f}, 最优分数: {best_score:.4f}")
    else:
        print("验证集没有unknown样本，根据已知类别概率分布自动设置阈值")
        print(f"已知类别样本最大概率分布:")
        print(f"  最小值: {max_probs.min():.4f}")
        print(f"  最大值: {max_probs.max():.4f}")
        print(f"  平均值: {max_probs.mean():.4f}")
        print(f"  中位数: {np.median(max_probs):.4f}")
        print(f"  前10%分位数: {np.percentile(max_probs, 10):.4f}")
        
        best_threshold = np.percentile(max_probs, 5)
        print(f"设置阈值为已知类别概率的前5%分位数: {best_threshold:.4f}")
        best_score = -1

    return best_threshold, best_score


def evaluate_model(model, X_test, y_test, device, num_classes, threshold=0.5):
    """
    在测试集上评估模型性能，计算准确率、精确率、召回率、F1、AUC等指标，
    并返回混淆矩阵、预测标签和预测概率。
    
    使用开集识别（Open-Set Recognition）策略：
    - 模型输出23类的概率
    - 如果最大概率低于threshold，将样本标记为第24类(unknown/可疑流量)
    
    参数:
        model: 训练好的模型
        X_test, y_test: 测试集特征和标签
        device: 当前设备
        num_classes: 评估时的类别数（24，包含unknown）
        threshold: softmax最大概率阈值，低于此值标记为unknown
    返回:
        metrics: 字典，包含各项评估指标
        y_pred: 预测类别标签（0-23，23表示unknown）
        y_prob_matrix: 每个样本的预测概率矩阵 (n_samples, 24)
    """
    print("\n" + "=" * 60)
    print("模型评估 (测试集) - 多分类")
    print("=" * 60)

    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        # 获取logits并转为numpy
        logits = model(X_test_tensor).cpu().numpy()
        # 计算softmax概率（数值稳定性：减去最大值）
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob_23 = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # 获取23类的预测结果
        y_pred_23 = np.argmax(y_prob_23, axis=1)
        # 获取每个样本的最大概率
        max_probs = np.max(y_prob_23, axis=1)
        
        # 开集识别：低于阈值的标记为unknown(23)
        y_pred = y_pred_23.copy()
        y_pred[max_probs < threshold] = num_classes - 1  # 23
        
        # 扩展概率矩阵到24类
        y_prob_matrix = np.zeros((len(y_test), num_classes))
        y_prob_matrix[:, :23] = y_prob_23
        y_prob_matrix[:, 23] = 1.0 - max_probs  # unknown的概率 = 1 - 最大已知类概率

    # 计算加权平均指标（适用于多分类不平衡情况）
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    # 计算AUC（One-vs-Rest，加权平均）
    try:
        test_auc = roc_auc_score(y_test, y_prob_matrix, multi_class='ovr',
                                 average='weighted', labels=list(range(num_classes)))
    except Exception:
        test_auc = float('nan')  # 某些情况下可能计算失败

    print(f"\n[测试集] 评估结果 (多分类, {num_classes} 类, threshold={threshold}):")
    print(f"  准确率 (Accuracy): {test_acc:.4f}")
    print(f"  精确率 (Precision weighted): {test_precision:.4f}")
    print(f"  召回率 (Recall weighted): {test_recall:.4f}")
    print(f"  F1-Score weighted: {test_f1:.4f}")
    print(f"  AUC (weighted OvR): {test_auc:.4f}")

    # 可疑流量（第24类）统计
    unknown_idx = num_classes - 1
    unknown_count = np.sum(y_test == unknown_idx)
    unknown_predicted = np.sum(y_pred == unknown_idx)
    known_correct = np.sum((y_test != unknown_idx) & (y_pred != unknown_idx) & (y_test == y_pred))
    known_total = np.sum(y_test != unknown_idx)
    
    # unknown类的指标
    unknown_true_positive = np.sum((y_test == unknown_idx) & (y_pred == unknown_idx))
    unknown_false_positive = np.sum((y_test != unknown_idx) & (y_pred == unknown_idx))
    unknown_false_negative = np.sum((y_test == unknown_idx) & (y_pred != unknown_idx))
    unknown_precision = unknown_true_positive / (unknown_true_positive + unknown_false_positive) if (unknown_true_positive + unknown_false_positive) > 0 else 0
    unknown_recall = unknown_true_positive / (unknown_true_positive + unknown_false_negative) if (unknown_true_positive + unknown_false_negative) > 0 else 0
    unknown_f1 = 2 * unknown_precision * unknown_recall / (unknown_precision + unknown_recall) if (unknown_precision + unknown_recall) > 0 else 0
    
    print(f"\n[可疑流量统计]")
    print(f"  测试集中真实可疑流量数: {unknown_count}")
    print(f"  模型预测为可疑流量数: {unknown_predicted}")
    print(f"  已知类别样本数: {known_total}")
    print(f"  已知类别正确分类数: {known_correct}")
    print(f"  已知类别分类准确率: {known_correct/known_total:.4f}" if known_total > 0 else "  已知类别分类准确率: N/A")
    print(f"\n[unknown类指标]")
    print(f"  True Positive(正确识别的可疑流量): {unknown_true_positive}")
    print(f"  False Positive(误报为可疑流量): {unknown_false_positive}")
    print(f"  False Negative(漏报的可疑流量): {unknown_false_negative}")
    print(f"  unknown精确率: {unknown_precision:.4f}")
    print(f"  unknown召回率: {unknown_recall:.4f}")
    print(f"  unknown F1: {unknown_f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    print(f"\n[混淆矩阵] {num_classes}x{num_classes}")

    metrics = {
        'test_acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
    }

    return metrics, y_pred, y_prob_matrix


# =============================================================================
# 4.5 阈值调优函数
# =============================================================================
def tune_threshold(model, X_val, y_val, device, num_classes):
    """
    在验证集上寻找最优的softmax概率阈值，用于识别unknown类别。
    
    参数:
        model: 训练好的模型
        X_val, y_val: 验证集特征和标签
        device: 当前设备
        num_classes: 评估类别数（24）
    
    返回:
        best_threshold: 最优阈值
    """
    model.eval()
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    with torch.no_grad():
        logits = model(X_val_tensor).cpu().numpy()
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob_23 = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        y_pred_23 = np.argmax(y_prob_23, axis=1)
        max_probs = np.max(y_prob_23, axis=1)
    
    # 遍历候选阈值
    thresholds = np.arange(0.3, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred = y_pred_23.copy()
        y_pred[max_probs < threshold] = num_classes - 1
        
        # 计算整体指标
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # 计算unknown类指标
        unknown_idx = num_classes - 1
        unknown_tp = np.sum((y_val == unknown_idx) & (y_pred == unknown_idx))
        unknown_fn = np.sum((y_val == unknown_idx) & (y_pred != unknown_idx))
        unknown_recall = unknown_tp / (unknown_tp + unknown_fn) if (unknown_tp + unknown_fn) > 0 else 0
        
        # 计算已知类别准确率
        known_mask = y_val != unknown_idx
        known_acc = accuracy_score(y_val[known_mask], y_pred[known_mask]) if np.sum(known_mask) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'f1': f1,
            'unknown_recall': unknown_recall,
            'known_acc': known_acc
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 选择最优阈值：平衡整体F1和unknown召回率
    # 使用加权得分：0.6 * f1 + 0.4 * unknown_recall
    results_df['score'] = 0.6 * results_df['f1'] + 0.4 * results_df['unknown_recall']
    best_idx = results_df['score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    
    print("\n阈值调优结果:")
    print(results_df[['threshold', 'accuracy', 'f1', 'unknown_recall', 'known_acc']].to_string(index=False))
    print(f"\n最优阈值: {best_threshold:.2f}")
    print(f"  整体F1: {results_df.loc[best_idx, 'f1']:.4f}")
    print(f"  unknown召回率: {results_df.loc[best_idx, 'unknown_recall']:.4f}")
    print(f"  已知类别准确率: {results_df.loc[best_idx, 'known_acc']:.4f}")
    
    # 绘制阈值-F1曲线
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['f1'], 'o-', label='整体F1', color='#1f77b4', linewidth=2)
    plt.plot(results_df['threshold'], results_df['unknown_recall'], 's-', label='unknown召回率', color='#ff7f0e', linewidth=2)
    plt.plot(results_df['threshold'], results_df['known_acc'], '^-', label='已知类别准确率', color='#2ca02c', linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'最优阈值={best_threshold:.2f}')
    
    plt.xlabel('阈值', fontsize=12, fontweight='bold')
    plt.ylabel('指标值', fontsize=12, fontweight='bold')
    plt.title('阈值调优曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(thresholds)
    plt.tight_layout()
    plt.savefig('results_cnn1d_threshold_tuning.png', dpi=300, bbox_inches='tight')
    print("阈值调优曲线图已保存: results_cnn1d_threshold_tuning.png")
    plt.close()
    
    return best_threshold


# =============================================================================
# 5. 结果可视化函数
# =============================================================================
def plot_results(y_test, y_pred, y_prob, history, num_classes, class_names=None):
    """
    绘制并保存结果图表：
        - 混淆矩阵热力图（含类别名称）
        - ROC曲线（多分类OvR）
        - 训练损失曲线和验证损失曲线
        - 验证准确率曲线
        - 每类攻击统计条形图（总数、正确数、错误数）
        - 误判分析图（每类被误判成什么）
        - 归一化混淆矩阵热力图
        - 每类精确率/召回率/F1柱状图
    参数:
        y_test: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率矩阵
        history: 训练历史字典
        num_classes: 类别数
        class_names: 类别名称列表（可选）
    """
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    # 图1: 混淆矩阵（单独大图）
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'fontsize': 8, 'fontweight': 'bold'},
                cbar_kws={'label': '样本数量'})
    plt.title(f'1D-CNN 混淆矩阵 ({num_classes} 类)', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('results_cnn1d_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵图已保存: results_cnn1d_confusion_matrix.png")
    plt.close()

    # 图2: ROC曲线（单独大图）
    plt.figure(figsize=(10, 8))
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=list(range(num_classes)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr',
                                  average='weighted', labels=list(range(num_classes)))
    except Exception:
        fpr, tpr = [0, 1], [0, 1]
        auc_score = float('nan')
    plt.plot(fpr, tpr, label=f'1D-CNN (AUC = {auc_score:.4f})', 
             color='#1f77b4', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类', linewidth=1.5)
    plt.xlabel('假正率 (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('真正率 (TPR)', fontsize=12, fontweight='bold')
    plt.title('ROC 曲线（多分类 OvR）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('results_cnn1d_roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC曲线图已保存: results_cnn1d_roc_curve.png")
    plt.close()

    # 图3: 训练历史曲线（损失和准确率）
    plt.figure(figsize=(16, 7))

    # 子图1: 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='训练损失', color='#1f77b4', 
             linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_losses'], label='验证损失', color='#ff7f0e', 
             linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss', fontsize=11, fontweight='bold')
    plt.title('训练过程 - 损失曲线', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accs'], label='验证准确率', color='#2ca02c', 
             linewidth=2, marker='^', markersize=4)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
    plt.title('训练过程 - 验证准确率曲线', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('results_cnn1d_training_history.png', dpi=300, bbox_inches='tight')
    print("训练过程图已保存: results_cnn1d_training_history.png")
    plt.close()

    # 图4: 每类攻击统计（总数、正确数、错误数）
    plt.figure(figsize=(20, 10))
    total_per_class = np.bincount(y_test, minlength=num_classes)
    correct_per_class = np.diag(cm)
    incorrect_per_class = total_per_class - correct_per_class

    x = np.arange(num_classes)
    width = 0.35

    bars_total = plt.bar(x - width/2, total_per_class, width, label='总数', 
                         color='#1f77b4', edgecolor='black', alpha=0.9)
    bars_correct = plt.bar(x + width/2, correct_per_class, width, label='正确', 
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
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('results_cnn1d_class_counts.png', dpi=300, bbox_inches='tight')
    print("每类攻击统计图已保存: results_cnn1d_class_counts.png")
    plt.close()

    # 图5: 误判分析（每类被误判成什么）
    plt.figure(figsize=(24, 16))
    total_per_class = np.bincount(y_test, minlength=num_classes)
    plot_idx = 1
    for true_class in range(num_classes):
        if total_per_class[true_class] == 0:
            continue
        row = cm[true_class]
        misclassified = row.copy()
        misclassified[true_class] = 0
        if misclassified.sum() == 0:
            continue

        mis_class_indices = np.where(misclassified > 0)[0]
        mis_class_counts = misclassified[mis_class_indices]
        mis_class_names = [class_names[i] for i in mis_class_indices]
        accuracy = row[true_class] / total_per_class[true_class] * 100

        plt.subplot(5, 5, plot_idx)
        bars = plt.barh(mis_class_names, mis_class_counts, 
                       color='#ff7f0e', edgecolor='black', alpha=0.9)
        plt.title(f'{class_names[true_class]}\n(准确率: {accuracy:.1f}%, 错误: {misclassified.sum()})', 
                  fontsize=10, fontweight='bold')
        plt.xlabel('误判数量', fontsize=9)
        plt.yticks(fontsize=8)
        plt.xlim(0, max(mis_class_counts) * 1.15)
        for i, v in enumerate(mis_class_counts):
            plt.text(v + max(mis_class_counts)*0.02, i, f'{v}', 
                     va='center', fontsize=8, fontweight='bold')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig('results_cnn1d_misclassification.png', dpi=300, bbox_inches='tight')
    print("误判分析图已保存: results_cnn1d_misclassification.png")
    plt.close()

    # 图6: 归一化混淆矩阵（按行归一化）
    plt.figure(figsize=(18, 14))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'fontsize': 8, 'fontweight': 'bold'},
                cbar_kws={'label': '比例', 'shrink': 0.8})
    plt.title(f'1D-CNN 归一化混淆矩阵 ({num_classes} 类)', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('results_cnn1d_confusion_normalized.png', dpi=300, bbox_inches='tight')
    print("归一化混淆矩阵图已保存: results_cnn1d_confusion_normalized.png")
    plt.close()

    # 图6: 每类精确率、召回率、F1
    from sklearn.metrics import precision_score, recall_score, f1_score

    precisions = precision_score(y_test, y_pred, average=None, zero_division=0,
                                 labels=list(range(num_classes)))
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0,
                           labels=list(range(num_classes)))
    f1_scores = f1_score(y_test, y_pred, average=None, zero_division=0,
                         labels=list(range(num_classes)))

    plt.figure(figsize=(20, 10))
    x = np.arange(num_classes)
    width = 0.25

    plt.bar(x - width, precisions, width, label='精确率', color='#1f77b4', edgecolor='black')
    plt.bar(x, recalls, width, label='召回率', color='#ff7f0e', edgecolor='black')
    plt.bar(x + width, f1_scores, width, label='F1', color='#2ca02c', edgecolor='black')

    for i in range(num_classes):
        plt.text(i - width, precisions[i] + 0.02, f'{precisions[i]:.2f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.text(i, recalls[i] + 0.02, f'{recalls[i]:.2f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.text(i + width, f1_scores[i] + 0.02, f'{f1_scores[i]:.2f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('攻击类型', fontsize=12, fontweight='bold')
    plt.ylabel('指标值', fontsize=12, fontweight='bold')
    plt.title('每类攻击精确率、召回率、F1-Score', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('results_cnn1d_per_class_metrics.png', dpi=300, bbox_inches='tight')
    print("每类指标图已保存: results_cnn1d_per_class_metrics.png")
    plt.close()


# =============================================================================
# 6. 结果保存函数
# =============================================================================
def save_results(model, metrics, train_time, history):
    """
    保存训练好的模型权重、评估指标CSV、训练历史CSV。
    参数:
        model: 训练好的模型
        metrics: 评估指标字典
        train_time: 训练耗时
        history: 训练历史字典
    """
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    # 保存模型权重（推荐保存state_dict，便于加载）
    torch.save(model.state_dict(), 'model_cnn1d.pth')
    print("模型已保存: model_cnn1d.pth")

    # 保存评估指标为CSV
    results_df = pd.DataFrame([metrics])
    results_df['train_time'] = train_time
    results_df.to_csv('results_cnn1d_metrics.csv', index=False)
    print("评估指标已保存: results_cnn1d_metrics.csv")

    # 保存训练历史为CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv('results_cnn1d_history.csv', index=False)
    print("训练历史已保存: results_cnn1d_history.csv")


# =============================================================================
# 7. 主函数：执行完整流程
# =============================================================================
def main():
    """
    主函数，按顺序执行：
        1. 加载预处理数据
        2. 训练模型
        3. 评估模型（含阈值调优）
        4. 可视化结果
        5. 保存模型和结果
    返回:
        model: 训练好的模型
        metrics: 评估指标
        history: 训练历史
    """
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes, class_names = load_preprocessed_data()
    
    # model_num_classes=23（模型输出维度），num_classes=24（评估类别数，含unknown）
    model_num_classes = 23
    model, train_time, history, device = train_cnn1d(X_train, X_val, y_train, y_val, model_num_classes)
    
    # 在验证集上调优阈值
    print("\n" + "=" * 60)
    print("阈值调优（在验证集上）")
    print("=" * 60)
    
    best_threshold = tune_threshold(model, X_val, y_val, device, num_classes)
    print(f"最优阈值: {best_threshold}")
    
    # 使用最优阈值评估测试集
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, device, num_classes, threshold=best_threshold)
    plot_results(y_test, y_pred, y_prob, history, num_classes, class_names)
    save_results(model, metrics, train_time, history)

    print("\n" + "=" * 60)
    print("1D-CNN 模型训练完成!")
    print("=" * 60)

    return model, metrics, history


# =============================================================================
# 程序入口
# =============================================================================
if __name__ == '__main__':
    model, metrics, history = main()