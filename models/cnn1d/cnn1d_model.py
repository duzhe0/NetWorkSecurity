# =============================================================================
# 文件名: cnn1d_model.py
# 描述: 基于PyTorch实现的1D卷积神经网络（CNN1D），用于KDD CUP 99数据集的多分类任务。
#       将120维特征视为长度为120的单通道时序信号，通过多层1D卷积+全局平均池化进行分类。
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
        input_dim (int): 输入特征维度（序列长度）
        num_classes (int): 输出类别数
        conv_channels (list): 各卷积层的输出通道数列表，例如[64,128,256]
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
    """
    print("=" * 60)
    print("1D-CNN 卷积神经网络训练")
    print("=" * 60)

    # 使用脚本所在目录的绝对路径推导项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "Train") + os.sep

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

    # 计算类别数（取三个数据集中的最大值+1，确保覆盖所有类别）
    num_classes = int(max(df_train['label_multiclass_encoded'].max(),
                          df_val['label_multiclass_encoded'].max(),
                          df_test['label_multiclass_encoded'].max()) + 1)

    # 尝试从文本文件加载类别名称（用于信息显示）
    class_file = os.path.join(data_dir, "encoder_multiclass_23_classes.txt")
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        num_classes = len(class_names)  # 以文件中的类别数为准
    print(f"\n【{num_classes} 分类任务】共 {num_classes} 个类别 (normal + {num_classes - 1} 种攻击)")
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"测试集样本数: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes


# =============================================================================
# 3. 模型训练函数
# =============================================================================
def train_cnn1d(X_train, X_val, y_train, y_val, num_classes, epochs=50, batch_size=64):
    """
    训练1D-CNN模型（多分类），在验证集上监控性能，并记录损失和准确率历史。
    参数:
        X_train, y_train: 训练集特征和标签
        X_val, y_val: 验证集特征和标签
        num_classes: 类别数
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
    print(f"  输出层: {num_classes} 维 (softmax via CrossEntropyLoss)")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")

    # 选择设备（GPU优先）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 初始化模型、损失函数和优化器
    model = CNN1D(input_dim, num_classes, conv_channels, kernel_size, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失（内部包含softmax）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将numpy数据转为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # 创建DataLoader用于训练集批处理
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
            val_acc = accuracy_score(y_val, y_val_pred)
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
def evaluate_model(model, X_test, y_test, device, num_classes):
    """
    在测试集上评估模型性能，计算准确率、精确率、召回率、F1、AUC等指标，
    并返回混淆矩阵、预测标签和预测概率。
    参数:
        model: 训练好的模型
        X_test, y_test: 测试集特征和标签
        device: 当前设备
        num_classes: 类别数
    返回:
        metrics: 字典，包含各项评估指标
        y_pred: 预测类别标签
        y_prob_matrix: 每个样本的预测概率矩阵 (n_samples, num_classes)
    """
    print("\n" + "=" * 60)
    print("模型评估 (测试集) - 多分类")
    print("=" * 60)

    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        # 获取logits并转为numpy
        logits = model(X_test_tensor).cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        # 计算softmax概率（数值稳定性：减去最大值）
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob_matrix = exp_logits / exp_logits.sum(axis=1, keepdims=True)

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

    print(f"\n[测试集] 评估结果 (多分类, {num_classes} 类):")
    print(f"  准确率 (Accuracy): {test_acc:.4f}")
    print(f"  精确率 (Precision weighted): {test_precision:.4f}")
    print(f"  召回率 (Recall weighted): {test_recall:.4f}")
    print(f"  F1-Score weighted: {test_f1:.4f}")
    print(f"  AUC (weighted OvR): {test_auc:.4f}")

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
# 5. 结果可视化函数
# =============================================================================
def plot_results(y_test, y_pred, y_prob, history, num_classes):
    """
    绘制并保存结果图表：
        - 混淆矩阵热力图
        - ROC曲线（多分类OvR）
        - 训练损失曲线和验证损失曲线
        - 验证准确率曲线
    参数:
        y_test: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率矩阵
        history: 训练历史字典
        num_classes: 类别数
    """
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    # 图1: 混淆矩阵 + ROC曲线
    plt.figure(figsize=(14, 6))

    # 子图1: 混淆矩阵
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'1D-CNN 混淆矩阵 ({num_classes} 类)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 子图2: ROC曲线（多分类OvR，将概率矩阵展平计算）
    plt.subplot(1, 2, 2)
    try:
        from sklearn.preprocessing import label_binarize
        # 将标签二值化，用于计算ROC
        y_bin = label_binarize(y_test, classes=list(range(num_classes)))
        # 计算平均ROC曲线（展平所有类别的FPR和TPR）
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        # 计算AUC（直接使用之前的auc值，但这里用roc_auc_score再算一次作为展示）
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr',
                                  average='weighted', labels=list(range(num_classes)))
    except Exception:
        # 若计算失败，绘制一条对角线
        fpr, tpr = [0, 1], [0, 1]
        auc_score = float('nan')
    plt.plot(fpr, tpr, label=f'1D-CNN (AUC = {auc_score:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线（多分类 OvR）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_cnn1d_evaluation.png', dpi=150, bbox_inches='tight')
    print("混淆矩阵和ROC曲线已保存: results_cnn1d_evaluation.png")
    plt.close()

    # 图2: 训练历史曲线（损失和准确率）
    plt.figure(figsize=(12, 5))

    # 子图1: 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='训练损失', color='blue')
    plt.plot(history['val_losses'], label='验证损失', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程 - 损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accs'], label='验证准确率', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练过程 - 验证准确率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_cnn1d_training_history.png', dpi=150, bbox_inches='tight')
    print("训练过程图已保存: results_cnn1d_training_history.png")
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
        3. 评估模型
        4. 可视化结果
        5. 保存模型和结果
    返回:
        model: 训练好的模型
        metrics: 评估指标
        history: 训练历史
    """
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes = load_preprocessed_data()
    model, train_time, history, device = train_cnn1d(X_train, X_val, y_train, y_val, num_classes)
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, device, num_classes)
    plot_results(y_test, y_pred, y_prob, history, num_classes)
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


    