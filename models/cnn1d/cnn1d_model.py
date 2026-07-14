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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CNN1D(nn.Module):
    """1D 卷积神经网络模型（多分类版：23 类 = normal + 22 种攻击）
    将 117 维特征视为长度为 117 的单通道序列，使用多层 1D 卷积 + 全局平均池化进行分类。
    """
    def __init__(self, input_dim, num_classes, conv_channels=[64, 128, 256],
                 kernel_size=3, dropout_rate=0.3):
        super(CNN1D, self).__init__()

        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # 自适应平均池化，确保不同输入长度都能得到固定维度输出
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)  # (batch, channels)
        return self.classifier(x)


def load_preprocessed_data():
    """从三个独立CSV文件加载训练集、验证集、测试集"""
    print("=" * 60)
    print("1D-CNN 卷积神经网络训练")
    print("=" * 60)

    data_dir = "../../Train/"

    df_train = pd.read_csv(data_dir + "KDDTrain_preprocessed_train.csv")
    df_val   = pd.read_csv(data_dir + "KDDTrain_preprocessed_val.csv")
    df_test  = pd.read_csv(data_dir + "KDDTrain_preprocessed_test.csv")

    print(f"\n[数据加载] 训练集形状: {df_train.shape}")
    print(f"[数据加载] 验证集形状: {df_val.shape}")
    print(f"[数据加载] 测试集形状: {df_test.shape}")

    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded',
                    'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train['label_multiclass_encoded'].values.astype(np.int64)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val['label_multiclass_encoded'].values.astype(np.int64)

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test['label_multiclass_encoded'].values.astype(np.int64)

    num_classes = int(max(df_train['label_multiclass_encoded'].max(),
                          df_val['label_multiclass_encoded'].max(),
                          df_test['label_multiclass_encoded'].max()) + 1)

    class_file = os.path.join(data_dir, "encoder_multiclass_23_classes.txt")
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        num_classes = len(class_names)
    print(f"\n【{num_classes} 分类任务】共 {num_classes} 个类别 (normal + {num_classes - 1} 种攻击)")
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"测试集样本数: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes


def train_cnn1d(X_train, X_val, y_train, y_val, num_classes, epochs=50, batch_size=64):
    """训练 1D-CNN 模型（多分类），在验证集上监控"""
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    input_dim = X_train.shape[1]

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = CNN1D(input_dim, num_classes, conv_channels, kernel_size, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n开始训练...")
    start_time = time.time()

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            X_val_dev = X_val_tensor.to(device)
            y_val_dev = y_val_tensor.to(device)
            val_outputs = model(X_val_dev)
            val_loss = criterion(val_outputs, y_val_dev).item()
            val_losses.append(val_loss)

            y_val_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, y_val_pred)
            val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.2f} 秒")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }

    return model, train_time, history, device


def evaluate_model(model, X_test, y_test, device, num_classes):
    """在测试集上评估模型性能（多分类）"""
    print("\n" + "=" * 60)
    print("模型评估 (测试集) - 多分类")
    print("=" * 60)

    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor).cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob_matrix = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_prob_matrix, multi_class='ovr',
                                 average='weighted', labels=list(range(num_classes)))
    except Exception:
        test_auc = float('nan')

    print(f"\n[测试集] 评估结果 (多分类, {num_classes} 类):")
    print(f"  准确率 (Accuracy): {test_acc:.4f}")
    print(f"  精确率 (Precision weighted): {test_precision:.4f}")
    print(f"  召回率 (Recall weighted): {test_recall:.4f}")
    print(f"  F1-Score weighted: {test_f1:.4f}")
    print(f"  AUC (weighted OvR): {test_auc:.4f}")

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


def plot_results(y_test, y_pred, y_prob, history, num_classes):
    """可视化结果（多分类）"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'1D-CNN 混淆矩阵 ({num_classes} 类)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    plt.subplot(1, 2, 2)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=list(range(num_classes)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr',
                                   average='weighted', labels=list(range(num_classes)))
    except Exception:
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

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='训练损失', color='blue')
    plt.plot(history['val_losses'], label='验证损失', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程 - 损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

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


def save_results(model, metrics, train_time, history):
    """保存模型和结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    torch.save(model.state_dict(), 'model_cnn1d.pth')
    print("模型已保存: model_cnn1d.pth")

    results_df = pd.DataFrame([metrics])
    results_df['train_time'] = train_time
    results_df.to_csv('results_cnn1d_metrics.csv', index=False)
    print("评估指标已保存: results_cnn1d_metrics.csv")

    history_df = pd.DataFrame(history)
    history_df.to_csv('results_cnn1d_history.csv', index=False)
    print("训练历史已保存: results_cnn1d_history.csv")


def main():
    """主函数"""
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes = load_preprocessed_data()
    model, train_time, history, device = train_cnn1d(X_train, X_val, y_train, y_val, num_classes)
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, device, num_classes)
    plot_results(y_test, y_pred, y_prob, history, num_classes)
    save_results(model, metrics, train_time, history)

    print("\n" + "=" * 60)
    print("1D-CNN 模型训练完成!")
    print("=" * 60)

    return model, metrics, history


if __name__ == '__main__':
    model, metrics, history = main()
