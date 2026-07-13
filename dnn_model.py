import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DNN(nn.Module):
    """深度神经网络模型"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(DNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_preprocessed_data():
    """加载预处理后的数据"""
    print("=" * 60)
    print("DNN 深度神经网络训练")
    print("=" * 60)

    data_path = "Train/KDDTrain_preprocessed.csv"
    df = pd.read_csv(data_path)
    print(f"\n[数据加载] 数据集形状: {df.shape}")

    # 特征列：排除标签列和辅助列
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y_binary = df['label_binary'].values  # 二分类标签

    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")

    return X, y_binary, feature_cols


def split_data(X, y, test_size=0.3):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n[数据划分] 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def prepare_torch_data(X_train, X_test, y_train, y_test, batch_size=64):
    """准备PyTorch数据"""
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test_tensor, y_test_tensor


def train_dnn(X_train, X_test, y_train, y_test, epochs=50, batch_size=64):
    """训练DNN模型"""
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    input_dim = X_train.shape[1]

    # 模型参数
    hidden_dims = [256, 128, 64]
    dropout_rate = 0.3
    learning_rate = 0.001

    print(f"\nDNN网络结构:")
    print(f"  输入层: {input_dim} 维")
    print(f"  隐藏层: {hidden_dims}")
    print(f"  Dropout率: {dropout_rate}")
    print(f"  输出层: 1 维 (sigmoid)")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = DNN(input_dim, hidden_dims, dropout_rate).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 准备数据
    train_loader, test_loader, X_test_tensor, y_test_tensor = prepare_torch_data(
        X_train, X_test, y_train, y_test, batch_size
    )

    # 训练
    print("\n开始训练...")
    start_time = time.time()

    train_losses = []
    test_losses = []
    test_accs = []

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

        # 测试集评估
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test_tensor.to(device)
            y_test_dev = y_test_tensor.to(device)
            test_outputs = model(X_test_dev)
            test_loss = criterion(test_outputs, y_test_dev).item()
            test_losses.append(test_loss)

            y_pred = (test_outputs.cpu().numpy() > 0.5).astype(int)
            test_acc = accuracy_score(y_test, y_pred.flatten())
            test_accs.append(test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Acc: {test_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.2f} 秒")

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

    return model, train_time, history, device


def evaluate_model(model, X_test, y_test, device):
    """评估模型性能"""
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y_prob = model(X_test_tensor).cpu().numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)

    # 测试集评估
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n[测试集] 评估结果:")
    print(f"  准确率 (Accuracy): {test_acc:.4f}")
    print(f"  精确率 (Precision): {test_precision:.4f}")
    print(f"  召回率 (Recall): {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[混淆矩阵]")
    print(cm)

    # 分类报告
    print(f"\n[分类报告]")
    print(classification_report(y_test, y_pred, target_names=['normal', 'attack']))

    # 评估指标字典
    metrics = {
        'test_acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
    }

    return metrics, y_pred, y_prob


def plot_results(y_test, y_pred, y_prob, history):
    """可视化结果"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    # 1. 混淆矩阵可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['normal', 'attack'],
                yticklabels=['normal', 'attack'])
    plt.title('DNN 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 2. ROC曲线
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'DNN (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_dnn_evaluation.png', dpi=150, bbox_inches='tight')
    print("混淆矩阵和ROC曲线已保存: results_dnn_evaluation.png")
    plt.close()

    # 3. 训练过程可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='训练损失', color='purple')
    plt.plot(history['test_losses'], label='测试损失', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程 - 损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['test_accs'], label='测试准确率', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练过程 - 准确率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_dnn_training_history.png', dpi=150, bbox_inches='tight')
    print("训练过程图已保存: results_dnn_training_history.png")
    plt.close()


def save_results(model, metrics, train_time, history):
    """保存模型和结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    # 保存PyTorch模型
    torch.save(model.state_dict(), 'model_dnn.pth')
    print("模型已保存: model_dnn.pth")

    # 保存评估结果
    results_df = pd.DataFrame([metrics])
    results_df['train_time'] = train_time
    results_df.to_csv('results_dnn_metrics.csv', index=False)
    print("评估指标已保存: results_dnn_metrics.csv")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv('results_dnn_history.csv', index=False)
    print("训练历史已保存: results_dnn_history.csv")


def main():
    """主函数"""
    # 1. 加载预处理数据
    X, y, feature_cols = load_preprocessed_data()

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. 训练DNN模型
    model, train_time, history, device = train_dnn(X_train, X_test, y_train, y_test)

    # 4. 评估模型
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, device)

    # 5. 可视化结果
    plot_results(y_test, y_pred, y_prob, history)

    # 6. 保存结果
    save_results(model, metrics, train_time, history)

    print("\n" + "=" * 60)
    print("DNN 模型训练完成!")
    print("=" * 60)

    return model, metrics, history


if __name__ == '__main__':
    model, metrics, history = main()