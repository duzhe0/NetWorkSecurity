import pandas as pd
import numpy as np
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
    """从三个独立CSV文件加载训练集、验证集、测试集"""
    print("=" * 60)
    print("DNN 深度神经网络训练")
    print("=" * 60)

    data_dir = "../../Train/"

    # 分别加载训练集、验证集、测试集
    df_train = pd.read_csv(data_dir + "KDDTrain_preprocessed_train.csv")
    df_val   = pd.read_csv(data_dir + "KDDTrain_preprocessed_val.csv")
    df_test  = pd.read_csv(data_dir + "KDDTrain_preprocessed_test.csv")

    print(f"\n[数据加载] 训练集形状: {df_train.shape}")
    print(f"[数据加载] 验证集形状: {df_val.shape}")
    print(f"[数据加载] 测试集形状: {df_test.shape}")

    # 特征列：排除标签列和辅助列
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train['label_binary'].values

    X_val = df_val[feature_cols].values
    y_val = df_val['label_binary'].values

    X_test = df_test[feature_cols].values
    y_test = df_test['label_binary'].values

    print(f"\n特征数量: {len(feature_cols)}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"测试集样本数: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_dnn(X_train, X_val, y_train, y_val, epochs=50, batch_size=64):
    """训练DNN模型，在验证集上监控"""
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

    # 准备训练和验证数据
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练
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

        # 验证集评估
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val_tensor.to(device)
            y_val_dev = y_val_tensor.to(device)
            val_outputs = model(X_val_dev)
            val_loss = criterion(val_outputs, y_val_dev).item()
            val_losses.append(val_loss)

            y_val_pred = (val_outputs.cpu().numpy() > 0.5).astype(int)
            val_acc = accuracy_score(y_val, y_val_pred.flatten())
            val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.2f} 秒")

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }

    return model, train_time, history, device


def evaluate_model(model, X_test, y_test, device):
    """在测试集上评估模型性能"""
    print("\n" + "=" * 60)
    print("模型评估 (测试集)")
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
    # 1. 加载预处理数据（训练集、验证集、测试集）
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_preprocessed_data()

    # 2. 训练DNN模型（在验证集上监控）
    model, train_time, history, device = train_dnn(X_train, X_val, y_train, y_val)

    # 3. 在测试集上做最终评估（仅一次）
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, device)

    # 4. 可视化结果
    plot_results(y_test, y_pred, y_prob, history)

    # 5. 保存结果
    save_results(model, metrics, train_time, history)

    print("\n" + "=" * 60)
    print("DNN 模型训练完成!")
    print("=" * 60)

    return model, metrics, history


if __name__ == '__main__':
    model, metrics, history = main()
