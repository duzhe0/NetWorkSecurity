import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
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
import sys

# 确保可以从 models.losses 导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.losses import FocalLoss
from models.embedding_utils import ServiceEmbeddingModel, compute_embedded_input_dim, SERVICE_EMBEDDING_DIM

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


class DNN(nn.Module):
    """深度神经网络模型（多分类版：22 类 = normal + 21 种攻击）"""
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout_rate=0.3):
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

        # 多分类输出层（num_classes 个 logits，不使用 sigmoid）
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_preprocessed_data():
    """从三个独立CSV文件加载训练集、验证集、测试集"""
    print("=" * 60)
    print("DNN 深度神经网络训练")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "Train") + os.sep

    # 分别加载训练集、验证集、测试集
    df_train = pd.read_csv(data_dir + "KDDTrain_preprocessed_train.csv")
    df_val   = pd.read_csv(data_dir + "KDDTrain_preprocessed_val.csv")
    df_test  = pd.read_csv(data_dir + "KDDTrain_preprocessed_test.csv")

    print(f"\n[数据加载] 训练集形状: {df_train.shape}")
    print(f"[数据加载] 验证集形状: {df_val.shape}")
    print(f"[数据加载] 测试集形状: {df_test.shape}")

    # 特征列：排除所有标签列和辅助列（含新增的 23 分类标签列）
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

    # 从文件获取真正的总类别数（可能因 split 有空隙，但总共 23 类）
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


def train_dnn(X_train, X_val, y_train, y_val, num_classes, feature_cols, epochs=50, batch_size=64,
              weight_scheme='sqrt'):
    """训练DNN模型（多分类），在验证集上监控
    
    weight_scheme: 'none' | 'balanced' | 'sqrt' | 'log1p'
    """
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    input_dim = X_train.shape[1]
    # 计算 Entity Embedding 后的实际输入维度
    embedded_dim = compute_embedded_input_dim(feature_cols, SERVICE_EMBEDDING_DIM)

    # 模型参数
    hidden_dims = [256, 128, 64]
    dropout_rate = 0.3
    learning_rate = 0.001

    print(f"\nDNN网络结构（多分类）:")
    print(f"  原始特征: {input_dim} 维（含 68 列 service OneHot + 1 列 service_encoded）")
    print(f"  Embedding后: {embedded_dim} 维（service 68维OneHot → {SERVICE_EMBEDDING_DIM}维Embedding）")
    print(f"  隐藏层: {hidden_dims}")
    print(f"  Dropout率: {dropout_rate}")
    print(f"  输出层: {num_classes} 维 (softmax via FocalLoss, gamma=2.0)")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    base_model = DNN(embedded_dim, num_classes, hidden_dims, dropout_rate)
    model = ServiceEmbeddingModel(base_model, vocab_size=69, feature_cols=feature_cols)
    model = model.to(device)

    # 类别加权
    if weight_scheme == 'none':
        class_weights_tensor = None
        print(f"\n[类别加权] 方案: none（无权重）")
    else:
        # 只对训练集中实际存在的类别计算权重，缺失类别权重=0
        unique_train = np.unique(y_train)
        balanced_full = np.zeros(num_classes)
        present_weights = compute_class_weight('balanced', classes=unique_train, y=y_train)
        for i, cls in enumerate(unique_train):
            balanced_full[cls] = present_weights[i]

        if weight_scheme == 'balanced':
            class_weights = balanced_full
        elif weight_scheme == 'sqrt':
            class_weights = np.sqrt(balanced_full)
        elif weight_scheme == 'log1p':
            class_weights = np.log1p(balanced_full)
        else:
            raise ValueError(f"未知的 weight_scheme: {weight_scheme}")
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        # 日志仅显示非零权重
        nonzero = class_weights[class_weights > 0]
        print(f"\n[类别加权] 方案: {weight_scheme}")
        print(f"[类别加权] 非零权重范围: {nonzero.min():.4f} ~ {nonzero.max():.4f}")
        if np.any(class_weights == 0):
            zero_classes = np.where(class_weights == 0)[0]
            print(f"[类别加权] 训练集缺失类别（权重=0）: {list(zero_classes)}")

    # 多分类：使用 Focal Loss（类别加权 + 难样本聚焦）
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 准备训练和验证数据
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

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

            # 多分类：取 argmax 作为预测
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

    # 保存训练历史
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
        # 多分类：取 argmax
        y_pred = np.argmax(logits, axis=1)
        # 概率矩阵（softmax）
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob_matrix = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # 测试集评估（多分类使用 weighted average）
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

    # 混淆矩阵（多分类 N×N）
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    print(f"\n[混淆矩阵] {num_classes}x{num_classes}")

    # 评估指标字典
    metrics = {
        'test_acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
    }

    return metrics, y_pred, y_prob_matrix


def evaluate_external_test_dnn(model, num_classes, class_names, device):
    """在外部 train_test 集上评估 DNN 模型（零信息泄露）"""
    print("\n" + "=" * 60)
    print("外部评估 (train_test)")
    print("=" * 60)

    import joblib as _joblib
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 加载预处理工具
    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_robust.pkl')
    service_le_path = os.path.join(base_dir, 'Train', 'encoder_service_embedding.pkl')
    ohe = _joblib.load(ohe_path)
    scaler = _joblib.load(scaler_path)
    service_le = _joblib.load(service_le_path)
    UNK_SERVICE_IDX = len(service_le.classes_)

    # 加载 train_test 原始数据
    test_path = os.path.join(base_dir, 'Train', 'train_test')
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"[外部数据] train_test 原始: {df_test.shape}")

    # 生成标签：未知攻击 → 最后一类（unknown_attack）
    if class_names:
        unknown_idx = len(class_names) - 1
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        df_test['label_multiclass_encoded'] = df_test['label'].map(label_to_idx).fillna(unknown_idx).astype(int)
    else:
        df_test['label_multiclass_encoded'] = 0

    y_test_orig = df_test['label_multiclass_encoded'].values

    # Service 安全编码（OOV → UNK）
    known_services = set(service_le.classes_)
    df_test['service_encoded'] = df_test['service'].apply(
        lambda x: service_le.transform([x])[0] if x in known_services else UNK_SERVICE_IDX
    )

    # 从训练集获取特征列顺序
    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    df_train_sample = pd.read_csv(train_csv_path, nrows=1)
    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                    'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]

    # One-Hot 编码（仅 transform）
    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
    df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
    df_test_enc = pd.concat([df_rest, df_ohe], axis=1)

    # 补齐训练集有的列
    for col in feature_cols:
        if col not in df_test_enc.columns:
            df_test_enc[col] = 0.0
    df_test_enc = df_test_enc[feature_cols]

    # 标准化（仅 transform）
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
    print(f"[外部数据] 预处理完成，特征矩阵: {X_test.shape}")

    # DNN 标签恒等映射（模型输出 24 维，未知攻击已在上游映射为 23）
    y_test = y_test_orig.copy()
    valid_mask = y_test >= 0
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[外部数据] 剔除 {n_dropped} 个非法标签样本（不应出现）")

    # 预测
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy()

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)

    y_true_v = y_test[valid_mask]
    y_pred_v = y_pred[valid_mask]
    y_prob_v = y_prob[valid_mask]

    ext_acc = accuracy_score(y_true_v, y_pred_v)
    ext_f1 = f1_score(y_true_v, y_pred_v, average='weighted', zero_division=0)

    print(f"[外部评估] Acc: {ext_acc:.4f}, F1: {ext_f1:.4f}")

    ext_cm = confusion_matrix(y_true_v, y_pred_v, labels=list(range(num_classes)))
    return {'ext_acc': ext_acc, 'ext_f1': ext_f1}, ext_cm


def plot_results(y_test, y_pred, y_prob, history, num_classes):
    """可视化结果（多分类）"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    # 1. 混淆矩阵可视化（多分类 N×N）
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f'DNN 混淆矩阵 ({num_classes} 类)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 2. 多分类 ROC（macro OvR，使用概率矩阵）
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
    plt.plot(fpr, tpr, label=f'DNN (AUC = {auc_score:.4f})', color='purple')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线（多分类 OvR）')
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
    """主函数：多方案对比 + 双轨评估"""
    # 1. 加载预处理数据（训练集、验证集、测试集）
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes = load_preprocessed_data()

    # 加载类别名
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    class_file = os.path.join(base_dir, 'Train', 'encoder_multiclass_23_classes.txt')
    class_names = []
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]

    # 方案列表
    schemes = ['none', 'balanced', 'sqrt', 'log1p']

    all_results = []

    print("\n" + "=" * 70)
    print("DNN 多方案类别加权对比实验")
    print("=" * 70)

    for scheme in schemes:
        print("\n" + "#" * 70)
        print(f"# 方案: {scheme}")
        print("#" * 70)

        # 2. 训练
        model, train_time, history, device = train_dnn(
            X_train, X_val, y_train, y_val, num_classes, feature_cols,
            weight_scheme=scheme)

        # 3. 内部测试集评估
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, device, num_classes)
        np.save(f'cm_internal_{scheme}.npy', metrics['confusion_matrix'])

        # 4. 外部 train_test 评估
        ext_metrics, ext_cm = evaluate_external_test_dnn(model, num_classes, class_names, device)
        np.save(f'cm_external_{scheme}.npy', ext_cm)

        # 5. 保存模型（按方案命名）
        torch.save(model.state_dict(), f'model_dnn_{scheme}.pth')
        print(f"[保存] model_dnn_{scheme}.pth")

        # 汇总记录
        result = {
            'scheme': scheme,
            'internal_acc': metrics['test_acc'],
            'internal_f1': metrics['f1'],
            'internal_auc': metrics['auc'],
            'external_acc': ext_metrics['ext_acc'],
            'external_f1': ext_metrics['ext_f1'],
            'train_time': train_time
        }
        all_results.append(result)

    # 打印汇总表
    print("\n" + "=" * 90)
    print("DNN 多方案双轨评估汇总")
    print("=" * 90)
    print(f"{'方案':<12} {'内部Acc':>10} {'内部F1':>10} {'内部AUC':>10} {'外部Acc':>10} {'外部F1':>10} {'耗时(s)':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['scheme']:<12} {r['internal_acc']:>10.4f} {r['internal_f1']:>10.4f} "
              f"{r['internal_auc']:>10.4f} {r['external_acc']:>10.4f} {r['external_f1']:>10.4f} "
              f"{r['train_time']:>8.1f}")

    # 保存汇总 CSV
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv('results_dnn_schemes_summary.csv', index=False)
    print("\n[保存] 汇总表已保存: results_dnn_schemes_summary.csv")

    print("\n" + "=" * 60)
    print("DNN 多方案对比完成!")
    print("=" * 60)

    return all_results


if __name__ == '__main__':
    all_results = main()
