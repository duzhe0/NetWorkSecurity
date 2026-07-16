"""
两阶段分类模型：
  Stage 1: 二分类（normal vs attack）
  Stage 2: 多分类（22种攻击），低置信 → unknown_attack
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.losses import FocalLoss
from models.embedding_utils import ServiceEmbeddingModel, compute_embedded_input_dim, SERVICE_EMBEDDING_DIM

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_dir = os.path.join(base_dir, 'Train')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Stage 1: Binary Classifier ====================

class BinaryDNN(nn.Module):
    """Stage 1: normal vs attack"""
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_two_stage_data():
    """加载预处理数据，分别准备 Stage 1 和 Stage 2 的标签"""
    # 加载类别名
    with open(os.path.join(train_dir, 'encoder_multiclass_23_classes.txt')) as f:
        class_names = [l.strip() for l in f if l.strip()]

    # 加载数据
    df_train = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'))
    df_val = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_val.csv'))
    df_test = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_test.csv'))

    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                    'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Stage 1 labels: binary (0=normal, 1=attack)
    y1_train = df_train['label_binary'].values.astype(np.int64)
    y1_val = df_val['label_binary'].values.astype(np.int64)
    y1_test = df_test['label_binary'].values.astype(np.int64)

    # Stage 2 labels: attack-only remapped to 0..21
    mc_train = df_train['label_multiclass_encoded'].values.astype(np.int64)
    mc_val = df_val['label_multiclass_encoded'].values.astype(np.int64)
    mc_test = df_test['label_multiclass_encoded'].values.astype(np.int64)

    # Build remapping: original label → 0..22 (22 attack types + 1 "not_attack" for normal)
    # 0-21 = specific attack types, 22 = not_attack (normal misclassified by Stage 1)
    orig_attack_labels = sorted([c for c in range(24) if c != 11 and c != 23])  # 22 attack classes
    s2_mapping = {orig: new for new, orig in enumerate(orig_attack_labels)}
    s2_mapping[11] = 22  # normal → "not_attack" class
    s2_reverse = {new: orig for new, orig in enumerate(orig_attack_labels)}
    s2_reverse[22] = 11  # not_attack → normal

    def remap(y_orig):
        """Map original labels to Stage 2 labels (0-22)"""
        y_new = np.full_like(y_orig, -1)
        for orig, new in s2_mapping.items():
            y_new[y_orig == orig] = new
        return y_new

    # Stage 2: all data except unknown_attack (include normal as "not_attack" class)
    def filter_for_s2(X, y_orig):
        mask = (y_orig != 23)  # exclude only unknown_attack
        return X[mask], remap(y_orig)[mask], mask

    X2_train, y2_train, m2_train = filter_for_s2(X_train, mc_train)
    X2_val, y2_val, m2_val = filter_for_s2(X_val, mc_val)
    X2_test, y2_test, m2_test = filter_for_s2(X_test, mc_test)

    return {
        'feature_cols': feature_cols,
        'class_names': class_names,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y1_train': y1_train, 'y1_val': y1_val, 'y1_test': y1_test,
        'X2_train': X2_train, 'X2_val': X2_val, 'X2_test': X2_test,
        'y2_train': y2_train, 'y2_val': y2_val, 'y2_test': y2_test,
        's2_reverse': s2_reverse,
        'mc_train': mc_train, 'mc_val': mc_val, 'mc_test': mc_test,
    }


def train_stage1(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=128):
    """训练 Stage 1 二分类"""
    # 类别权重（处理 imbalance）
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    w0 = len(y_train) / (2 * n_normal)
    w1 = len(y_train) / (2 * n_attack)
    class_weights = torch.tensor([w0, w1], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.LongTensor(y_val).to(device)

    best_acc = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for i in range(0, len(Xt), batch_size):
            bx = Xt[i:i+batch_size]
            by = yt[i:i+batch_size]
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()
        scheduler.step(1 - val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Stage1 Epoch {epoch:2d}/{epochs} - Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return best_acc


def train_stage2(model, X_train, y_train, X_val, y_val, num_classes=22, epochs=50, batch_size=64):
    """训练 Stage 2 多分类攻击识别（FocalLoss）"""
    # 类别权重
    from collections import Counter
    cnt = Counter(y_train)
    weights = np.ones(num_classes)
    for c in range(num_classes):
        if c in cnt and cnt[c] > 0:
            weights[c] = len(y_train) / (num_classes * cnt[c])
    weights = np.sqrt(weights)  # sqrt scheme works best
    alpha = torch.tensor(weights, device=device, dtype=torch.float32)

    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.LongTensor(y_val).to(device)

    best_acc = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for i in range(0, len(Xt), batch_size):
            bx = Xt[i:i+batch_size]
            by = yt[i:i+batch_size]
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()
        scheduler.step(1 - val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Stage2 Epoch {epoch:2d}/{epochs} - Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return best_acc


def two_stage_predict(model_s1, model_s2, X, s2_reverse, conf_threshold=0.5):
    """两阶段联合预测"""
    N = len(X)
    Xt = torch.FloatTensor(X).to(device)

    # Stage 1: binary
    model_s1.eval()
    with torch.no_grad():
        s1_logits = model_s1(Xt)
        s1_prob = torch.softmax(s1_logits, dim=1).cpu().numpy()
    is_attack = s1_prob[:, 1] >= 0.5

    # Stage 2: attack multi-class
    model_s2.eval()
    with torch.no_grad():
        s2_logits = model_s2(Xt)
        s2_prob = torch.softmax(s2_logits, dim=1).cpu().numpy()
    s2_max_prob = s2_prob.max(axis=1)
    s2_pred = s2_prob.argmax(axis=1)

    # Combined: map back to original 24-class labels
    # 11 = normal, 23 = unknown_attack
    final = np.full(N, 11, dtype=np.int64)  # default normal
    for i in range(N):
        if is_attack[i]:
            s2_class = s2_pred[i]
            if s2_class == 22:  # Stage 2 says "not_attack" → override to normal
                final[i] = 11
            elif s2_max_prob[i] >= conf_threshold:
                final[i] = s2_reverse[s2_class]  # specific attack
            else:
                final[i] = 23  # unknown_attack

    return final, s1_prob, s2_prob


def evaluate_two_stage(model_s1, model_s2, data, s2_reverse, label, conf_threshold=0.5):
    """完整评估两阶段分类"""
    X = data['X_test']
    y_true = label
    class_names = data['class_names']

    y_pred, s1_prob, s2_prob = two_stage_predict(
        model_s1, model_s2, X, s2_reverse, conf_threshold)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 分析
    n_normal_pred = (y_pred == 11).sum()
    n_unknown_pred = (y_pred == 23).sum()
    n_attack_pred = len(y_pred) - n_normal_pred - n_unknown_pred

    # unknown_attack recall
    if 23 < len(class_names):
        uk_mask = (y_true == 23)
        n_uk = uk_mask.sum()
        n_uk_caught = ((y_pred == 23) & uk_mask).sum()
    else:
        n_uk = 0
        n_uk_caught = 0

    print(f"\n[两阶段评估] threshold={conf_threshold}")
    print(f"  Acc: {acc:.4f}, F1: {f1:.4f}")
    print(f"  判为 normal:     {n_normal_pred}")
    print(f"  判为 unknown:    {n_unknown_pred}")
    print(f"  判为具体攻击:    {n_attack_pred}")
    if n_uk > 0:
        print(f"  unknown_attack:  实际{n_uk}, 命中{n_uk_caught} ({n_uk_caught/n_uk*100:.1f}%)")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {'acc': acc, 'f1': f1, 'cm': cm}


def main():
    print("=" * 60)
    print("两阶段分类模型：Stage1 二分类 + Stage2 攻击细分")
    print("=" * 60)
    print(f"Device: {device}")

    data = load_two_stage_data()
    feature_cols = data['feature_cols']
    class_names = data['class_names']
    print(f"特征列: {len(feature_cols)}")

    # ===== Stage 1: Binary DNN =====
    print(f"\n--- Stage 1: 二分类 (normal vs attack) ---")
    print(f"  训练集: {len(data['X_train'])} (normal={int((data['y1_train']==0).sum())}, attack={int((data['y1_train']==1).sum())})")

    embedded_dim = compute_embedded_input_dim(feature_cols, SERVICE_EMBEDDING_DIM)
    s1_base = BinaryDNN(embedded_dim, hidden_dims=[128, 64], dropout=0.3)
    model_s1 = ServiceEmbeddingModel(s1_base, vocab_size=69,
                                      feature_cols=feature_cols).to(device)

    s1_acc = train_stage1(model_s1, data['X_train'], data['y1_train'],
                          data['X_val'], data['y1_val'], epochs=30)
    print(f"  Stage1 最佳 Val Acc: {s1_acc:.4f}")

    # 测试 Stage 1
    Xt_test = torch.FloatTensor(data['X_test']).to(device)
    model_s1.eval()
    with torch.no_grad():
        s1_logits = model_s1(Xt_test)
        s1_test_acc = (s1_logits.argmax(1) == torch.LongTensor(data['y1_test']).to(device)).float().mean().item()
    print(f"  Stage1 内部测试 Acc: {s1_test_acc:.4f}")

    # ===== Stage 2: Attack DNN =====
    print(f"\n--- Stage 2: 攻击分类 (23类=22攻击+not_attack) ---")
    print(f"  训练集: {len(data['X2_train'])} (含normal作为not_attack类)")

    from models.dnn.dnn_model import DNN
    s2_base = DNN(embedded_dim, 23, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    model_s2 = ServiceEmbeddingModel(s2_base, vocab_size=69,
                                      feature_cols=feature_cols).to(device)

    s2_acc = train_stage2(model_s2, data['X2_train'], data['y2_train'],
                          data['X2_val'], data['y2_val'], num_classes=23, epochs=50)
    print(f"  Stage2 最佳 Val Acc: {s2_acc:.4f}")

    # 测试 Stage 2
    X2t = torch.FloatTensor(data['X2_test']).to(device)
    y2t = torch.LongTensor(data['y2_test']).to(device)
    model_s2.eval()
    with torch.no_grad():
        s2_logits = model_s2(X2t)
        s2_test_acc = (s2_logits.argmax(1) == y2t).float().mean().item()
    print(f"  Stage2 内部测试 Acc: {s2_test_acc:.4f}")

    # ===== 联合评估：内部测试集 =====
    print(f"\n{'='*60}")
    print("联合评估：内部测试集 (n={})".format(len(data['X_test'])))
    print("=" * 60)
    for th in [0.5, 0.7, 0.9]:
        metrics = evaluate_two_stage(model_s1, model_s2, data, data['s2_reverse'],
                                     data['mc_test'], conf_threshold=th)
        if 'cm_int' not in data:
            data['best_metrics'] = metrics

    # ===== 外部评估：train_test =====
    print(f"\n{'='*60}")
    print("联合评估：外部 train_test 集")
    print("=" * 60)

    # 加载外部数据
    ext_path = os.path.join(base_dir, 'Train', 'KDDTest+_20Percent.txt')
    if not os.path.exists(ext_path):
        ext_path = os.path.join(base_dir, 'Train', 'KDDTest+.txt')
    if not os.path.exists(ext_path):
        ext_path = os.path.join(base_dir, 'Train', 'KDDTrain+_20Percent.txt')
    if not os.path.exists(ext_path):
        print("外部测试文件不存在，跳过")
        return

    # 复用 DNN 的 evaluate_external_test 预处理逻辑
    from models.dnn.dnn_model import evaluate_external_test_dnn as ext_eval
    from models.dnn.dnn_model import CATEGORICAL_FEATURES

    # 创建伪模型对象来模拟预测
    # 直接在这里写外部评估逻辑
    # 加载预处理工具
    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_robust.pkl')
    service_le_path = os.path.join(base_dir, 'Train', 'encoder_service_embedding.pkl')
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
    service_le = joblib.load(service_le_path)
    UNK_SERVICE_IDX = len(service_le.classes_)

    # 读取外部数据
    col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty']

    df_ext = pd.read_csv(ext_path, header=None, names=col_names, low_memory=False)
    print(f"[外部数据] 原始: {df_ext.shape}")

    # 标签处理
    with open(os.path.join(train_dir, 'encoder_multiclass_23_classes.txt')) as f:
        canonical_classes = [l.strip() for l in f if l.strip()]

    def map_to_24class(label):
        if label == 'normal': return 11
        if label in canonical_classes:
            return canonical_classes.index(label)
        return 23  # unknown_attack

    df_ext['label_multiclass_encoded'] = df_ext['label'].apply(map_to_24class)
    y_ext = df_ext['label_multiclass_encoded'].values

    # Service 安全编码
    known_services = set(service_le.classes_)
    df_ext['service_encoded'] = df_ext['service'].apply(
        lambda x: service_le.transform([x])[0] if x in known_services else UNK_SERVICE_IDX)

    # OneHot + feature construction
    df_ohe = pd.DataFrame(
        ohe.transform(df_ext[CATEGORICAL_FEATURES]),
        columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df_ext.index
    )
    df_rest = df_ext.drop(columns=CATEGORICAL_FEATURES)

    # Remove columns that were dropped in preprocessing
    drop_cols = ['num_outbound_cmds', 'is_host_login', 'urgent', 'su_attempted', 'label', 'difficulty',
                 'label_multiclass_encoded']
    for c in drop_cols:
        if c in df_rest.columns:
            df_rest = df_rest.drop(columns=[c])

    df_ext_enc = pd.concat([df_rest, df_ohe], axis=1)

    # Fill missing columns
    train_feature_cols = [c for c in pd.read_csv(
        os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1).columns
        if c not in ['label', 'difficulty', 'label_binary', 'label_category',
                     'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']]
    for col in train_feature_cols:
        if col not in df_ext_enc.columns:
            df_ext_enc[col] = 0.0

    # Binary Indicator
    ZERO_INFLATED_COLS = ['src_bytes', 'dst_bytes', 'duration',
                          'num_failed_logins', 'num_shells', 'num_access_files',
                          'num_file_creations', 'num_root']
    for col in ZERO_INFLATED_COLS:
        df_ext_enc[f'is_zero_{col}'] = (df_ext_enc[col] == 0).astype(int)

    # Log1p
    LOGP1_COLS = ['src_bytes', 'dst_bytes', 'duration', 'hot']
    for col in LOGP1_COLS:
        df_ext_enc[col] = np.log1p(df_ext_enc[col].clip(lower=0))

    # Scaler
    numeric_cols = [c for c in train_feature_cols
                    if c not in [x for x in train_feature_cols if x.startswith('protocol_type_') or
                                 x.startswith('service_') or x.startswith('flag_')]]
    scaler_cols = [c for c in numeric_cols if c in df_ext_enc.columns]
    df_ext_enc[scaler_cols] = scaler.transform(df_ext_enc[scaler_cols])

    X_ext = df_ext_enc[train_feature_cols].values.astype(np.float32)
    print(f"[外部数据] 预处理完成，特征矩阵: {X_ext.shape}")

    # 评估
    for th in [0.5, 0.7, 0.9]:
        y_pred_ext, _, _ = two_stage_predict(
            model_s1, model_s2, X_ext, data['s2_reverse'], conf_threshold=th)

        ext_acc = accuracy_score(y_ext, y_pred_ext)
        ext_f1 = f1_score(y_ext, y_pred_ext, average='weighted', zero_division=0)

        n_uk = (y_ext == 23).sum()
        n_uk_caught = ((y_pred_ext == 23) & (y_ext == 23)).sum()

        print(f"\n[外部评估] threshold={th}")
        print(f"  Acc: {ext_acc:.4f}, F1: {ext_f1:.4f}")
        print(f"  判为 normal: {(y_pred_ext==11).sum()}, unknown: {(y_pred_ext==23).sum()}, "
              f"具体攻击: {((y_pred_ext!=11) & (y_pred_ext!=23)).sum()}")
        if n_uk > 0:
            print(f"  unknown_attack: 实际{n_uk}, 识别{n_uk_caught} ({n_uk_caught/n_uk*100:.1f}%)")


if __name__ == '__main__':
    main()
