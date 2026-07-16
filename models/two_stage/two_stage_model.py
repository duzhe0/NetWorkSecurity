"""
两阶段分类模型：
  Stage 1: 二分类（normal vs attack）
  Stage 2: 多分类（22种攻击），低置信 → unknown_attack
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==================== 固定随机种子 ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    with open(os.path.join(train_dir, 'encoder_multiclass_23_classes.txt')) as f:
        class_names = [l.strip() for l in f if l.strip()]

    df_train = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'))
    df_val   = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_val.csv'))
    df_test  = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_test.csv'))

    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                    'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val   = df_val[feature_cols].values.astype(np.float32)
    X_test  = df_test[feature_cols].values.astype(np.float32)

    y1_train = df_train['label_binary'].values.astype(np.int64)
    y1_val = df_val['label_binary'].values.astype(np.int64)
    y1_test = df_test['label_binary'].values.astype(np.int64)

    mc_train = df_train['label_multiclass_encoded'].values.astype(np.int64)
    mc_val = df_val['label_multiclass_encoded'].values.astype(np.int64)
    mc_test = df_test['label_multiclass_encoded'].values.astype(np.int64)

    orig_attack_labels = sorted([c for c in range(24) if c != 11 and c != 23])
    s2_mapping = {orig: new for new, orig in enumerate(orig_attack_labels)}
    s2_mapping[11] = 22
    s2_reverse = {new: orig for new, orig in enumerate(orig_attack_labels)}
    s2_reverse[22] = 11

    def remap(y_orig):
        y_new = np.full_like(y_orig, -1)
        for orig, new in s2_mapping.items():
            y_new[y_orig == orig] = new
        return y_new

    def filter_for_s2(X, y_orig):
        mask = (y_orig != 23)
        return X[mask], remap(y_orig)[mask]

    X2_train, y2_train = filter_for_s2(X_train, mc_train)
    X2_val, y2_val = filter_for_s2(X_val, mc_val)
    X2_test, y2_test = filter_for_s2(X_test, mc_test)

    # 稀有类过采样（Stage 2 训练集）
    X2_train_bal, y2_train_bal = oversample_rare_classes(X2_train, y2_train, min_samples=500, noise_std=0.05)

    return {
        'feature_cols': feature_cols,
        'class_names': class_names,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y1_train': y1_train, 'y1_val': y1_val, 'y1_test': y1_test,
        'X2_train': X2_train_bal, 'X2_val': X2_val, 'X2_test': X2_test,
        'y2_train': y2_train_bal, 'y2_val': y2_val, 'y2_test': y2_test,
        's2_reverse': s2_reverse,
        'mc_train': mc_train, 'mc_val': mc_val, 'mc_test': mc_test,
    }


def oversample_rare_classes(X, y, min_samples=500, noise_std=0.05, class_min_samples=None):
    """对样本数 < min_samples 的稀有类进行过采样（加小噪声防过拟合）
    
    Args:
        class_min_samples: dict, 指定特定类的 min_samples，如 {20: 200, 21: 200}
    """
    cnt = Counter(y)
    X_list, y_list = [X], [y]
    n_added = 0

    for cls, count in cnt.items():
        cls_min = class_min_samples.get(cls, min_samples) if class_min_samples else min_samples
        if count >= cls_min:
            continue
        mask = (y == cls)
        X_cls = X[mask]
        n_needed = cls_min - count
        # 重复采样 + 加高斯噪声
        idx = np.random.choice(len(X_cls), size=n_needed, replace=True)
        X_new = X_cls[idx].copy()
        X_new += np.random.normal(0, noise_std * np.std(X_cls, axis=0), X_new.shape).astype(np.float32)
        y_new = np.full(n_needed, cls, dtype=np.int64)
        X_list.append(X_new)
        y_list.append(y_new)
        n_added += n_needed

    if n_added > 0:
        print(f"  [过采样] 稀有类共增加 {n_added} 样本 (min_samples={min_samples})")
        for cls in sorted(cnt.keys()):
            cls_min = class_min_samples.get(cls, min_samples) if class_min_samples else min_samples
            if cnt[cls] < cls_min:
                print(f"    类{cls}: {cnt[cls]} -> {cls_min}")
        return np.vstack(X_list), np.concatenate(y_list)
    return X, y


# ==================== 训练函数 ====================

def train_stage1(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=128):
    """训练 Stage 1 二分类"""
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
            bx, by = Xt[i:i+batch_size], yt[i:i+batch_size]
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


def train_stage2(model, X_train, y_train, X_val, y_val, num_classes=23, epochs=80, batch_size=64):
    """训练 Stage 2 多分类攻击识别（FocalLoss + 余弦退火）"""
    cnt = Counter(y_train)
    # 直接用逆频率（不加 sqrt），极端不平衡需要更强权重
    weights = np.ones(num_classes)
    for c in range(num_classes):
        if c in cnt and cnt[c] > 0:
            weights[c] = len(y_train) / (num_classes * cnt[c])
    # 对权重做 clipping，防止极端值
    weights = np.clip(weights, 0.1, 10.0)
    alpha = torch.tensor(weights, device=device, dtype=torch.float32)

    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.LongTensor(y_val).to(device)

    best_acc = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i+batch_size]
            bx, by = Xt[idx], yt[idx]
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()
            # 也计算 macro F1 作为参考
            val_pred = val_logits.argmax(1).cpu().numpy()
            val_f1_macro = f1_score(y_val, val_pred, average='macro', zero_division=0)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Stage2 Epoch {epoch:2d}/{epochs} - Val Acc: {val_acc:.4f}, Macro F1: {val_f1_macro:.4f}")

    model.load_state_dict(best_state)
    return best_acc


# ==================== 预测与评估 ====================

def two_stage_predict(model_s1, model_s2, X, s2_reverse, s1_threshold=0.5, s2_threshold=0.5):
    """两阶段联合预测
    
    Args:
        s1_threshold: Stage1 二分类阈值，控制 normal vs attack 判定
        s2_threshold: Stage2 置信度阈值，控制是否接受攻击细分结果
    """
    N = len(X)
    Xt = torch.FloatTensor(X).to(device)

    model_s1.eval()
    model_s2.eval()
    with torch.no_grad():
        s1_logits = model_s1(Xt)
        s1_prob = torch.softmax(s1_logits, dim=1).cpu().numpy()
        s2_logits = model_s2(Xt)
        s2_prob = torch.softmax(s2_logits, dim=1).cpu().numpy()

    is_attack = s1_prob[:, 1] >= s1_threshold
    s2_max_prob = s2_prob.max(axis=1)
    s2_pred = s2_prob.argmax(axis=1)

    final = np.full(N, 11, dtype=np.int64)
    for i in range(N):
        if is_attack[i]:
            s2_class = s2_pred[i]
            if s2_class == 22:
                final[i] = 11
            elif s2_max_prob[i] >= s2_threshold:
                final[i] = s2_reverse.get(s2_class, 23)
            else:
                final[i] = 23

    return final, s1_prob, s2_prob


def evaluate_two_stage(model_s1, model_s2, data, s2_reverse, label, s1_threshold=0.5, s2_threshold=0.5):
    """完整评估两阶段分类"""
    X = data['X_test']
    y_true = label
    class_names = data['class_names']

    y_pred, s1_prob, s2_prob = two_stage_predict(
        model_s1, model_s2, X, s2_reverse, s1_threshold, s2_threshold)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    n_normal_pred = (y_pred == 11).sum()
    n_unknown_pred = (y_pred == 23).sum()
    n_attack_pred = len(y_pred) - n_normal_pred - n_unknown_pred

    if 23 < len(class_names):
        uk_mask = (y_true == 23)
        n_uk = uk_mask.sum()
        n_uk_caught = ((y_pred == 23) & uk_mask).sum()
    else:
        n_uk = 0
        n_uk_caught = 0

    print(f"\n[两阶段评估] s1_th={s1_threshold}, s2_th={s2_threshold}")
    print(f"  Acc: {acc:.4f}, F1: {f1:.4f}")
    print(f"  判为 normal:     {n_normal_pred}")
    print(f"  判为 unknown:    {n_unknown_pred}")
    print(f"  判为具体攻击:    {n_attack_pred}")
    if n_uk > 0:
        print(f"  unknown_attack:  实际{n_uk}, 命中{n_uk_caught} ({n_uk_caught/n_uk*100:.1f}%)")

    return {'acc': acc, 'f1': f1}


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("两阶段分类模型：Stage1 二分类 + Stage2 攻击细分")
    print("=" * 60)
    print(f"Device: {device}, Seed: {SEED}")

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

    Xt_test = torch.FloatTensor(data['X_test']).to(device)
    model_s1.eval()
    with torch.no_grad():
        s1_logits = model_s1(Xt_test)
        s1_test_acc = (s1_logits.argmax(1) == torch.LongTensor(data['y1_test']).to(device)).float().mean().item()
    print(f"  Stage1 内部测试 Acc: {s1_test_acc:.4f}")

    torch.save(model_s1.state_dict(), os.path.join(base_dir, 'model_stage1_best.pth'))
    print(f"  Stage1 模型已保存: model_stage1_best.pth")

    # ===== Stage 2: Attack DNN =====
    print(f"\n--- Stage 2: 攻击分类 (23类=22攻击+not_attack) ---")
    print(f"  训练集: {len(data['X2_train'])} (过采样后, 含normal作为not_attack类)")
    cnt_s2 = Counter(data['y2_train'])
    print(f"  类别分布: min={min(cnt_s2.values())}, max={max(cnt_s2.values())}, "
          f"median={np.median(list(cnt_s2.values())):.0f}")

    from models.dnn.dnn_model import DNN
    s2_base = DNN(embedded_dim, 23, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    model_s2 = ServiceEmbeddingModel(s2_base, vocab_size=69,
                                      feature_cols=feature_cols).to(device)

    s2_acc = train_stage2(model_s2, data['X2_train'], data['y2_train'],
                          data['X2_val'], data['y2_val'], num_classes=23, epochs=80)
    print(f"  Stage2 最佳 Val Acc: {s2_acc:.4f}")

    X2t = torch.FloatTensor(data['X2_test']).to(device)
    y2t = torch.LongTensor(data['y2_test']).to(device)
    model_s2.eval()
    with torch.no_grad():
        s2_logits = model_s2(X2t)
        s2_test_acc = (s2_logits.argmax(1) == y2t).float().mean().item()
    print(f"  Stage2 内部测试 Acc: {s2_test_acc:.4f}")

    torch.save(model_s2.state_dict(), os.path.join(base_dir, 'model_stage2_best.pth'))
    print(f"  Stage2 模型已保存: model_stage2_best.pth")

    # ===== 联合评估：内部测试集 =====
    print(f"\n{'='*60}")
    print("联合评估：内部测试集 (n={})".format(len(data['X_test'])))
    print("=" * 60)
    for th in [0.5, 0.7, 0.9]:
        evaluate_two_stage(model_s1, model_s2, data, data['s2_reverse'],
                           data['mc_test'], s2_threshold=th)

    # ===== 外部评估 =====
    print(f"\n{'='*60}")
    print("联合评估：外部 train_test 集")
    print("=" * 60)

    ext_path = os.path.join(base_dir, 'Train', 'train_test')
    if not os.path.exists(ext_path):
        print("外部测试文件 Train/train_test 不存在，跳过")
        return

    from models.dnn.dnn_model import CATEGORICAL_FEATURES

    ohe = joblib.load(os.path.join(base_dir, 'Train', 'encoder_onehot.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'Train', 'scaler_robust.pkl'))
    service_le = joblib.load(os.path.join(base_dir, 'Train', 'encoder_service_embedding.pkl'))
    UNK_SERVICE_IDX = len(service_le.classes_)

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

    with open(os.path.join(train_dir, 'encoder_multiclass_23_classes.txt')) as f:
        canonical_classes = [l.strip() for l in f if l.strip()]

    def map_to_24class(label):
        if label == 'normal': return 11
        if label in canonical_classes:
            return canonical_classes.index(label)
        return 23

    df_ext['label_multiclass_encoded'] = df_ext['label'].apply(map_to_24class)
    y_ext = df_ext['label_multiclass_encoded'].values

    known_services = set(service_le.classes_)
    df_ext['service_encoded'] = df_ext['service'].apply(
        lambda x: service_le.transform([x])[0] if x in known_services else UNK_SERVICE_IDX)

    df_ohe = pd.DataFrame(
        ohe.transform(df_ext[CATEGORICAL_FEATURES]),
        columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df_ext.index
    )
    df_rest = df_ext.drop(columns=CATEGORICAL_FEATURES)

    drop_cols = ['num_outbound_cmds', 'is_host_login', 'urgent', 'su_attempted',
                 'label', 'difficulty', 'label_multiclass_encoded']
    for c in drop_cols:
        if c in df_rest.columns:
            df_rest = df_rest.drop(columns=[c])

    df_ext_enc = pd.concat([df_rest, df_ohe], axis=1)

    # ===== 手工特征工程（与 data_preprocessing 保持一致） =====
    ftp_telnet_cols = [c for c in ['service_ftp', 'service_telnet', 'service_ftp_data']
                       if c in df_ext_enc.columns]
    df_ext_enc['is_ftp_telnet'] = df_ext_enc[ftp_telnet_cols].max(axis=1) if ftp_telnet_cols else 0
    eps = 1e-6
    df_ext_enc['svc_auth_fail_score'] = (
        df_ext_enc['is_ftp_telnet'] *
        df_ext_enc['num_failed_logins'] / (df_ext_enc['num_failed_logins'] + df_ext_enc['num_compromised'] + eps)
    )
    df_ext_enc['serror_logged_cross'] = df_ext_enc['serror_rate'] * df_ext_enc['logged_in']

    train_feature_cols = [c for c in pd.read_csv(
        os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1).columns
        if c not in ['label', 'difficulty', 'label_binary', 'label_category',
                     'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']]
    for col in train_feature_cols:
        if col not in df_ext_enc.columns:
            df_ext_enc[col] = 0.0

    ZERO_INFLATED_COLS = ['src_bytes', 'dst_bytes', 'duration',
                          'num_failed_logins', 'num_shells', 'num_access_files',
                          'num_file_creations', 'num_root']
    for col in ZERO_INFLATED_COLS:
        df_ext_enc[f'is_zero_{col}'] = (df_ext_enc[col] == 0).astype(int)

    LOGP1_COLS = ['src_bytes', 'dst_bytes', 'duration', 'hot']
    for col in LOGP1_COLS:
        df_ext_enc[col] = np.log1p(df_ext_enc[col].clip(lower=0))

    numeric_cols = [c for c in train_feature_cols
                    if not (c.startswith('protocol_type_') or c.startswith('service_') or c.startswith('flag_'))]
    scaler_cols = [c for c in numeric_cols if c in df_ext_enc.columns]
    df_ext_enc[scaler_cols] = scaler.transform(df_ext_enc[scaler_cols])

    X_ext = df_ext_enc[train_feature_cols].values.astype(np.float32)
    print(f"[外部数据] 预处理完成，特征矩阵: {X_ext.shape}")

    for th in [0.5, 0.7, 0.9]:
        y_pred_ext, s1p_ext, s2p_ext = two_stage_predict(
            model_s1, model_s2, X_ext, data['s2_reverse'], s2_threshold=th)

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

        if th == 0.5:
            y1_true = (y_ext != 11).astype(int)
            s1_pred_binary = (s1p_ext[:, 1] >= 0.5).astype(int)
            print(f"\n  [Stage1 详细]")
            print(f"  Stage1 binary acc: {accuracy_score(y1_true, s1_pred_binary):.4f}")
            print(f"  真 normal -> pred normal: {((y1_true==0)&(s1_pred_binary==0)).sum()}")
            print(f"  真 normal -> pred attack: {((y1_true==0)&(s1_pred_binary==1)).sum()}")
            print(f"  真 attack -> pred attack: {((y1_true==1)&(s1_pred_binary==1)).sum()}")
            print(f"  真 attack -> pred normal: {((y1_true==1)&(s1_pred_binary==0)).sum()}")

            print(f"\n  [按真实标签分组]")
            for label_name, label_val in [('normal', 11), ('known_attack', -1), ('unknown', 23)]:
                if label_val == -1:
                    mask = (y_ext != 11) & (y_ext != 23)
                else:
                    mask = (y_ext == label_val)
                n = mask.sum()
                if n == 0:
                    continue
                correct = (y_pred_ext[mask] == y_ext[mask]).sum()
                pred_norm = (y_pred_ext[mask] == 11).sum()
                pred_unk = (y_pred_ext[mask] == 23).sum()
                pred_atk = n - pred_norm - pred_unk
                print(f"  真={label_name}({n}): 正确{correct}({correct/n*100:.1f}%), "
                      f"预测为 normal={pred_norm}, unknown={pred_unk}, attack={pred_atk}")

            print(f"\n  [每类攻击准确率]")
            for c in sorted(set(y_ext)):
                if c == 11:
                    continue
                mask = (y_ext == c)
                n_c = mask.sum()
                if n_c == 0:
                    continue
                correct_c = (y_pred_ext[mask] == y_ext[mask]).sum()
                cname = canonical_classes[c] if c < len(canonical_classes) else f'idx{c}'
                pred_as_norm = (y_pred_ext[mask] == 11).sum()
                pred_as_unk = (y_pred_ext[mask] == 23).sum()
                print(f"  {cname}({n_c}): acc={correct_c/n_c*100:.1f}%, "
                      f"->normal={pred_as_norm}, ->unknown={pred_as_unk}")


if __name__ == '__main__':
    main()
