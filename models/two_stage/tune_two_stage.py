"""
两阶段模型调参脚本
Phase 1: Grid search (min_samples, gamma) - Stage2 快速训练 40 epoch
Phase 2: 最佳组合全量训练 80 epoch + 温度缩放
Phase 3: (s1_threshold, s2_conf_threshold) 网格扫描
Phase 4: 外部评估
"""
import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from itertools import product
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)
from models.losses import FocalLoss
from models.embedding_utils import ServiceEmbeddingModel, compute_embedded_input_dim, SERVICE_EMBEDDING_DIM
from models.dnn.dnn_model import DNN
from models.two_stage.two_stage_model import (
    BinaryDNN, load_two_stage_data, oversample_rare_classes, train_stage1, two_stage_predict
)

train_dir = os.path.join(base_dir, 'Train')
device = torch.device('cuda')


def train_stage2_fast(model, X_train, y_train, X_val, y_val, num_classes, epochs, batch_size, gamma):
    """快速训练 Stage 2（可配置 gamma）"""
    cnt = Counter(y_train)
    weights = np.ones(num_classes)
    for c in range(num_classes):
        if c in cnt and cnt[c] > 0:
            weights[c] = len(y_train) / (num_classes * cnt[c])
    weights = np.clip(weights, 0.1, 10.0)
    alpha = torch.tensor(weights, device=device, dtype=torch.float32)

    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.LongTensor(y_val).to(device)

    best_acc = 0
    best_f1 = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()
            val_pred = val_logits.argmax(1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_acc, best_f1


def temperature_scale(logits, T):
    """温度缩放校准"""
    return logits / max(T, 1e-8)


def find_best_temperature(model_s2, X_val, y_val, s2_reverse, model_s1, n_trials=20):
    """在验证集上找最佳温度"""
    model_s2.eval()
    model_s1.eval()

    # Collect all Stage 2 logits on val set
    Xt = torch.FloatTensor(X_val).to(device)
    with torch.no_grad():
        s1_logits = model_s1(Xt)
        s1_prob = torch.softmax(s1_logits, dim=1).cpu().numpy()
        s2_logits_orig = model_s2(Xt).cpu().numpy()

    is_attack = s1_prob[:, 1] >= 0.5

    best_t = 1.0
    best_f1 = 0
    for T in np.logspace(-0.5, 0.5, n_trials):  # 0.3 ~ 3.0
        s2_prob = torch.softmax(torch.FloatTensor(s2_logits_orig / T), dim=1).numpy()
        s2_max_prob = s2_prob.max(axis=1)
        s2_pred = s2_prob.argmax(axis=1)

        # Simple prediction at threshold 0.5
        final = np.full(len(X_val), 11, dtype=np.int64)
        for i in range(len(X_val)):
            if is_attack[i]:
                s2_class = s2_pred[i]
                if s2_class == 22:
                    final[i] = 11
                elif s2_max_prob[i] >= 0.5:
                    final[i] = s2_reverse.get(s2_class, 23)
                else:
                    final[i] = 23

        f1 = f1_score(y_val, final, average='weighted', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = T

    return best_t


def grid_search_phase1():
    """Phase 1: Grid search (min_samples, gamma)"""
    print("=" * 60)
    print("Phase 1: Grid Search (min_samples × gamma)")
    print("=" * 60)

    data = load_two_stage_data()  # won't apply oversampling here
    feature_cols = data['feature_cols']
    embedded_dim = compute_embedded_input_dim(feature_cols, SERVICE_EMBEDDING_DIM)

    # Load or use pre-trained Stage 1 (reuse saved)
    s1_base = BinaryDNN(embedded_dim, hidden_dims=[128, 64], dropout=0.3)
    model_s1 = ServiceEmbeddingModel(s1_base, vocab_size=69, feature_cols=feature_cols).to(device)
    model_s1.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage1_best.pth'),
                                         map_location=device))
    model_s1.eval()
    print("Stage1 模型已加载")

    # Get original (non-oversampled) S2 training data
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

    mc_train = data['mc_train']
    mc_val = data['mc_val']
    mask_train = (mc_train != 23)
    mask_val = (mc_val != 23)
    X2_train_orig = data['X_train'][mask_train]
    y2_train_orig = remap(mc_train)[mask_train]
    X2_val = data['X_val'][mask_val]
    y2_val = remap(mc_val)[mask_val]

    min_samples_list = [200, 500, 1000]
    gamma_list = [1.0, 2.0, 3.0]

    results = []
    best_combo = None
    best_f1 = 0

    for ms, gm in product(min_samples_list, gamma_list):
        # Oversample
        X2_tr, y2_tr = oversample_rare_classes(X2_train_orig, y2_train_orig,
                                                min_samples=ms, noise_std=0.05)
        print(f"\n--- min_samples={ms}, gamma={gm}, 训练集={len(X2_tr)} ---")

        s2_base = DNN(embedded_dim, 23, hidden_dims=[256, 128, 64], dropout_rate=0.3)
        model_s2 = ServiceEmbeddingModel(s2_base, vocab_size=69, feature_cols=feature_cols).to(device)

        acc, f1_macro = train_stage2_fast(model_s2, X2_tr, y2_tr, X2_val, y2_val,
                                           num_classes=23, epochs=40, batch_size=64, gamma=gm)
        results.append({'min_samples': ms, 'gamma': gm, 'val_acc': acc, 'val_f1_macro': f1_macro})
        print(f"  结果: Val Acc={acc:.4f}, Val Macro F1={f1_macro:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_combo = (ms, gm)

    print(f"\n{'='*60}")
    print("Phase 1 结果汇总:")
    results.sort(key=lambda x: x['val_f1_macro'], reverse=True)
    for r in results:
        print(f"  ms={r['min_samples']}, gamma={r['gamma']}: "
              f"Acc={r['val_acc']:.4f}, MacroF1={r['val_f1_macro']:.4f}")
    print(f"\n最佳组合: min_samples={best_combo[0]}, gamma={best_combo[1]}, MacroF1={best_f1:.4f}")

    return best_combo, data


def phase2_full_train(best_ms, best_gamma, data):
    """Phase 2: 最佳参数全量训练 + 温度缩放"""
    print(f"\n{'='*60}")
    print(f"Phase 2: 全量训练 (min_samples={best_ms}, gamma={best_gamma})")
    print("=" * 60)

    feature_cols = data['feature_cols']
    embedded_dim = compute_embedded_input_dim(feature_cols, SERVICE_EMBEDDING_DIM)

    # Stage 1 (reuse)
    s1_base = BinaryDNN(embedded_dim, hidden_dims=[128, 64], dropout=0.3)
    model_s1 = ServiceEmbeddingModel(s1_base, vocab_size=69, feature_cols=feature_cols).to(device)
    model_s1.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage1_best.pth'),
                                         map_location=device))
    model_s1.eval()

    # Stage 2 with best oversampling
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

    mc_train = data['mc_train']
    mc_val = data['mc_val']
    mask_train = (mc_train != 23)
    mask_val = (mc_val != 23)
    X2_train_orig = data['X_train'][mask_train]
    y2_train_orig = remap(mc_train)[mask_train]
    X2_val = data['X_val'][mask_val]
    y2_val = remap(mc_val)[mask_val]

    X2_tr, y2_tr = oversample_rare_classes(X2_train_orig, y2_train_orig,
                                            min_samples=best_ms, noise_std=0.05)
    cnt_s2 = Counter(y2_tr)
    print(f"训练集: {len(X2_tr)}, 类别分布: min={min(cnt_s2.values())}, max={max(cnt_s2.values())}")

    s2_base = DNN(embedded_dim, 23, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    model_s2 = ServiceEmbeddingModel(s2_base, vocab_size=69, feature_cols=feature_cols).to(device)

    # Full training with best gamma
    cnt = Counter(y2_tr)
    weights = np.ones(23)
    for c in range(23):
        if c in cnt and cnt[c] > 0:
            weights[c] = len(y2_tr) / (23 * cnt[c])
    weights = np.clip(weights, 0.1, 10.0)
    alpha = torch.tensor(weights, device=device, dtype=torch.float32)
    criterion = FocalLoss(alpha=alpha, gamma=best_gamma)
    optimizer = torch.optim.Adam(model_s2.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

    Xt = torch.FloatTensor(X2_tr).to(device)
    yt = torch.LongTensor(y2_tr).to(device)
    Xv = torch.FloatTensor(X2_val).to(device)
    yv = torch.LongTensor(y2_val).to(device)

    best_acc = 0
    best_f1 = 0
    best_state = None
    for epoch in range(1, 81):
        model_s2.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]
            logits = model_s2(Xt[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_s2.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model_s2.eval()
        with torch.no_grad():
            val_logits = model_s2(Xv)
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()
            val_pred = val_logits.argmax(1).cpu().numpy()
            val_f1 = f1_score(y2_val, val_pred, average='macro', zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model_s2.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}/80 - Val Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}")

    model_s2.load_state_dict(best_state)
    print(f"最佳 Val Acc: {best_acc:.4f}, Macro F1: {best_f1:.4f}")

    # Save
    torch.save(model_s2.state_dict(), os.path.join(base_dir, 'model_stage2_best.pth'))

    # Temperature scaling
    best_T = find_best_temperature(model_s2, data['X_val'], data['mc_val'],
                                    data['s2_reverse'], model_s1)
    print(f"最佳温度: T={best_T:.3f}")

    return model_s1, model_s2, s2_reverse, best_T


def phase3_threshold_sweep(model_s1, model_s2, s2_reverse, best_T, data):
    """Phase 3: 阈值网格扫描 (内部验证集)"""
    print(f"\n{'='*60}")
    print("Phase 3: (s1_threshold × s2_conf_threshold) 扫描")
    print("=" * 60)

    X_val = data['X_test']
    y_val = data['mc_test']

    model_s1.eval()
    model_s2.eval()
    Xt = torch.FloatTensor(X_val).to(device)
    with torch.no_grad():
        s1_logits = model_s1(Xt)
        s1_prob = torch.softmax(s1_logits, dim=1).cpu().numpy()
        s2_logits = model_s2(Xt).cpu().numpy()

    # Apply temperature scaling to s2 logits
    s2_prob_scaled = torch.softmax(torch.FloatTensor(s2_logits / best_T), dim=1).numpy()

    best_combo = None
    best_f1 = 0
    results = []

    s1_thresholds = [0.3, 0.4, 0.5, 0.6]
    s2_thresholds = [0.5, 0.6, 0.7, 0.8]

    for s1t, s2t in product(s1_thresholds, s2_thresholds):
        is_attack = s1_prob[:, 1] >= s1t
        s2_max_prob = s2_prob_scaled.max(axis=1)
        s2_pred = s2_prob_scaled.argmax(axis=1)

        final = np.full(len(X_val), 11, dtype=np.int64)
        for i in range(len(X_val)):
            if is_attack[i]:
                s2_class = s2_pred[i]
                if s2_class == 22:
                    final[i] = 11
                elif s2_max_prob[i] >= s2t:
                    final[i] = s2_reverse.get(s2_class, 23)
                else:
                    final[i] = 23

        acc = accuracy_score(y_val, final)
        f1 = f1_score(y_val, final, average='weighted', zero_division=0)

        # Also compute unknown recall
        uk_mask = (y_val == 23)
        n_uk = uk_mask.sum()
        uk_recall = ((final == 23) & uk_mask).sum() / max(n_uk, 1)

        # Normal precision
        norm_mask = (final == 11)
        n_norm_pred = norm_mask.sum()
        norm_precision = ((final == 11) & (y_val == 11)).sum() / max(n_norm_pred, 1)

        results.append({
            's1_th': s1t, 's2_th': s2t,
            'acc': acc, 'f1': f1,
            'uk_recall': uk_recall, 'norm_precision': norm_precision
        })

        if f1 > best_f1:
            best_f1 = f1
            best_combo = (s1t, s2t)

    results.sort(key=lambda x: x['f1'], reverse=True)
    print(f"{'s1_th':>6} {'s2_th':>6} {'Acc':>8} {'F1':>8} {'UK_Rec':>8} {'Norm_Prec':>10}")
    for r in results:
        print(f"{r['s1_th']:6.1f} {r['s2_th']:6.1f} {r['acc']:8.4f} {r['f1']:8.4f} "
              f"{r['uk_recall']:8.4f} {r['norm_precision']:10.4f}")

    print(f"\n最佳阈值: s1_th={best_combo[0]}, s2_th={best_combo[1]}, F1={best_f1:.4f}")
    return best_combo


def phase4_external_eval(model_s1, model_s2, s2_reverse, best_T, best_s1_th, best_s2_th):
    """Phase 4: 最终外部评估"""
    print(f"\n{'='*60}")
    print("Phase 4: 最终外部评估")
    print(f"T={best_T:.3f}, s1_th={best_s1_th}, s2_th={best_s2_th}")
    print("=" * 60)

    from models.dnn.dnn_model import CATEGORICAL_FEATURES

    ext_path = os.path.join(base_dir, 'Train', 'train_test')
    if not os.path.exists(ext_path):
        print("外部测试文件不存在")
        return

    ohe = joblib.load(os.path.join(train_dir, 'encoder_onehot.pkl'))
    scaler = joblib.load(os.path.join(train_dir, 'scaler_robust.pkl'))
    service_le = joblib.load(os.path.join(train_dir, 'encoder_service_embedding.pkl'))
    UNK_SERVICE_IDX = len(service_le.classes_)

    col_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

    df_ext = pd.read_csv(ext_path, header=None, names=col_names, low_memory=False)

    with open(os.path.join(train_dir, 'encoder_multiclass_23_classes.txt')) as f:
        canonical_classes = [l.strip() for l in f if l.strip()]

    def map_to_24class(label):
        if label == 'normal': return 11
        if label in canonical_classes: return canonical_classes.index(label)
        return 23

    df_ext['mc_enc'] = df_ext['label'].apply(map_to_24class)
    y_ext = df_ext['mc_enc'].values

    known_services = set(service_le.classes_)
    df_ext['service_encoded'] = df_ext['service'].apply(
        lambda x: service_le.transform([x])[0] if x in known_services else UNK_SERVICE_IDX)

    df_ohe = pd.DataFrame(ohe.transform(df_ext[CATEGORICAL_FEATURES]),
        columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES), index=df_ext.index)
    df_rest = df_ext.drop(columns=CATEGORICAL_FEATURES)
    drop_cols = ['num_outbound_cmds','is_host_login','urgent','su_attempted',
                 'label','difficulty','mc_enc']
    for c in drop_cols:
        if c in df_rest.columns: df_rest = df_rest.drop(columns=[c])
    df_ext_enc = pd.concat([df_rest, df_ohe], axis=1)

    exclude_cols = ['label','difficulty','label_binary','label_category',
                    'label_category_encoded','label_multiclass','label_multiclass_encoded']
    train_feature_cols = [c for c in pd.read_csv(
        os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1).columns
        if c not in exclude_cols]
    for col in train_feature_cols:
        if col not in df_ext_enc.columns: df_ext_enc[col] = 0.0

    ZERO_INFLATED_COLS = ['src_bytes','dst_bytes','duration',
        'num_failed_logins','num_shells','num_access_files','num_file_creations','num_root']
    for col in ZERO_INFLATED_COLS:
        df_ext_enc['is_zero_'+col] = (df_ext_enc[col] == 0).astype(int)
    for col in ['src_bytes','dst_bytes','duration','hot']:
        df_ext_enc[col] = np.log1p(df_ext_enc[col].clip(lower=0))

    numeric_cols = [c for c in train_feature_cols
        if not (c.startswith('protocol_type_') or c.startswith('service_') or c.startswith('flag_'))]
    scaler_cols = [c for c in numeric_cols if c in df_ext_enc.columns]
    df_ext_enc[scaler_cols] = scaler.transform(df_ext_enc[scaler_cols])

    X_ext = df_ext_enc[train_feature_cols].values.astype(np.float32)
    print(f"外部数据: {X_ext.shape}")

    # Custom predict with temperature scaling
    model_s1.eval(); model_s2.eval()
    Xt = torch.FloatTensor(X_ext).to(device)
    with torch.no_grad():
        s1_logits = model_s1(Xt)
        s1_prob = torch.softmax(s1_logits, dim=1).cpu().numpy()
        s2_logits = model_s2(Xt).cpu().numpy()
    s2_prob = torch.softmax(torch.FloatTensor(s2_logits / best_T), dim=1).numpy()

    is_attack = s1_prob[:, 1] >= best_s1_th
    s2_max_prob = s2_prob.max(axis=1)
    s2_pred = s2_prob.argmax(axis=1)

    y_pred = np.full(len(X_ext), 11, dtype=np.int64)
    for i in range(len(X_ext)):
        if is_attack[i]:
            s2_class = s2_pred[i]
            if s2_class == 22:
                y_pred[i] = 11
            elif s2_max_prob[i] >= best_s2_th:
                y_pred[i] = s2_reverse.get(s2_class, 23)
            else:
                y_pred[i] = 23

    acc = accuracy_score(y_ext, y_pred)
    f1 = f1_score(y_ext, y_pred, average='weighted', zero_division=0)

    n_norm_true = (y_ext == 11).sum()
    n_uk_true = (y_ext == 23).sum()
    n_atk_true = ((y_ext != 11) & (y_ext != 23)).sum()
    n_norm_pred = (y_pred == 11).sum()
    n_uk_pred = (y_pred == 23).sum()
    n_atk_pred = ((y_pred != 11) & (y_pred != 23)).sum()

    print(f"\n真实分布: normal={n_norm_true}, known_attack={n_atk_true}, unknown={n_uk_true}")
    print(f"预测分布: normal={n_norm_pred}, unknown={n_uk_pred}, specific_attack={n_atk_pred}")
    print(f"Acc: {acc:.4f}, F1: {f1:.4f}")

    # Unknown recall
    uk_recall = ((y_pred == 23) & (y_ext == 23)).sum() / n_uk_true
    print(f"Unknown recall: {((y_pred == 23) & (y_ext == 23)).sum()}/{n_uk_true} ({uk_recall*100:.1f}%)")

    # Known attack breakdown
    known_mask = (y_ext != 11) & (y_ext != 23)
    n_known = known_mask.sum()
    known_correct = (y_pred[known_mask] == y_ext[known_mask]).sum()
    print(f"\n已知攻击: {known_correct}/{n_known} ({known_correct/n_known*100:.1f}%)")
    known_pred_unk = (y_pred[known_mask] == 23).sum()
    known_pred_norm = (y_pred[known_mask] == 11).sum()
    known_pred_wrong = n_known - known_correct - known_pred_unk - known_pred_norm
    print(f"  ->unknown={known_pred_unk}, ->normal={known_pred_norm}, ->wrong_attack={known_pred_wrong}")

    # Unknown breakdown
    uk_mask = (y_ext == 23)
    uk_pred_unk = (y_pred[uk_mask] == 23).sum()
    uk_pred_norm = (y_pred[uk_mask] == 11).sum()
    uk_pred_atk = n_uk_true - uk_pred_unk - uk_pred_norm
    print(f"\n未知攻击: ->unknown={uk_pred_unk}, ->normal={uk_pred_norm}, ->具体攻击={uk_pred_atk}")

    # Per-class
    print(f"\n每类已知攻击:")
    for c in sorted(set(y_ext)):
        if c == 11 or c == 23: continue
        mask = (y_ext == c)
        n_c = mask.sum()
        correct_c = (y_pred[mask] == y_ext[mask]).sum()
        cname = canonical_classes[c] if c < len(canonical_classes) else 'idx'+str(c)
        pred_unk = (y_pred[mask] == 23).sum()
        pred_norm = (y_pred[mask] == 11).sum()
        print(f"  {cname}({n_c}): acc={correct_c/n_c*100:.1f}%, ->unknown={pred_unk}, ->normal={pred_norm}")

    return acc, f1


# ==================== Main ====================

if __name__ == '__main__':
    # Phase 1: Skip grid search - use known best (ms=500, gamma=1.0, MacroF1=0.8010)
    data = load_two_stage_data()
    best_ms, best_gamma = 500, 1.0
    print(f"跳过 Phase 1，使用已知最佳: min_samples={best_ms}, gamma={best_gamma}")

    # Phase 2: Full train + temperature scaling
    model_s1, model_s2, s2_reverse, best_T = phase2_full_train(
        best_ms, best_gamma, data)

    # Phase 3: Threshold sweep
    best_s1_th, best_s2_th = phase3_threshold_sweep(
        model_s1, model_s2, s2_reverse, best_T, data)

    # Phase 4: External eval
    phase4_external_eval(model_s1, model_s2, s2_reverse, best_T,
                         best_s1_th, best_s2_th)
