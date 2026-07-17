"""Stage1 ROC 曲线评估：外部测试集上的 Normal vs Attack 二分类 ROC/AUC

输出: Visualization/roc_stage1_external.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

parser = argparse.ArgumentParser(description='Stage1 ROC 曲线评估')
parser.add_argument('--test_file', type=str, default='train_test',
                    help='外部测试文件（Train目录下），如 train_test / KDDTest+ / KDDTest-21')
args_roc = parser.parse_args()

from models.embedding_utils import compute_embedded_input_dim, ServiceEmbeddingModel, SERVICE_EMBEDDING_DIM
from models.two_stage.two_stage_model import BinaryDNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir = os.path.join(base_dir, 'Train')

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
ZERO_INFLATED_COLS = ['src_bytes', 'dst_bytes', 'duration', 'num_failed_logins',
                       'num_shells', 'num_access_files', 'num_file_creations', 'num_root']

# ===== 1. 加载 Stage1 模型 =====
exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
df_train = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1)
feature_cols = [c for c in df_train.columns if c not in exclude_cols]
embedded_dim = compute_embedded_input_dim(feature_cols, SERVICE_EMBEDDING_DIM)

s1_base = BinaryDNN(embedded_dim, hidden_dims=[128, 64], dropout=0.3,
                    use_residual=False, activation='relu')
model_s1 = ServiceEmbeddingModel(s1_base, vocab_size=69, feature_cols=feature_cols).to(device)
model_s1.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage1_best.pth'),
                                     map_location=device))
model_s1.eval()
print(f"[模型] Stage1 加载完成，参数量: {sum(p.numel() for p in model_s1.parameters()):,}")

# ===== 2. 加载预处理工件 =====
ohe = joblib.load(os.path.join(train_dir, 'encoder_onehot.pkl'))
scaler = joblib.load(os.path.join(train_dir, 'scaler_robust.pkl'))
service_le = joblib.load(os.path.join(train_dir, 'encoder_service_embedding.pkl'))
UNK_SERVICE_IDX = len(service_le.classes_)

with open(os.path.join(train_dir, 'encoder_multiclass_23_classes.txt')) as f:
    canonical_classes = [l.strip() for l in f if l.strip()]

# ===== 3. 加载并预处理外部测试集 =====
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

ext_path = os.path.join(base_dir, 'Train', args_roc.test_file)
test_basename = args_roc.test_file.replace('.txt', '').replace('.csv', '')
df_ext = pd.read_csv(ext_path, header=None, names=col_names, low_memory=False)

def map_to_24class(label):
    if label == 'normal':
        return 11
    if label in canonical_classes:
        return canonical_classes.index(label)
    return 23

df_ext['mc_enc'] = df_ext['label'].apply(map_to_24class)
# Stage1 二分类标签: 0=normal, 1=attack
y_binary = (df_ext['mc_enc'] != 11).astype(int).values

known_services = set(service_le.classes_)
df_ext['service_encoded'] = df_ext['service'].apply(
    lambda x: service_le.transform([x])[0] if x in known_services else UNK_SERVICE_IDX)

df_ohe = pd.DataFrame(ohe.transform(df_ext[CATEGORICAL_FEATURES]),
                       columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES), index=df_ext.index)
df_rest = df_ext.drop(columns=CATEGORICAL_FEATURES)
drop_cols = ['num_outbound_cmds', 'is_host_login', 'urgent', 'su_attempted',
             'label', 'difficulty', 'mc_enc']
for c in drop_cols:
    if c in df_rest.columns:
        df_rest = df_rest.drop(columns=[c])
df_ext_enc = pd.concat([df_rest, df_ohe], axis=1)

train_feature_cols = [c for c in pd.read_csv(
    os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1).columns
    if c not in exclude_cols]
for col in train_feature_cols:
    if col not in df_ext_enc.columns:
        df_ext_enc[col] = 0.0

# ===== 4. 构造手工特征（与 data_preprocessing 保持一致） =====
ftp_telnet_cols = [c for c in ['service_ftp', 'service_telnet', 'service_ftp_data']
                   if c in df_ext_enc.columns]
df_ext_enc['is_ftp_telnet'] = df_ext_enc[ftp_telnet_cols].max(axis=1) if ftp_telnet_cols else 0
eps = 1e-6
df_ext_enc['svc_auth_fail_score'] = (
    df_ext_enc['is_ftp_telnet'] *
    df_ext_enc['num_failed_logins'] / (df_ext_enc['num_failed_logins'] +
                                        df_ext_enc['num_compromised'] + eps)
)
df_ext_enc['serror_logged_cross'] = df_ext_enc['serror_rate'] * df_ext_enc['logged_in']
df_ext_enc['is_failed_login'] = (
    (df_ext_enc['num_failed_logins'] > 0) & (df_ext_enc['logged_in'] == 0)
).astype(int)

for col in ZERO_INFLATED_COLS:
    df_ext_enc['is_zero_' + col] = (df_ext_enc[col] == 0).astype(int)
for col in ['src_bytes', 'dst_bytes', 'duration', 'hot']:
    df_ext_enc[col] = np.log1p(df_ext_enc[col].clip(lower=0))

numeric_cols = [c for c in train_feature_cols
    if not (c.startswith('protocol_type_') or c.startswith('service_') or c.startswith('flag_'))]
binary_cols = [c for c in numeric_cols if c.startswith('is_zero_') or
               c.startswith('is_ftp_telnet') or c == 'is_failed_login']
scaler_cols = [c for c in numeric_cols if c in df_ext_enc.columns and c not in binary_cols]
df_ext_enc[scaler_cols] = scaler.transform(df_ext_enc[scaler_cols])
for col in binary_cols:
    if col in df_ext_enc.columns:
        df_ext_enc[col] = df_ext_enc[col] * 6 - 3

X_ext = df_ext_enc[train_feature_cols].values.astype(np.float32)
print(f"[数据] 外部测试集: {X_ext.shape}, Normal={int((y_binary==0).sum())}, Attack={int((y_binary==1).sum())}")

# ===== 5. Stage1 推理，获取攻击概率 =====
X_tensor = torch.tensor(X_ext).to(device)
with torch.no_grad():
    logits = model_s1(X_tensor)  # (N, 2): [normal_logit, attack_logit]
    probs = torch.softmax(logits, dim=1)  # (N, 2)
    attack_probs = probs[:, 1].cpu().numpy()  # 攻击概率

# ===== 6. 计算 ROC 曲线和 AUC =====
fpr, tpr, thresholds = roc_curve(y_binary, attack_probs)
roc_auc = auc(fpr, tpr)

# 找最优阈值（Youden's J = TPR - FPR）
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

print(f"\n[ROC] AUC = {roc_auc:.4f}")
print(f"[ROC] 最优阈值 (Youden): {best_threshold:.4f}")
print(f"[ROC]   TPR = {best_tpr:.4f}, FPR = {best_fpr:.4f}")

# ===== 7. 绘图 =====
plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random (AUC = 0.5)')

# 标记最优阈值点
plt.scatter(best_fpr, best_tpr, color='red', s=80, zorder=5,
            label=f'Best threshold = {best_threshold:.3f}\n(TPR={best_tpr:.3f}, FPR={best_fpr:.3f})')

# 标记训练时使用的阈值 s1=0.5
s1_idx = np.argmin(np.abs(thresholds - 0.5))
s1_fpr_pt = fpr[s1_idx]
s1_tpr_pt = tpr[s1_idx]
plt.scatter(s1_fpr_pt, s1_tpr_pt, color='green', s=60, zorder=5, marker='s',
            label=f's1=0.50 (TPR={s1_tpr_pt:.3f}, FPR={s1_fpr_pt:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Normal → Attack)', fontsize=12)
plt.ylabel('True Positive Rate (Attack Detected)', fontsize=12)
plt.title(f'Stage1 ROC Curve — External Test Set\nNormal vs Attack (Binary Classification)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)

out_path = os.path.join(base_dir, 'Visualization', f'roc_stage1_{test_basename}.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[保存] {out_path}")
print("Done.")
