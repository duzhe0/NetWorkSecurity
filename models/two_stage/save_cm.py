"""
保存 Two-Stage 模型的混淆矩阵（外部测试集），格式与其它模型一致
用法: python save_cm.py --threshold 0.6
"""
import os, sys, random, argparse
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=None, help='统一阈值（同时设置 s1 和 s2）')
parser.add_argument('--s1_threshold', type=float, default=0.5)
parser.add_argument('--s2_threshold', type=float, default=0.5)
args = parser.parse_args()

if args.threshold is not None:
    S1_TH = args.threshold
    S2_TH = args.threshold
else:
    S1_TH = args.s1_threshold
    S2_TH = args.s2_threshold

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)
from models.embedding_utils import ServiceEmbeddingModel, compute_embedded_input_dim, SERVICE_EMBEDDING_DIM
from models.dnn.dnn_model import DNN, CATEGORICAL_FEATURES
from models.two_stage.two_stage_model import BinaryDNN, two_stage_predict

train_dir = os.path.join(base_dir, 'Train')
device = torch.device('cuda')

# Load models
df_train = pd.read_csv(os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1)
exclude_cols = ['label','difficulty','label_binary','label_category','label_category_encoded','label_multiclass','label_multiclass_encoded']
feature_cols = [c for c in df_train.columns if c not in exclude_cols]
embedded_dim = compute_embedded_input_dim(feature_cols, SERVICE_EMBEDDING_DIM)

s1_base = BinaryDNN(embedded_dim, hidden_dims=[128, 64], dropout=0.3)
model_s1 = ServiceEmbeddingModel(s1_base, vocab_size=69, feature_cols=feature_cols).to(device)
model_s1.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage1_best.pth'), map_location=device))

s2_base = DNN(embedded_dim, 23, hidden_dims=[256, 128, 64], dropout_rate=0.3)
model_s2 = ServiceEmbeddingModel(s2_base, vocab_size=69, feature_cols=feature_cols).to(device)
model_s2.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage2_best.pth'), map_location=device))

orig_attack_labels = sorted([c for c in range(24) if c != 11 and c != 23])
s2_reverse = {new: orig for new, orig in enumerate(orig_attack_labels)}
s2_reverse[22] = 11

# Load and preprocess external test set
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

ext_path = os.path.join(base_dir, 'Train', 'train_test')
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
drop_cols = ['num_outbound_cmds','is_host_login','urgent','su_attempted','label','difficulty','mc_enc']
for c in drop_cols:
    if c in df_rest.columns: df_rest = df_rest.drop(columns=[c])
df_ext_enc = pd.concat([df_rest, df_ohe], axis=1)

train_feature_cols = [c for c in pd.read_csv(
    os.path.join(train_dir, 'KDDTrain_preprocessed_train.csv'), nrows=1).columns
    if c not in exclude_cols]
for col in train_feature_cols:
    if col not in df_ext_enc.columns: df_ext_enc[col] = 0.0

# ===== 手工特征（与 data_preprocessing 保持一致） =====
ftp_telnet_cols = [c for c in ['service_ftp','service_telnet','service_ftp_data']
                   if c in df_ext_enc.columns]
df_ext_enc['is_ftp_telnet'] = df_ext_enc[ftp_telnet_cols].max(axis=1) if ftp_telnet_cols else 0
eps = 1e-6
df_ext_enc['svc_auth_fail_score'] = (
    df_ext_enc['is_ftp_telnet'] *
    df_ext_enc['num_failed_logins'] / (df_ext_enc['num_failed_logins'] + df_ext_enc['num_compromised'] + eps)
)
df_ext_enc['serror_logged_cross'] = df_ext_enc['serror_rate'] * df_ext_enc['logged_in']

ZERO_INFLATED_COLS = ['src_bytes','dst_bytes','duration','num_failed_logins','num_shells','num_access_files','num_file_creations','num_root']
for col in ZERO_INFLATED_COLS:
    df_ext_enc['is_zero_'+col] = (df_ext_enc[col] == 0).astype(int)
for col in ['src_bytes','dst_bytes','duration','hot']:
    df_ext_enc[col] = np.log1p(df_ext_enc[col].clip(lower=0))

numeric_cols = [c for c in train_feature_cols
    if not (c.startswith('protocol_type_') or c.startswith('service_') or c.startswith('flag_'))]
scaler_cols = [c for c in numeric_cols if c in df_ext_enc.columns]
df_ext_enc[scaler_cols] = scaler.transform(df_ext_enc[scaler_cols])

X_ext = df_ext_enc[train_feature_cols].values.astype(np.float32)
print(f"外部数据: {X_ext.shape}, 标签: {len(y_ext)}")

# Predict
print(f"使用双阈值: s1={S1_TH}, s2={S2_TH}")
y_pred, _, _ = two_stage_predict(model_s1, model_s2, X_ext, s2_reverse,
                                  s1_threshold=S1_TH, s2_threshold=S2_TH)

# Build full 24-class labels (0-23)
all_labels = list(range(24))

# Compute confusion matrix
cm = confusion_matrix(y_ext, y_pred, labels=all_labels)
print(f"混淆矩阵: {cm.shape}, 总样本: {cm.sum()}")

# Accuracy
acc = cm.diagonal().sum() / cm.sum()
print(f"Accuracy: {acc:.4f}")

# Save as CSV (same format as other models)
# 双阈值时用 s1_s2 格式命名；单阈值时用 th 格式命名（向后兼容）
if args.threshold is not None:
    th_suffix = f"_th{str(args.threshold).replace('.', '')}"
else:
    th_suffix = f"_s1{str(S1_TH).replace('.', '')}_s2{str(S2_TH).replace('.', '')}"
output_csv = os.path.join(base_dir, 'models', 'two_stage', f'results_two_stage_external_test{th_suffix}.csv')
df_result = pd.DataFrame([{
    'test_acc': acc,
    'test_samples': int(cm.sum()),
    'test_file': 'train_test',
    'confusion_matrix': cm.tolist()
}])
df_result.to_csv(output_csv, index=False)
print(f"混淆矩阵已保存: {output_csv}")

# Save .npy for visualization tools
np.save(os.path.join(base_dir, 'models', 'two_stage', f'cm_external{th_suffix}.npy'), cm)
print("npy 文件已保存")
