"""
双阈值网格扫描：s1_threshold × s2_threshold
只加载一次模型和数据，遍历所有组合
"""
import os, sys, random, argparse
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

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

s1_base = BinaryDNN(embedded_dim, hidden_dims=[128, 64], dropout=0.3,
                    use_residual=False, activation='relu')
model_s1 = ServiceEmbeddingModel(s1_base, vocab_size=69, feature_cols=feature_cols).to(device)
model_s1.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage1_best.pth'), map_location=device))

s2_base = DNN(embedded_dim, 23, hidden_dims=[256, 128, 64], dropout_rate=0.3)
model_s2 = ServiceEmbeddingModel(s2_base, vocab_size=69, feature_cols=feature_cols).to(device)
model_s2.load_state_dict(torch.load(os.path.join(base_dir, 'model_stage2_best.pth'), map_location=device))

orig_attack_labels = sorted([c for c in range(24) if c != 11 and c != 23])
s2_reverse = {new: orig for new, orig in enumerate(orig_attack_labels)}
s2_reverse[22] = 11

# Load and preprocess external test set
parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='train_test',
                    help='外部测试文件（Train目录下），如 train_test / KDDTest+ / KDDTest-21')
args = parser.parse_args()

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

ext_path = os.path.join(base_dir, 'Train', args.test_file)
test_basename = args.test_file.replace('.txt', '').replace('.csv', '')
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
df_ext_enc['is_failed_login'] = (
    (df_ext_enc['num_failed_logins'] > 0) & (df_ext_enc['logged_in'] == 0)
).astype(int)

ZERO_INFLATED_COLS = ['src_bytes','dst_bytes','duration','num_failed_logins','num_shells','num_access_files','num_file_creations','num_root']
for col in ZERO_INFLATED_COLS:
    df_ext_enc['is_zero_'+col] = (df_ext_enc[col] == 0).astype(int)
for col in ['src_bytes','dst_bytes','duration','hot']:
    df_ext_enc[col] = np.log1p(df_ext_enc[col].clip(lower=0))

numeric_cols = [c for c in train_feature_cols
    if not (c.startswith('protocol_type_') or c.startswith('service_') or c.startswith('flag_'))]
# 二值特征不参与RobustScaler，改为手动映射 0→-3, 1→+3
binary_cols = [c for c in numeric_cols if c.startswith('is_zero_') or c.startswith('is_ftp_telnet') or c == 'is_failed_login']
scaler_cols = [c for c in numeric_cols if c in df_ext_enc.columns and c not in binary_cols]
df_ext_enc[scaler_cols] = scaler.transform(df_ext_enc[scaler_cols])
for col in binary_cols:
    if col in df_ext_enc.columns:
        df_ext_enc[col] = df_ext_enc[col] * 6 - 3

X_ext = df_ext_enc[train_feature_cols].values.astype(np.float32)
print(f"外部数据: {X_ext.shape}, 标签: {len(y_ext)}")

# 基线指标
n_uk = (y_ext == 23).sum()
print(f"\nunknown_attack 样本: {n_uk}")

# ============ 网格扫描 ============
s1_range = [0.3, 0.4, 0.5, 0.6, 0.7]
s2_range = [0.5, 0.6, 0.7, 0.75, 0.8]

print(f"\n{'='*70}")
print(f"双阈值网格扫描: s1 ∈ {s1_range}, s2 ∈ {s2_range}")
print(f"{'='*70}")
print(f"{'s1_th':>6}  {'s2_th':>6}  {'Acc':>8}  {'F1':>8}  {'UK_recall':>10}  {'normal→attack':>14}  {'normal→satan':>13}")
print(f"{'-'*70}")

results = []
for s1_th in s1_range:
    for s2_th in s2_range:
        y_pred, s1_prob, s2_prob = two_stage_predict(
            model_s1, model_s2, X_ext, s2_reverse,
            s1_threshold=s1_th, s2_threshold=s2_th)

        acc = accuracy_score(y_ext, y_pred)
        f1 = f1_score(y_ext, y_pred, average='weighted', zero_division=0)

        # unknown_attack recall
        n_uk_caught = ((y_pred == 23) & (y_ext == 23)).sum()
        uk_recall = n_uk_caught / n_uk if n_uk > 0 else 0

        # normal misclassified as attack (coming through Stage1)
        normal_mask = (y_ext == 11)
        normal_to_attack = ((normal_mask) & (y_pred != 11)).sum()

        # normal→satan specifically
        normal_to_satan = ((normal_mask) & (y_pred == 2)).sum()

        results.append({
            's1_th': s1_th, 's2_th': s2_th,
            'acc': acc, 'f1': f1,
            'uk_recall': uk_recall,
            'normal_to_attack': normal_to_attack,
            'normal_to_satan': normal_to_satan,
        })

        print(f"{s1_th:>6.2f}  {s2_th:>6.2f}  {acc:>8.4f}  {f1:>8.4f}  {uk_recall:>10.4f}  {normal_to_attack:>14d}  {normal_to_satan:>13d}")

# 按 Acc 排序输出 Top 10
print(f"\n{'='*70}")
print("Top 10 组合 (按 Acc 排序):")
print(f"{'='*70}")
results.sort(key=lambda x: -x['acc'])
for i, r in enumerate(results[:10]):
    print(f"{i+1:>2}. s1={r['s1_th']:.2f}, s2={r['s2_th']:.2f}  "
          f"Acc={r['acc']:.4f}  F1={r['f1']:.4f}  "
          f"UK_recall={r['uk_recall']:.4f}  "
          f"normal→attack={r['normal_to_attack']}  normal→satan={r['normal_to_satan']}")

# 保存结果
df_results = pd.DataFrame(results)
output = os.path.join(base_dir, 'models', 'two_stage', f'grid_search_{test_basename}.csv')
df_results.to_csv(output, index=False)
print(f"\n结果已保存: {output}")
