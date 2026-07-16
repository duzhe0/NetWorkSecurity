import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# 1. KDD Cup 99 数据集列名定义（41个特征 + 1个标签 + 1个难度）
# ============================================================
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

# ============================================================
# 2. 攻击类型映射（22种具体攻击 -> 4大类 + normal）
# ============================================================
ATTACK_CATEGORIES = {
    # DoS (拒绝服务攻击)
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
    'processtable': 'dos', 'worm': 'dos',
    # R2L (远程未授权访问)
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
    'warezmaster': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    # U2R (本地提权攻击)
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r',
    # Probe (探测/扫描)
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    # Normal (正常流量)
    'normal': 'normal'
}

# 类别型特征
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# 数值型特征
NUMERIC_FEATURES = [col for col in COLUMN_NAMES[:-2] if col not in CATEGORICAL_FEATURES]


def load_data(file_path):
    """加载KDD数据集"""
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
    print(f"[数据加载] 数据集形状: {df.shape}")
    print(f"[数据加载] 样本数量: {df.shape[0]}")
    print(f"[数据加载] 特征数量: {len(COLUMN_NAMES) - 2}")
    return df


def explore_data(df):
    """数据探索"""
    print("\n" + "=" * 60)
    print("一、数据探索")
    print("=" * 60)

    print("\n【前5行数据】")
    print(df.head())

    print("\n【数据基本信息】")
    print(df.info())

    print("\n【数值型特征统计描述】")
    print(df[NUMERIC_FEATURES].describe().round(2))

    print("\n【标签分布（全部）】")
    label_counts = df['label'].value_counts()
    print(f"标签种类数: {len(label_counts)}")
    print(label_counts)

    print("\n【协议类型分布】")
    print(df['protocol_type'].value_counts())

    print("\n【连接状态(flag)分布】")
    print(df['flag'].value_counts())

    print("\n【服务类型分布 (前10)】")
    print(df['service'].value_counts().head(10))

    print("\n【缺失值检查】")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("数据集中无缺失值")
    else:
        print(missing[missing > 0])

    return df


def preprocess_labels(df):
    """标签处理：二分类 + 5大分类 + 24细分类"""
    print("\n" + "=" * 60)
    print("二、标签预处理")
    print("=" * 60)

    # 1. 二分类标签：normal=0, attack=1
    df['label_binary'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    print("\n【二分类标签分布】  normal=0, attack=1")
    print(df['label_binary'].value_counts())
    print(f"正常流量占比: {(1 - df['label_binary'].mean()) * 100:.2f}%")
    print(f"攻击流量占比: {df['label_binary'].mean() * 100:.2f}%")

    # 2. 多分类标签：normal / dos / probe / r2l / u2r
    df['label_category'] = df['label'].map(ATTACK_CATEGORIES)
    df['label_category'] = df['label_category'].fillna('other')

    print("\n【多分类标签分布（5大类）】")
    print(df['label_category'].value_counts())

    # 3. 5大分类数值编码
    le_category = LabelEncoder()
    df['label_category_encoded'] = le_category.fit_transform(df['label_category'])
    print("\n【5大分类标签编码映射】")
    for i, class_name in enumerate(le_category.classes_):
        print(f"  {i} -> {class_name}")

    # 4. 24 细分类标签：normal + 22 种具体攻击 + unknown_attack
    df['label_multiclass'] = df['label']  # 原始具体类型

    # 从 .txt 文件读取标准类别列表（含 perl 和 unknown_attack），按文件顺序编码
    class_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'Train', 'encoder_multiclass_23_classes.txt')
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            canonical_classes = [line.strip() for line in f if line.strip()]
        # 未知类型（不在文件中的）统一映射到最后一类（unknown_attack）
        unknown_idx = len(canonical_classes) - 1
        label_to_idx = {name: i for i, name in enumerate(canonical_classes)}
        df['label_multiclass_encoded'] = df['label_multiclass'].map(label_to_idx).fillna(unknown_idx).astype(int)
        print(f"\n[标签映射] 从 {class_file} 加载 {len(canonical_classes)} 类标准映射")
    else:
        le_multiclass = LabelEncoder()
        df['label_multiclass_encoded'] = le_multiclass.fit_transform(df['label_multiclass'])
        canonical_classes = list(le_multiclass.classes_)

    print("\n【24 分类标签分布（训练集实际出现 23 类）】")
    mc_counts = df['label_multiclass'].value_counts()
    actual_labels = sorted(df['label_multiclass_encoded'].unique())
    print(f"总类别数（含 unknown_attack）: {len(canonical_classes)}, 实际出现: {len(actual_labels)}")
    for idx in actual_labels:
        name = canonical_classes[idx] if idx < len(canonical_classes) else f"index_{idx}"
        count = (df['label_multiclass_encoded'] == idx).sum()
        print(f"  {idx}: {name} = {count}")
    if len(actual_labels) < len(canonical_classes):
        missing = set(range(len(canonical_classes))) - set(actual_labels)
        print(f"训练集中缺失的类别索引: {sorted(missing)}")
        for m in sorted(missing):
            name = canonical_classes[m] if m < len(canonical_classes) else f"index_{m}"
            print(f"  {m}: {name} (0 样本)")

    print("\n【24 分类标签标准映射】")
    for i, class_name in enumerate(canonical_classes):
        has_sample = "[Y]" if i in actual_labels else "[N]"
        print(f"  {i} -> {class_name} {has_sample}")

    return df, le_category, canonical_classes


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    先切分数据集：训练集 / 验证集 / 测试集
    切分在预处理（标准化）之前完成，避免数据泄漏

    划分比例：训练集70% / 验证集10% / 测试集20%
    使用 24 分类标签做分层采样；当某些类样本过少时，退化为 random split
    """
    print("\n" + "=" * 60)
    print("三、数据集划分（先切分，后标准化，防止数据泄漏）")
    print("=" * 60)

    # 使用 24 分类标签做分层，保证稀有类别在三个集合中都有代表
    y = df['label_multiclass_encoded'].values

    # 检查每个类别的样本数：若少于 2（split 后会 < 1），则改用 random split
    class_counts = pd.Series(y).value_counts()
    too_few_classes = class_counts[class_counts < 2].index.tolist()
    use_stratify = len(too_few_classes) == 0

    if not use_stratify:
        print(f"  [WARNING] 以下 {len(too_few_classes)} 个类别样本数 < 2，无法使用 stratified split：")
        for c in too_few_classes:
            class_name = df.loc[df['label_multiclass_encoded'] == c, 'label_multiclass'].iloc[0]
            print(f"      - {class_name} (encoded={c}, count={class_counts[c]})")
        print(f"  改用 random split（不带 stratify）")

    # 第一步：分出测试集（20%）
    if use_stratify:
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

    # 第二步：从剩余数据中分出验证集
    # val_size 相对于全量数据，所以相对 train_val 的比例为 val_size / (1 - test_size)
    val_ratio = val_size / (1 - test_size)
    y_train_val = df_train_val['label_multiclass_encoded'].values
    # 同样判断 train_val 中是否还有过少类别的类
    tv_class_counts = pd.Series(y_train_val).value_counts()
    tv_too_few = tv_class_counts[tv_class_counts < 2].index.tolist()
    tv_use_stratify = len(tv_too_few) == 0

    if tv_use_stratify:
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
        )
    else:
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_ratio, random_state=random_state
        )

    print(f"\n训练集: {len(df_train)} 样本 ({len(df_train)/len(df)*100:.1f}%)")
    print(f"验证集: {len(df_val)} 样本 ({len(df_val)/len(df)*100:.1f}%)")
    print(f"测试集: {len(df_test)} 样本 ({len(df_test)/len(df)*100:.1f}%)")

    # 检查各集标签分布（以 24 分类为例）
    print("\n【24 分类在三个集合的样本数】")
    for name, subset in [('训练集', df_train), ('验证集', df_val), ('测试集', df_test)]:
        dist = subset['label_multiclass'].value_counts()
        print(f"  {name}: 类别数={dist.size}, 正常={int(dist.get('normal', 0))}, 总攻击={len(subset) - int(dist.get('normal', 0))}")

    return df_train, df_val, df_test


def encode_and_scale(df_train, df_val, df_test):
    """
    在训练集上 fit OneHotEncoder 和 RobustScaler，
    对验证集和测试集只做 transform，严格防止数据泄漏
    """
    print("\n" + "=" * 60)
    print("四、特征编码与标准化（仅训练集fit，验证/测试集只transform）")
    print("=" * 60)

    # ========== 特征剔除：移除无信息列 ==========
    DROP_COLS = ['num_outbound_cmds', 'is_host_login', 'urgent', 'su_attempted']
    for col in DROP_COLS:
        if col in df_train.columns:
            df_train = df_train.drop(columns=[col])
            df_val = df_val.drop(columns=[col])
            df_test = df_test.drop(columns=[col])
    print(f"\n[特征剔除] 已移除 {len(DROP_COLS)} 列: {DROP_COLS}")
    print(f"  移除原因: 常数或近似常数（信息量≈0）")
    print(f"  移除后数值特征: {len([c for c in NUMERIC_FEATURES if c not in DROP_COLS])} 列")

    # ========== 零膨胀 Binary Indicator ==========
    # 对大量为 0 的列创建 is_zero 标记，保留"是否为 0"的区分信号
    ZERO_INFLATED_COLS = ['src_bytes', 'dst_bytes', 'duration',
                           'num_failed_logins', 'num_shells', 'num_access_files',
                           'num_file_creations', 'num_root']
    BINARY_INDICATOR_COLS = []
    for col in ZERO_INFLATED_COLS:
        if col in df_train.columns:
            indicator_name = f'is_zero_{col}'
            df_train[indicator_name] = (df_train[col] == 0).astype(int)
            df_val[indicator_name] = (df_val[col] == 0).astype(int)
            df_test[indicator_name] = (df_test[col] == 0).astype(int)
            BINARY_INDICATOR_COLS.append(indicator_name)

    zero_ratio = {col: (df_train[col] == 0).mean() for col in ZERO_INFLATED_COLS if col in df_train.columns}
    print(f"\n[Binary Indicator] 新增 {len(BINARY_INDICATOR_COLS)} 列: {BINARY_INDICATOR_COLS}")
    for col, ratio in zero_ratio.items():
        print(f"  {col}: {ratio:.1%} 的值为 0")

    # ---------- Service LabelEncoder（给 Entity Embedding 用） ----------
    # 保留整数编码的 service，供 DL 模型的 nn.Embedding 层使用
    # OneHot 的 68 列 service_* 仍保留给 XGBoost
    service_le = LabelEncoder()
    service_le.fit(df_train['service'])
    SERVICE_VOCAB_SIZE = len(service_le.classes_) + 1  # +1 for UNK (unknown service)

    def _safe_service_encode(series, encoder, unk_idx):
        """编码 service，未见过的值映射到 UNK 索引"""
        result = []
        class_set = set(encoder.classes_)
        for val in series:
            if val in class_set:
                result.append(encoder.transform([val])[0])
            else:
                result.append(unk_idx)
        return np.array(result, dtype=np.int64)

    UNK_SERVICE_IDX = SERVICE_VOCAB_SIZE - 1
    service_le.vocab_size = SERVICE_VOCAB_SIZE
    service_le.unk_idx = UNK_SERVICE_IDX
    df_train['service_encoded'] = service_le.transform(df_train['service'])
    df_val['service_encoded'] = _safe_service_encode(df_val['service'], service_le, UNK_SERVICE_IDX)
    df_test['service_encoded'] = _safe_service_encode(df_test['service'], service_le, UNK_SERVICE_IDX)

    oov_train = np.sum(df_train['service_encoded'] == UNK_SERVICE_IDX)
    oov_val = np.sum(df_val['service_encoded'] == UNK_SERVICE_IDX)
    oov_test = np.sum(df_test['service_encoded'] == UNK_SERVICE_IDX)
    print(f"\n[Service Embedding] LabelEncoder fitted, vocab_size={SERVICE_VOCAB_SIZE} (含UNK={UNK_SERVICE_IDX})")
    print(f"  OOV样本: train={oov_train}, val={oov_val}, test={oov_test}")

    # ---------- One-Hot 编码 ----------
    print("\n[One-Hot 编码] 使用 sklearn OneHotEncoder（可持久化，对新数据一致）")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    ohe.fit(df_train[CATEGORICAL_FEATURES])

    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    print(f"  训练集类别数: protocol_type={df_train['protocol_type'].nunique()}, "
          f"service={df_train['service'].nunique()}, flag={df_train['flag'].nunique()}")
    print(f"  One-Hot 编码后新增特征数: {len(ohe_feature_names)}")

    def apply_ohe(df_subset):
        ohe_array = ohe.transform(df_subset[CATEGORICAL_FEATURES])
        df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_subset.index)
        df_rest = df_subset.drop(columns=CATEGORICAL_FEATURES)
        return pd.concat([df_rest, df_ohe], axis=1)

    df_train_enc = apply_ohe(df_train)
    df_val_enc = apply_ohe(df_val)
    df_test_enc = apply_ohe(df_test)

    # 对齐列：确保验证集和测试集的列与训练集完全一致
    # handle_unknown='ignore' 已处理未知类别（输出全0），但需补齐缺失列
    for col in df_train_enc.columns:
        if col not in df_val_enc.columns:
            df_val_enc[col] = 0
        if col not in df_test_enc.columns:
            df_test_enc[col] = 0

    df_val_enc = df_val_enc[df_train_enc.columns]
    df_test_enc = df_test_enc[df_train_enc.columns]

    # ========== 手工特征工程：针对 R2L（guess_passwd）增强 ==========
    # 1. is_ftp_telnet: service 是 ftp/telnet/ftp_data 的二元指示
    ftp_telnet_cols = [c for c in ['service_ftp', 'service_telnet', 'service_ftp_data']
                       if c in df_train_enc.columns]
    df_train_enc['is_ftp_telnet'] = df_train_enc[ftp_telnet_cols].max(axis=1) if ftp_telnet_cols else 0
    if ftp_telnet_cols:
        df_val_enc['is_ftp_telnet'] = df_val_enc[ftp_telnet_cols].max(axis=1)
        df_test_enc['is_ftp_telnet'] = df_test_enc[ftp_telnet_cols].max(axis=1)
    else:
        df_val_enc['is_ftp_telnet'] = 0
        df_test_enc['is_ftp_telnet'] = 0

    # 2. svc_auth_fail_score: ftp/telnet 服务下的登录失败比例
    eps = 1e-6
    df_train_enc['svc_auth_fail_score'] = (
        df_train_enc['is_ftp_telnet'] *
        df_train_enc['num_failed_logins'] / (df_train_enc['num_failed_logins'] + df_train_enc['num_compromised'] + eps)
    )
    df_val_enc['svc_auth_fail_score'] = (
        df_val_enc['is_ftp_telnet'] *
        df_val_enc['num_failed_logins'] / (df_val_enc['num_failed_logins'] + df_val_enc['num_compromised'] + eps)
    )
    df_test_enc['svc_auth_fail_score'] = (
        df_test_enc['is_ftp_telnet'] *
        df_test_enc['num_failed_logins'] / (df_test_enc['num_failed_logins'] + df_test_enc['num_compromised'] + eps)
    )

    # 3. serror_logged_cross: SYN错误率 × 登录状态 交叉项
    df_train_enc['serror_logged_cross'] = df_train_enc['serror_rate'] * df_train_enc['logged_in']
    df_val_enc['serror_logged_cross'] = df_val_enc['serror_rate'] * df_val_enc['logged_in']
    df_test_enc['serror_logged_cross'] = df_test_enc['serror_rate'] * df_test_enc['logged_in']

    # 4. is_failed_login: (有登录尝试) & (未成功登录) — guess_passwd 的核心信号
    df_train_enc['is_failed_login'] = (
        (df_train_enc['num_failed_logins'] > 0) & (df_train_enc['logged_in'] == 0)
    ).astype(int)
    df_val_enc['is_failed_login'] = (
        (df_val_enc['num_failed_logins'] > 0) & (df_val_enc['logged_in'] == 0)
    ).astype(int)
    df_test_enc['is_failed_login'] = (
        (df_test_enc['num_failed_logins'] > 0) & (df_test_enc['logged_in'] == 0)
    ).astype(int)

    print(f"\n[特征增强] 新增 4 个手工特征（针对 R2L/guess_passwd）:")
    print(f"  is_ftp_telnet:       service∈{{ftp,telnet,ftp_data}} 的二元指示")
    print(f"  svc_auth_fail_score: ftp/telnet 下的 num_failed_logins/(num_failed_logins+num_compromised+ε)")
    print(f"  serror_logged_cross: serror_rate × logged_in 交叉项")
    print(f"  is_failed_login:     (num_failed_logins>0) & (logged_in==0) — 登录失败标志")

    # ========== Log1p 变换：压缩极端偏态分布 ==========
    LOGP1_COLS = ['src_bytes', 'dst_bytes', 'duration', 'hot']
    for col in LOGP1_COLS:
        if col in df_train_enc.columns:
            df_train_enc[col] = np.log1p(df_train_enc[col])
            df_val_enc[col] = np.log1p(df_val_enc[col])
            df_test_enc[col] = np.log1p(df_test_enc[col])
    applied_cols = [c for c in LOGP1_COLS if c in df_train_enc.columns]
    print(f"\n[Log1p 变换] 对 {len(applied_cols)} 个偏态列应用 log1p: {applied_cols}")
    for col in applied_cols:
        print(f"  {col}: min={df_train_enc[col].min():.2f}, max={df_train_enc[col].max():.2f}, "
              f"median={df_train_enc[col].median():.2f}")

    # ---------- 数值型特征缩放 ----------
    print("\n[缩放] RobustScaler: 仅在训练集上 fit（中位数/IQR，抗异常值）")

    # 二值特征不参与缩放（is_zero_*, is_ftp_telnet, is_failed_login 均为 0/1）
    AUTH_BINARY_COLS = ['is_ftp_telnet', 'is_failed_login']
    AUTH_CONTINUOUS_COLS = ['svc_auth_fail_score', 'serror_logged_cross']
    BINARY_COLS_ALL = BINARY_INDICATOR_COLS + [c for c in AUTH_BINARY_COLS if c in df_train_enc.columns]

    # 连续数值特征：原始数值列 + 认证连续特征
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_train_enc.columns]
    numeric_cols.extend([c for c in AUTH_CONTINUOUS_COLS if c in df_train_enc.columns])

    print(f"  连续特征缩放: {len(numeric_cols)} 列")
    print(f"  二值特征跳过缩放: {len(BINARY_COLS_ALL)} 列（{BINARY_COLS_ALL}）")
    scaler = RobustScaler()
    scaler.fit(df_train_enc[numeric_cols])

    df_train_scaled = df_train_enc.copy()
    df_val_scaled = df_val_enc.copy()
    df_test_scaled = df_test_enc.copy()

    # 仅对连续特征做 transform，二值特征保持原值
    df_train_scaled[numeric_cols] = scaler.transform(df_train_enc[numeric_cols])
    df_val_scaled[numeric_cols] = scaler.transform(df_val_enc[numeric_cols])
    df_test_scaled[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])

    print(f"  已对 {len(numeric_cols)} 个连续数值特征进行缩放")
    print(f"  训练集缩放后中位数(前5): {np.median(df_train_scaled[numeric_cols[:5]].values, axis=0).round(4)}")
    print(f"  训练集缩放后IQR(前5): {np.subtract(*np.percentile(df_train_scaled[numeric_cols[:5]].values, [75, 25], axis=0)).round(4)}")

    # 二值特征映射到 ±3：0→-3, 1→+3，与连续特征缩放后的值域对齐
    BINARY_MAP = {-3: 0, 3: 1}
    for col in BINARY_COLS_ALL:
        df_train_scaled[col] = df_train_scaled[col] * 6 - 3
        df_val_scaled[col] = df_val_scaled[col] * 6 - 3
        df_test_scaled[col] = df_test_scaled[col] * 6 - 3
    print(f"  二值特征映射: 0→-3, 1→+3 ({len(BINARY_COLS_ALL)} 列)")

    return df_train_scaled, df_val_scaled, df_test_scaled, ohe, scaler, ohe_feature_names, service_le


def oversample_rare_classes(df_train, label_col='label_multiclass_encoded', min_samples=200, noise_sigma=0.05):
    """对训练集中样本数 < min_samples 的稀有类做噪声过采样。

    策略：
    - 数值型列：加高斯噪声 N(0, noise_sigma)
    - Binary Indicator 列：保持原值（不引入中间值）
    - OneHot 列：保持原值（不破坏 one-hot 结构）

    Returns:
        df_augmented, oversample_stats: 增强后的DataFrame和统计信息
    """
    import numpy as np

    y = df_train[label_col].values.astype(int)
    class_counts = {}
    for c in sorted(set(y)):
        class_counts[c] = int((y == c).sum())

    # 找出需要过采样的稀有类
    rare_classes = {c: cnt for c, cnt in class_counts.items() if cnt < min_samples}
    if not rare_classes:
        print("[过采样] 所有类样本数 >= min_samples，无需过采样")
        return df_train, {}

    print(f"[过采样] 发现 {len(rare_classes)} 个稀有类（< {min_samples} 样本）:")
    for c, cnt in sorted(rare_classes.items(), key=lambda x: x[1]):
        print(f"  类 {c}: {cnt} -> {min_samples}")

    # 区分列类型
    exclude_cols = [label_col, 'label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded',
                    'label_multiclass', 'label_multiclass_encoded']
    # 只对 feature 列做增强
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    binary_cols = [c for c in feature_cols if c.startswith('is_zero_')]
    ohe_protocol = [c for c in feature_cols if c.startswith('protocol_type_')]
    ohe_service = [c for c in feature_cols if c.startswith('service_')]
    ohe_flag = [c for c in feature_cols if c.startswith('flag_')]
    ohe_cols = ohe_protocol + ohe_service + ohe_flag
    numeric_cols = [c for c in feature_cols
                    if c not in binary_cols and c not in ohe_cols and c != 'service_encoded']

    print(f"  数值列: {len(numeric_cols)}, 二元列: {len(binary_cols)}, OneHot: {len(ohe_cols)}")

    # 生成增强样本
    new_rows = []
    rng = np.random.RandomState(42)
    stats = {}

    for cls, cnt in sorted(rare_classes.items(), key=lambda x: x[1]):
        need = min_samples - cnt
        cls_mask = y == cls
        cls_df = df_train[cls_mask]

        # 对于每个需要生成的样本，随机选一个原样本加噪声
        generated = 0
        for i in range(need):
            idx = rng.randint(0, cnt)
            row = cls_df.iloc[idx].copy()

            # 数值列加噪声
            row[numeric_cols] = row[numeric_cols] + rng.normal(0, noise_sigma, len(numeric_cols))

            # Binary列和OHE列保持不变（已经是0/1）
            new_rows.append(row)
            generated += 1

        stats[cls] = {'before': cnt, 'after': cnt + generated}

    # 拼接
    df_augmented = pd.concat([df_train, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"[过采样] 训练集 {len(df_train)} -> {len(df_augmented)} 条")

    return df_augmented, stats


def main(data_file='KDDTrain+.txt', data_dir='../Train'):
    """主函数：完整的数据预处理流程（防数据泄漏版本）

    Args:
        data_file: 训练数据文件名（KDDTrain+.txt 或 KDDTrain+_20Percent.txt）
        data_dir: 数据目录路径
    """
    data_path = f"{data_dir}/{data_file}"

    print("=" * 60)
    print("KDD Cup 99 网络入侵检测 - 数据预处理")
    print("（防数据泄漏版本：先切分，后拟合）")
    print("=" * 60)
    print(f"使用数据集: {data_file}")

    # Step 1: 加载数据
    df = load_data(data_path)

    # Step 2: 数据探索
    df = explore_data(df)

    # Step 3: 标签处理（二分类 + 5大分类 + 24细分类）
    df, le_category, canonical_classes = preprocess_labels(df)

    # Step 4: 先切分数据集（在标准化之前！）
    df_train, df_val, df_test = split_data(df, test_size=0.2, val_size=0.1)

    # Step 5: 在训练集上 fit 编码器和标准化器，验证/测试集只 transform
    df_train_processed, df_val_processed, df_test_processed, ohe, scaler, ohe_feature_names, service_le = \
        encode_and_scale(df_train, df_val, df_test)

    # Step 5.5: 对训练集稀有类做噪声过采样（验证/测试集不动）
    df_train_processed, oversample_stats = oversample_rare_classes(
        df_train_processed, label_col='label_multiclass_encoded', min_samples=200, noise_sigma=0.05
    )

    # 确定特征列（排除所有标签列：原始label, difficulty, label_binary, label_category, label_category_encoded, label_multiclass, label_multiclass_encoded）
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded',
                    'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [col for col in df_train_processed.columns if col not in exclude_cols]

    # ========== 预处理结果汇总 ==========
    print("\n" + "=" * 60)
    print("数据预处理结果汇总")
    print("=" * 60)

    print(f"\n原始数据集形状: {df.shape}")
    print(f"训练集形状: {df_train_processed.shape}")
    print(f"验证集形状: {df_val_processed.shape}")
    print(f"测试集形状: {df_test_processed.shape}")
    print(f"特征矩阵列数: {len(feature_cols)}")
    print(f"  - 数值型特征: {len([c for c in NUMERIC_FEATURES if c in feature_cols])} 个（已缩放）+ "
          f"{sum(1 for c in feature_cols if 'is_zero_' in c)} 个 Binary Indicator")
    print(f"  - One-Hot编码特征: {len(ohe_feature_names)} 个")
    print(f"  - Entity Embedding: service_encoded（整数索引，供DL模型nn.Embedding）")
    print(f"标签列:")
    print(f"  - label_binary: 二分类 (normal/attack)")
    print(f"  - label_category_encoded: 5大分类 (normal/dos/probe/r2l/u2r)")
    print(f"  - label_multiclass_encoded: 24细分类 (normal + 22种攻击 + unknown_attack)")
    print(f"\n关键原则:")
    print(f"  - RobustScaler 仅在训练集上 fit，验证/测试集只 transform")
    print(f"  - OneHotEncoder 仅在训练集上 fit，验证/测试集只 transform")
    print(f"  - 验证集用于训练过程监控（Early Stopping / Epoch评估）")
    print(f"  - 测试集绝对不参与训练过程，仅做最终评估")

    # 保存数据
    output_dir = data_dir

    df_train_processed.to_csv(f"{output_dir}/KDDTrain_preprocessed_train.csv", index=False)
    df_val_processed.to_csv(f"{output_dir}/KDDTrain_preprocessed_val.csv", index=False)
    df_test_processed.to_csv(f"{output_dir}/KDDTrain_preprocessed_test.csv", index=False)

    # 保存编码器和标准化器（供未来新数据使用）
    joblib.dump(ohe, f"{output_dir}/encoder_onehot.pkl")

    joblib.dump(scaler, f"{output_dir}/scaler_robust.pkl")
    joblib.dump(service_le, f"{output_dir}/encoder_service_embedding.pkl")
    joblib.dump({'feature_cols': feature_cols, 'ohe_feature_names': list(ohe_feature_names)},
                f"{output_dir}/preprocessing_metadata.pkl")
    # 保存 24 分类标签列表（供模型推理时将预测索引映射回具体类型）
    joblib.dump(canonical_classes, f"{output_dir}/encoder_multiclass_23.pkl")
    # .txt 文件已在项目根目录下维护，此处不再覆盖
    # 保存 5 大类标签编码器（normal/dos/probe/r2l/u2r，供 5 分类任务推理映射）
    joblib.dump(le_category, f"{output_dir}/encoder_category_5.pkl")
    class_list_path_5 = f"{output_dir}/encoder_category_5_classes.txt"
    with open(class_list_path_5, 'w') as f:
        for c in le_category.classes_:
            f.write(c + '\n')

    print(f"\n已保存文件:")
    print(f"  训练集: {output_dir}/KDDTrain_preprocessed_train.csv")
    print(f"  验证集: {output_dir}/KDDTrain_preprocessed_val.csv")
    print(f"  测试集: {output_dir}/KDDTrain_preprocessed_test.csv")
    print(f"  OneHot编码器: {output_dir}/encoder_onehot.pkl")
    print(f"  缩放器: {output_dir}/scaler_robust.pkl")
    print(f"  Service Embedding编码器: {output_dir}/encoder_service_embedding.pkl")
    print(f"  预处理元数据: {output_dir}/preprocessing_metadata.pkl")
    print(f"  23分类标签编码器: {output_dir}/encoder_multiclass_23.pkl")
    print(f"  5分类标签编码器: {output_dir}/encoder_category_5.pkl")

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)

    return df_train_processed, df_val_processed, df_test_processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KDD Cup 99 数据预处理')
    parser.add_argument('--dataset', type=str, default='full',
                        choices=['full', '20percent'],
                        help='使用的数据集: full(完整训练集125k) 或 20percent(20%训练集25k)，默认 full')
    parser.add_argument('--data_dir', type=str, default='../Train',
                        help='数据目录路径，默认 ../Train')
    args = parser.parse_args()
    
    data_file = 'KDDTrain+.txt' if args.dataset == 'full' else 'KDDTrain+_20Percent.txt'
    train_data, val_data, test_data = main(data_file=data_file, data_dir=args.data_dir)