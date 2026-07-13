import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
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
    """标签处理：二分类 + 多分类"""
    print("\n" + "=" * 60)
    print("二、标签预处理")
    print("=" * 60)

    # 1. 二分类标签：normal=0, attack=1
    df['label_binary'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    print("\n【二分类标签分布】  normal=0, attack=1")
    print(df['label_binary'].value_counts())
    print(f"正常流量占比: {df['label_binary'].mean() * 100:.2f}%")
    print(f"攻击流量占比: {(1 - df['label_binary'].mean()) * 100:.2f}%")

    # 2. 多分类标签：normal / dos / probe / r2l / u2r
    df['label_category'] = df['label'].map(ATTACK_CATEGORIES)
    df['label_category'] = df['label_category'].fillna('other')

    print("\n【多分类标签分布（5大类）】")
    print(df['label_category'].value_counts())

    # 3. 多分类数值编码
    le = LabelEncoder()
    df['label_category_encoded'] = le.fit_transform(df['label_category'])
    print("\n【多分类标签编码映射】")
    for i, class_name in enumerate(le.classes_):
        print(f"  {i} -> {class_name}")

    return df, le


def encode_categorical_features(df):
    """类别型特征编码：One-Hot编码"""
    print("\n" + "=" * 60)
    print("三、类别型特征编码（One-Hot Encoding）")
    print("=" * 60)

    print(f"\n编码前特征数量: {df.shape[1]}")

    # 对 protocol_type, service, flag 进行 One-Hot 编码
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES,
                                prefix=CATEGORICAL_FEATURES, drop_first=False)

    print(f"编码后特征数量: {df_encoded.shape[1]}")
    print(f"新增 One-Hot 特征数量: {df_encoded.shape[1] - df.shape[1]}")

    # 打印新增的列名
    new_cols = [col for col in df_encoded.columns if any(
        col.startswith(prefix + '_') for prefix in CATEGORICAL_FEATURES)]
    print(f"\n新增的 One-Hot 特征列 ({len(new_cols)} 个):")
    for col in new_cols:
        print(f"  {col}")

    return df_encoded


def scale_numeric_features(df, feature_cols):
    """数值型特征标准化"""
    print("\n" + "=" * 60)
    print("四、数值型特征标准化（StandardScaler）")
    print("=" * 60)

    scaler = StandardScaler()
    df_scaled = df.copy()

    # 只对数值型特征做标准化
    numeric_cols_in_df = [col for col in feature_cols if col in df.columns]
    df_scaled[numeric_cols_in_df] = scaler.fit_transform(df[numeric_cols_in_df])

    print(f"已对 {len(numeric_cols_in_df)} 个数值型特征进行标准化")
    print("\n标准化前部分特征统计（前5个）:")
    print(df[numeric_cols_in_df[:5]].describe().round(3))
    print("\n标准化后部分特征统计（前5个）:")
    print(df_scaled[numeric_cols_in_df[:5]].describe().round(3))

    return df_scaled, scaler


def main():
    """主函数：完整的数据预处理流程"""
    data_path = "Train_Dataset/KDDTrain+_20Percent.txt"

    print("=" * 60)
    print("KDD Cup 99 网络入侵检测 - 数据预处理")
    print("=" * 60)

    # Step 1: 加载数据
    df = load_data(data_path)

    # Step 2: 数据探索
    df = explore_data(df)

    # Step 3: 标签处理（二分类 + 多分类）
    df, label_encoder = preprocess_labels(df)

    # Step 4: 类别型特征 One-Hot 编码
    df_encoded = encode_categorical_features(df)

    # 准备特征矩阵 X（排除原始标签和辅助列）
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]

    # Step 5: 数值型特征标准化
    df_processed, scaler = scale_numeric_features(df_encoded, feature_cols)

    # ========== 预处理结果汇总 ==========
    print("\n" + "=" * 60)
    print("数据预处理结果汇总")
    print("=" * 60)

    print(f"\n原始数据集形状: {df.shape}")
    print(f"预处理后数据集形状: {df_processed.shape}")
    print(f"特征矩阵列数: {len(feature_cols)}")
    print(f"  - 数值型特征: {len(NUMERIC_FEATURES)} 个（已标准化）")
    print(f"  - One-Hot编码特征: {len(feature_cols) - len(NUMERIC_FEATURES)} 个")
    print(f"标签列: label_binary(二分类), label_category_encoded(多分类)")
    print(f"难度等级列: difficulty（已保留，可选用于过滤）")

    # 保存预处理后的数据
    output_path = "Train_Dataset/KDDTrain_preprocessed.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"\n预处理后的数据已保存至: {output_path}")

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)

    return df_processed


if __name__ == '__main__':
    processed_data = main()
