import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import argparse
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

# KDD Cup 99 训练集的 23 种流量类型（normal + 22 种具体攻击），按字母序固定。
# 无论使用完整训练集(KDDTrain+.txt, 23 类) 还是 20% 子集(KDDTrain+_20Percent.txt, 缺 perl 仅 22 类)，
# 都以此列表为准做编码，保证类别数恒为 23，模型输出维度与混淆矩阵恒为 23。
MULTICLASS_23_CLASSES = [
    'back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap',
    'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap',
    'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan',
    'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'
]

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
    """标签处理：二分类 + 5大分类 + 23细分类"""
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

    # 4. 23 细分类标签：normal + 22 种具体攻击类型
    # 使用固定的 23 类列表编码（而非 LabelEncoder.fit_transform 自动发现），
    # 保证无论训练集是否包含全部稀有攻击（如 perl），编码空间恒为 23 类。
    # 不在 23 类列表中的标签（例如测试集才出现的新攻击）编码为 -1，后续评估时剔除。
    df['label_multiclass'] = df['label']  # 原始具体类型
    class_to_idx = {name: i for i, name in enumerate(MULTICLASS_23_CLASSES)}
    df['label_multiclass_encoded'] = df['label'].map(class_to_idx).fillna(-1).astype(int)
    # LabelEncoder 仅用于持久化 classes_（与历史产物兼容），在固定列表上 fit
    le_multiclass = LabelEncoder()
    le_multiclass.fit(MULTICLASS_23_CLASSES)

    print("\n【23 分类标签分布】")
    mc_counts = df['label_multiclass'].value_counts()
    print(f"23 分类总类别数: {len(le_multiclass.classes_)}")
    print(mc_counts)

    print("\n【23 分类标签编码映射】")
    for i, class_name in enumerate(le_multiclass.classes_):
        print(f"  {i} -> {class_name}")

    return df, le_category, le_multiclass


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    先切分数据集：训练集 / 验证集 / 测试集
    切分在预处理（标准化）之前完成，避免数据泄漏

    划分比例：训练集70% / 验证集10% / 测试集20%
    使用 23 分类标签做分层采样；当某些类样本过少时，退化为 random split
    """
    print("\n" + "=" * 60)
    print("三、数据集划分（先切分，后标准化，防止数据泄漏）")
    print("=" * 60)

    # 使用 23 分类标签做分层，保证稀有类别在三个集合中都有代表
    y = df['label_multiclass_encoded'].values

    # 检查每个类别的样本数：若少于 2（split 后会 < 1），则改用 random split
    class_counts = pd.Series(y).value_counts()
    too_few_classes = class_counts[class_counts < 2].index.tolist()
    use_stratify = len(too_few_classes) == 0

    if not use_stratify:
        print(f"  ⚠️ 以下 {len(too_few_classes)} 个类别样本数 < 2，无法使用 stratified split：")
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

    # 检查各集标签分布（以 23 分类为例）
    print("\n【23 分类在三个集合的样本数】")
    for name, subset in [('训练集', df_train), ('验证集', df_val), ('测试集', df_test)]:
        dist = subset['label_multiclass'].value_counts()
        print(f"  {name}: 类别数={dist.size}, 正常={int(dist.get('normal', 0))}, 总攻击={len(subset) - int(dist.get('normal', 0))}")

    return df_train, df_val, df_test


def encode_and_scale(df_train, df_val, df_test):
    """
    在训练集上 fit OneHotEncoder 和 StandardScaler，
    对验证集和测试集只做 transform，严格防止数据泄漏
    """
    print("\n" + "=" * 60)
    print("四、特征编码与标准化（仅训练集fit，验证/测试集只transform）")
    print("=" * 60)

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

    # ---------- 数值型特征标准化 ----------
    print("\n[标准化] StandardScaler: 仅在训练集上 fit")

    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_train_enc.columns]
    scaler = StandardScaler()
    scaler.fit(df_train_enc[numeric_cols])

    df_train_scaled = df_train_enc.copy()
    df_val_scaled = df_val_enc.copy()
    df_test_scaled = df_test_enc.copy()

    # 训练集: fit_transform（已经fit了，这里用transform）
    df_train_scaled[numeric_cols] = scaler.transform(df_train_enc[numeric_cols])
    # 验证集: 仅 transform
    df_val_scaled[numeric_cols] = scaler.transform(df_val_enc[numeric_cols])
    # 测试集: 仅 transform
    df_test_scaled[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])

    print(f"  已对 {len(numeric_cols)} 个数值型特征进行标准化")
    print(f"  训练集标准化后均值(前5): {df_train_scaled[numeric_cols[:5]].mean().values.round(4)}")
    print(f"  训练集标准化后标准差(前5): {df_train_scaled[numeric_cols[:5]].std().values.round(4)}")

    return df_train_scaled, df_val_scaled, df_test_scaled, ohe, scaler, ohe_feature_names


def main(data_file='KDDTrain+_20Percent.txt', data_dir='../Train'):
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

    # Step 3: 标签处理（二分类 + 5大分类 + 23细分类）
    df, le_category, le_multiclass = preprocess_labels(df)

    # Step 4: 先切分数据集（在标准化之前！）
    df_train, df_val, df_test = split_data(df, test_size=0.2, val_size=0.1)

    # Step 5: 在训练集上 fit 编码器和标准化器，验证/测试集只 transform
    df_train_processed, df_val_processed, df_test_processed, ohe, scaler, ohe_feature_names = \
        encode_and_scale(df_train, df_val, df_test)

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
    print(f"  - 数值型特征: {len([c for c in NUMERIC_FEATURES if c in feature_cols])} 个（已标准化）")
    print(f"  - One-Hot编码特征: {len(ohe_feature_names)} 个")
    print(f"标签列:")
    print(f"  - label_binary: 二分类 (normal/attack)")
    print(f"  - label_category_encoded: 5大分类 (normal/dos/probe/r2l/u2r)")
    print(f"  - label_multiclass_encoded: 23细分类 (normal + 22种具体攻击)")
    print(f"\n关键原则:")
    print(f"  - StandardScaler 仅在训练集上 fit，验证/测试集只 transform")
    print(f"  - OneHotEncoder 仅在训练集上 fit，验证/测试集只 transform")
    print(f"  - 验证集用于训练过程监控（Early Stopping / Epoch评估）")
    print(f"  - 测试集绝对不参与训练过程，仅做最终评估")

    # 保存数据
    output_dir = "../Train"

    df_train_processed.to_csv(f"{output_dir}/KDDTrain_preprocessed_train.csv", index=False)
    df_val_processed.to_csv(f"{output_dir}/KDDTrain_preprocessed_val.csv", index=False)
    df_test_processed.to_csv(f"{output_dir}/KDDTrain_preprocessed_test.csv", index=False)

    # 保存编码器和标准化器（供未来新数据使用）
    joblib.dump(ohe, f"{output_dir}/encoder_onehot.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler_standard.pkl")
    joblib.dump({'feature_cols': feature_cols, 'ohe_feature_names': list(ohe_feature_names)},
                f"{output_dir}/preprocessing_metadata.pkl")
    # 保存 23 分类标签编码器（供模型推理时将预测索引映射回具体类型）
    joblib.dump(le_multiclass, f"{output_dir}/encoder_multiclass_23.pkl")
    # 额外保存一份纯类别列表（跨 sklearn 版本兼容）
    class_list_path = f"{output_dir}/encoder_multiclass_23_classes.txt"
    with open(class_list_path, 'w') as f:
        for c in le_multiclass.classes_:
            f.write(c + '\n')
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
    print(f"  标准化器: {output_dir}/scaler_standard.pkl")
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
                        help='使用的数据集: full(完整训练集125k) 或 20percent(20%训练集25k)，默认 20percent')
    parser.add_argument('--data_dir', type=str, default='../Train',
                        help='数据目录路径，默认 ../Train')
    args = parser.parse_args()
    
    data_file = 'KDDTrain+.txt' if args.dataset == 'full' else 'KDDTrain+_20Percent.txt'
    train_data, val_data, test_data = main(data_file=data_file, data_dir=args.data_dir)