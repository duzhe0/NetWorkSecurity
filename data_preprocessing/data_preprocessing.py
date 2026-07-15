import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

ATTACK_CATEGORIES = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'procmon': 'dos',
    'udpstorm': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'worm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
    'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
    'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'httptunnel': 'r2l',
    'named': 'r2l', 'sendmail': 'r2l', 'snmpget': 'r2l', 'xsnoop': 'r2l',
    'xlock': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]


def load_data(data_path):
    df = pd.read_csv(data_path, names=COLUMN_NAMES, header=None)
    return df


def explore_data(df):
    return df


def preprocess_labels(df):
    df = df.copy()
    df['label_binary'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df['label_category'] = df['label'].map(ATTACK_CATEGORIES).fillna('normal')
    le_category = LabelEncoder()
    df['label_category_encoded'] = le_category.fit_transform(df['label_category'])
    df['label_multiclass'] = df['label']
    le_multiclass = LabelEncoder()
    df['label_multiclass_encoded'] = le_multiclass.fit_transform(df['label_multiclass'])
    return df, le_category, le_multiclass


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    stratify_col = 'label_multiclass_encoded'
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[stratify_col]
    )
    val_relative = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_relative, random_state=random_state,
        stratify=df_train_val[stratify_col]
    )
    return df_train, df_val, df_test


def encode_and_scale(df_train, df_val, df_test):
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df_train[CATEGORICAL_FEATURES])

    train_ohe = ohe.transform(df_train[CATEGORICAL_FEATURES])
    val_ohe = ohe.transform(df_val[CATEGORICAL_FEATURES])
    test_ohe = ohe.transform(df_test[CATEGORICAL_FEATURES])

    ohe_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    df_train_ohe = pd.DataFrame(train_ohe, columns=ohe_names, index=df_train.index)
    df_val_ohe = pd.DataFrame(val_ohe, columns=ohe_names, index=df_val.index)
    df_test_ohe = pd.DataFrame(test_ohe, columns=ohe_names, index=df_test.index)

    scaler = StandardScaler()
    scaler.fit(df_train[NUMERIC_FEATURES])

    train_scaled = scaler.transform(df_train[NUMERIC_FEATURES])
    val_scaled = scaler.transform(df_val[NUMERIC_FEATURES])
    test_scaled = scaler.transform(df_test[NUMERIC_FEATURES])

    df_train_scaled = pd.DataFrame(train_scaled, columns=NUMERIC_FEATURES, index=df_train.index)
    df_val_scaled = pd.DataFrame(val_scaled, columns=NUMERIC_FEATURES, index=df_val.index)
    df_test_scaled = pd.DataFrame(test_scaled, columns=NUMERIC_FEATURES, index=df_test.index)

    label_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                  'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']

    df_train_p = pd.concat([df_train_scaled, df_train_ohe, df_train[label_cols]], axis=1)
    df_val_p = pd.concat([df_val_scaled, df_val_ohe, df_val[label_cols]], axis=1)
    df_test_p = pd.concat([df_test_scaled, df_test_ohe, df_test[label_cols]], axis=1)

    return df_train_p, df_val_p, df_test_p, ohe, scaler, ohe_names
