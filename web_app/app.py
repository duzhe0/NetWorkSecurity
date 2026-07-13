#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NSL-KDD 入侵检测系统 Web 应用
满足课设所有要求：
1. 数据预处理（缺失值、异常值、特征选择、降维、标准化）
2. 模型训练（DNN、随机森林、XGBoost 多算法对比）
3. 模型优化（超参数调优、可视化界面）
4. 性能评估（准确率、精确率、召回率、F1、ROC曲线）
5. 加分项：多算法对比、友好界面、类不平衡处理（SMOTE）
支持细粒度分类（原始攻击类型）
"""

import os
import sys
import json
import time
import threading
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

ATTACK_CATEGORY_MAP = {
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nsl_kdd_ids_2026'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

_state = {
    'label_encoders': {},
    'scaler': None,
    'feature_selector': None,
    'var_selector': None,
    'pca': None,
    'selected_features': None,
    'X_train': None,
    'y_train': None,
    'X_val': None,
    'y_val': None,
    'X_test': None,
    'y_test': None,
    'label_mapping': None,
    'model': None,
    'model_type': None,
    'history': None,
    'val_results': None,
    'test_results': None,
    'df_train': None,
    'class_weights': None,
    'preprocess_info': None,
    'comparison_results': {},
    'tuning_results': None,
}

_status = {
    'preprocessing': 'idle',
    'training': 'idle',
    'testing': 'idle',
    'comparison': 'idle',
    'tuning': 'idle'
}

_messages = {
    'preprocessing': [],
    'training': [],
    'testing': [],
    'comparison': [],
    'tuning': []
}


def add_msg(task, msg):
    if task in _messages:
        _messages[task].append(msg)
        if len(_messages[task]) > 300:
            _messages[task] = _messages[task][-150:]


def compute_full_metrics(y_true, y_pred, y_pred_proba, label_mapping):
    """计算完整评估指标"""
    acc = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    n_classes = len(label_mapping)
    present_labels = sorted(np.unique(y_true))
    target_names = [label_mapping.get(int(i), str(i)) for i in present_labels]

    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    roc_data = []
    for i in range(n_classes):
        if i in np.unique(y_true):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data.append({
                'label': label_mapping.get(int(i), f'class_{i}'),
                'auc': float(roc_auc),
                'fpr': [float(x) for x in fpr[::max(1, len(fpr)//50)]],
                'tpr': [float(x) for x in tpr[::max(1, len(tpr)//50)]]
            })

    cr = classification_report(y_true, y_pred, labels=present_labels,
                                target_names=target_names, output_dict=True, zero_division=0)
    per_class = []
    for name in target_names:
        if name in cr:
            per_class.append({
                'label': name,
                'precision': float(cr[name]['precision']),
                'recall': float(cr[name]['recall']),
                'f1': float(cr[name]['f1-score']),
                'support': int(cr[name]['support'])
            })

    return {
        'accuracy': float(acc),
        'precision_macro': float(prec_m),
        'precision_weighted': float(prec_w),
        'recall_macro': float(rec_m),
        'recall_weighted': float(rec_w),
        'f1_macro': float(f1_m),
        'f1_weighted': float(f1_w),
        'confusion_matrix': cm.tolist(),
        'roc_data': roc_data,
        'per_class': per_class,
        'target_names': target_names,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data/info')
def data_info():
    try:
        train_file = os.path.join(PROJECT_ROOT, 'KDDTrain+.txt')
        test_file = os.path.join(PROJECT_ROOT, 'KDDTest+.txt')

        info = {
            'train_exists': os.path.exists(train_file),
            'test_exists': os.path.exists(test_file),
            'train_path': train_file,
            'test_path': test_file,
            'xgb_available': XGB_AVAILABLE,
            'imblearn_available': IMBLEARN_AVAILABLE,
        }

        if _state['df_train'] is not None:
            df = _state['df_train']
            unique, counts = np.unique(_state['y_train'], return_counts=True)
            mapping = _state['label_mapping'] or {}
            cat_dist = {str(mapping.get(int(u), u)): int(c) for u, c in zip(unique, counts)}

            proto_dist = df['protocol_type'].value_counts().to_dict()
            flag_dist = df['flag'].value_counts().to_dict()
            service_top10 = df['service'].value_counts().head(10).to_dict()

            info['preprocessed'] = True
            info['total_samples'] = int(len(df))
            info['feature_count'] = int(_state['X_train'].shape[1])
            info['category_count'] = int(len(mapping))
            info['category_distribution'] = cat_dist
            info['protocol_distribution'] = {str(k): int(v) for k, v in proto_dist.items()}
            info['flag_distribution'] = {str(k): int(v) for k, v in flag_dist.items()}
            info['service_top10'] = {str(k): int(v) for k, v in service_top10.items()}

            if _state['preprocess_info']:
                info['preprocess_info'] = _state['preprocess_info']
        else:
            info['preprocessed'] = False

        return jsonify({'status': 'success', 'data': info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    if _status['preprocessing'] == 'running':
        return jsonify({'status': 'error', 'message': '预处理正在进行中'})

    _status['preprocessing'] = 'running'
    _messages['preprocessing'] = []

    data = request.json or {}
    use_feature_selection = data.get('use_feature_selection', True)
    n_features = int(data.get('n_features', 30))
    use_pca = data.get('use_pca', False)
    n_components = int(data.get('n_components', 20))
    test_size = float(data.get('test_size', 0.7))
    dataset = data.get('dataset', 'train_test')
    classification_mode = data.get('classification_mode', 'fine')
    handle_outliers = data.get('handle_outliers', True)
    balance_method = data.get('balance_method', 'none')

    def run():
        try:
            add_msg('preprocessing', '='*50)
            add_msg('preprocessing', '【步骤1】数据预处理')
            add_msg('preprocessing', '='*50)
            add_msg('preprocessing', f'分类模式: {classification_mode}')
            add_msg('preprocessing', f'异常值处理: {handle_outliers}')
            add_msg('preprocessing', f'特征选择: {use_feature_selection} (K={n_features})')
            add_msg('preprocessing', f'PCA降维: {use_pca}')
            add_msg('preprocessing', f'类不平衡处理: {balance_method}')
            add_msg('preprocessing', f'训练集/测试集比例: {1-test_size:.0%} / {test_size:.0%}')

            if dataset == 'small':
                train_file = os.path.join(PROJECT_ROOT, 'KDDTrain+_20Percent.txt')
            elif dataset == 'train_test':
                train_file = os.path.join(PROJECT_ROOT, 'train_test')
            else:
                train_file = os.path.join(PROJECT_ROOT, 'KDDTrain+.txt')

            if not os.path.exists(train_file):
                add_msg('preprocessing', f'错误: 训练文件不存在: {train_file}')
                _status['preprocessing'] = 'error'
                return

            # 1. 加载数据
            add_msg('preprocessing', f'\n[1.1] 加载数据集: {os.path.basename(train_file)}')
            df = pd.read_csv(train_file, header=None, names=COLUMN_NAMES)
            original_count = len(df)
            add_msg('preprocessing', f'  原始数据: {original_count} 样本, {len(COLUMN_NAMES)-2} 特征')

            # 2. 处理缺失值
            add_msg('preprocessing', f'\n[1.2] 处理缺失值')
            missing_before = df.isnull().sum().sum()
            df = df.fillna(0)
            add_msg('preprocessing', f'  缺失值数量: {missing_before} (已填充为0)')

            # 3. 异常值处理
            if handle_outliers:
                add_msg('preprocessing', f'\n[1.3] 异常值处理 (IQR方法)')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outlier_count = 0
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        outlier_count += outliers
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
                add_msg('preprocessing', f'  处理异常值: {outlier_count} 个 (已裁剪到IQR范围)')

            # 4. 编码分类变量
            add_msg('preprocessing', f'\n[1.4] 编码分类变量')
            categorical_cols = ['protocol_type', 'service', 'flag']
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                add_msg('preprocessing', f'  {col}: {len(le.classes_)} 类别')

            # 5. 标签编码
            if classification_mode == 'fine':
                add_msg('preprocessing', f'\n[1.5] 细粒度分类（原始攻击类型）')
                df['encoded_label'] = df['label'].astype(str)
                le_label = LabelEncoder()
                df['encoded_label'] = le_label.fit_transform(df['encoded_label'])
                label_encoders['label'] = le_label
                label_mapping = dict(zip(le_label.transform(le_label.classes_), le_label.classes_))
            else:
                add_msg('preprocessing', f'\n[1.5] 粗粒度分类（5大类）')
                df['main_category'] = df['label'].map(ATTACK_CATEGORY_MAP).fillna('other')
                le_label = LabelEncoder()
                df['encoded_label'] = le_label.fit_transform(df['main_category'])
                label_encoders['main_category'] = le_label
                label_mapping = dict(zip(le_label.transform(le_label.classes_), le_label.classes_))

            unique, counts = np.unique(df['encoded_label'], return_counts=True)
            add_msg('preprocessing', f'  类别数: {len(unique)}')
            for u, c in zip(unique, counts):
                lbl = label_mapping.get(int(u), str(u))
                add_msg('preprocessing', f'    {lbl}: {c} 条')

            # 6. 特征和标签分离
            feature_cols = [col for col in COLUMN_NAMES if col not in ['label', 'difficulty']]
            X = df[feature_cols]
            y = df['encoded_label']

            # 6.1 过滤样本极少的类别（少于2个样本的类别无法分层划分）
            class_counts = y.value_counts()
            rare_classes = class_counts[class_counts < 2].index.tolist()
            if rare_classes:
                rare_names = [label_mapping.get(int(c), str(c)) for c in rare_classes]
                add_msg('preprocessing', f'\n[1.5.1] 过滤稀有类别（样本数<2）')
                add_msg('preprocessing', f'  移除 {len(rare_classes)} 个稀有类别: {rare_names}')
                rare_mask = ~y.isin(rare_classes)
                removed_count = (~rare_mask).sum()
                df = df[rare_mask].reset_index(drop=True)

            # 6.2 重新编码标签使其连续（XGBoost要求标签必须从0连续编号）
            # 基于原始攻击类型字符串重新拟合LabelEncoder
            if classification_mode == 'fine':
                le_new = LabelEncoder()
                df['encoded_label'] = le_new.fit_transform(df['label'].astype(str))
                label_mapping = {int(i): name for i, name in enumerate(le_new.classes_)}
                label_encoders['label'] = le_new
            else:
                le_new = LabelEncoder()
                df['encoded_label'] = le_new.fit_transform(df['main_category'].astype(str))
                label_mapping = {int(i): name for i, name in enumerate(le_new.classes_)}
                label_encoders['main_category'] = le_new
            if rare_classes:
                add_msg('preprocessing', f'  重新编码标签为连续值: 0-{len(label_mapping)-1}')
                add_msg('preprocessing', f'  移除 {removed_count} 条样本，剩余 {len(df)} 条')

            X = df[feature_cols]
            y = df['encoded_label']

            # 7. 标准化
            add_msg('preprocessing', f'\n[1.6] 数据标准化 (StandardScaler)')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            add_msg('preprocessing', f'  标准化完成: 均值=0, 方差=1')

            # 7.1 过滤零方差特征（避免SelectKBest产生NaN）
            from sklearn.feature_selection import VarianceThreshold
            var_selector = VarianceThreshold(threshold=0.0)
            X_scaled_arr = var_selector.fit_transform(X_scaled)
            zero_var_mask = var_selector.get_support()
            zero_var_features = [col for col, kept in zip(X.columns, zero_var_mask) if not kept]
            if zero_var_features:
                add_msg('preprocessing', f'  移除 {len(zero_var_features)} 个零方差特征: {zero_var_features}')
            X = X.loc[:, zero_var_mask]
            X_scaled = X_scaled_arr

            # 8. 特征选择
            if use_feature_selection:
                add_msg('preprocessing', f'\n[1.7] 特征选择 (SelectKBest, K={n_features})')
                actual_k = min(n_features, X_scaled.shape[1])
                if actual_k < n_features:
                    add_msg('preprocessing', f'  特征数调整为: {actual_k} (原特征数不足)')
                feature_selector = SelectKBest(f_classif, k=actual_k)
                X_scaled = feature_selector.fit_transform(pd.DataFrame(X_scaled, columns=X.columns), y)
                selected_features = X.columns[feature_selector.get_support()].tolist()
                feature_scores = feature_selector.scores_
                feature_pvalues = feature_selector.pvalues_
                selected_mask = feature_selector.get_support()
                # 处理可能的NaN
                feature_scores = np.nan_to_num(feature_scores, nan=0.0)

                feature_importance = []
                for i, col in enumerate(X.columns):
                    if selected_mask[i]:
                        feature_importance.append({
                            'feature': col,
                            'score': float(feature_scores[i]),
                            'pvalue': float(feature_pvalues[i])
                        })
                feature_importance.sort(key=lambda x: x['score'], reverse=True)
                add_msg('preprocessing', f'  选中 {n_features} 个最佳特征')
                add_msg('preprocessing', f'  Top 10 重要特征:')
                for fi in feature_importance[:10]:
                    add_msg('preprocessing', f'    {fi["feature"]}: F-score={fi["score"]:.2f}')
            else:
                feature_selector = None
                selected_features = X.columns.tolist()
                feature_importance = []

            # 9. PCA降维
            if use_pca:
                add_msg('preprocessing', f'\n[1.8] PCA降维 (n_components={n_components})')
                pca = PCA(n_components=n_components)
                X_scaled = pca.fit_transform(X_scaled)
                explained_variance = pca.explained_variance_ratio_.tolist()
                add_msg('preprocessing', f'  累计解释方差: {sum(explained_variance)*100:.2f}%')
            else:
                pca = None
                explained_variance = []

            # 10. 数据划分
            add_msg('preprocessing', f'\n[1.9] 数据划分 (验证集比例={test_size})')
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )
                add_msg('preprocessing', f'  使用分层抽样')
            except ValueError:
                add_msg('preprocessing', f'  分层抽样失败，改用普通划分')
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42
                )
            add_msg('preprocessing', f'  训练集: {len(X_train)} 样本')
            add_msg('preprocessing', f'  验证集: {len(X_val)} 样本')

            # 11. 类不平衡处理
            balance_info = {'method': balance_method, 'original_dist': {}}
            classes_orig = np.unique(y_train)
            for c in classes_orig:
                balance_info['original_dist'][label_mapping.get(int(c), str(c))] = int((y_train == c).sum())

            if balance_method != 'none' and IMBLEARN_AVAILABLE:
                add_msg('preprocessing', f'\n[1.10] 类不平衡处理: {balance_method}')
                if balance_method == 'smote':
                    min_count = int(np.bincount(y_train).min())
                    if min_count <= 2:
                        add_msg('preprocessing', f'  最小样本数={min_count}，SMOTE无法工作，改用随机过采样')
                        sampler = RandomOverSampler(random_state=42)
                    else:
                        k = min(5, min_count - 1)
                        sampler = SMOTE(random_state=42, k_neighbors=k)
                        add_msg('preprocessing', f'  SMOTE k_neighbors={k}')
                elif balance_method == 'oversample':
                    sampler = RandomOverSampler(random_state=42)
                elif balance_method == 'undersample':
                    sampler = RandomUnderSampler(random_state=42)
                else:
                    sampler = None

                if sampler is not None:
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    add_msg('preprocessing', f'  平衡后训练集: {len(X_train)} 样本')
                    balance_info['balanced_dist'] = {}
                    for c in np.unique(y_train):
                        balance_info['balanced_dist'][label_mapping.get(int(c), str(c))] = int((y_train == c).sum())
            else:
                add_msg('preprocessing', f'\n[1.10] 类不平衡处理: 跳过')

            # 12. 计算类别权重
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights_dict = dict(zip(classes, class_weights))

            add_msg('preprocessing', '\n' + '='*50)
            add_msg('preprocessing', '预处理完成！')
            add_msg('preprocessing', '='*50)

            preprocess_info = {
                'original_count': original_count,
                'missing_values': int(missing_before),
                'outlier_count': int(outlier_count) if handle_outliers else 0,
                'feature_count_before': len(feature_cols),
                'feature_count_after': X_scaled.shape[1],
                'selected_features': selected_features,
                'feature_importance': feature_importance[:15],
                'explained_variance': explained_variance,
                'balance_info': balance_info,
                'class_count': len(unique),
                'train_count': len(X_train),
                'val_count': len(X_val),
            }

            _state['label_encoders'] = label_encoders
            _state['scaler'] = scaler
            _state['feature_selector'] = feature_selector
            _state['var_selector'] = var_selector
            _state['pca'] = pca
            _state['selected_features'] = selected_features
            _state['X_train'] = X_train
            _state['y_train'] = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
            _state['X_val'] = X_val
            _state['y_val'] = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
            _state['label_mapping'] = label_mapping
            _state['df_train'] = df
            _state['class_weights'] = class_weights_dict
            _state['preprocess_info'] = preprocess_info

            _status['preprocessing'] = 'completed'

        except Exception as e:
            import traceback
            add_msg('preprocessing', f'错误: {str(e)}')
            add_msg('preprocessing', traceback.format_exc())
            _status['preprocessing'] = 'error'

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})


@app.route('/api/preprocess/status')
def preprocess_status():
    last_idx = request.args.get('last_idx', 0, type=int)
    new_messages = _messages['preprocessing'][last_idx:]
    return jsonify({
        'status': _status['preprocessing'],
        'messages': new_messages,
        'last_idx': len(_messages['preprocessing'])
    })


def build_dnn_model(input_shape, num_classes, model_type='improved', dropout_rate=0.4, learning_rate=0.001):
    """构建DNN模型 - 优化版：Focal Loss + Cosine Decay + 更深网络"""
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import LeakyReLU, Add

    if model_type == 'improved':
        # 残差连接的深层网络
        inputs = Input(shape=input_shape)
        x = Dense(512, kernel_regularizer=l2(1e-5))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(dropout_rate)(x)

        # Block 1
        shortcut = Dense(256)(x)
        x = Dense(256, kernel_regularizer=l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(dropout_rate * 0.8)(x)
        x = Dense(256, kernel_regularizer=l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.1)(x)

        # Block 2
        shortcut = Dense(128)(x)
        x = Dense(128, kernel_regularizer=l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(dropout_rate * 0.6)(x)
        x = Dense(128, kernel_regularizer=l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(64, kernel_regularizer=l2(1e-5))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(dropout_rate * 0.5)(x)

        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
    else:
        model = Sequential([
            Dense(512, input_shape=input_shape, kernel_regularizer=l2(1e-5)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(dropout_rate),
            Dense(256, kernel_regularizer=l2(1e-5)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(dropout_rate),
            Dense(128, kernel_regularizer=l2(1e-5)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(dropout_rate),
            Dense(64, kernel_regularizer=l2(1e-5)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(dropout_rate),
            Dense(32, kernel_regularizer=l2(1e-5)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dense(num_classes, activation='softmax')
        ])

    # 使用 Cosine Decay 学习率调度
    from tensorflow.keras.optimizers.schedules import CosineDecay
    lr_schedule = CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=3000
    )

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@app.route('/api/train', methods=['POST'])
def train():
    if _state['scaler'] is None:
        return jsonify({'status': 'error', 'message': '请先进行数据预处理'})
    if _status['training'] == 'running':
        return jsonify({'status': 'error', 'message': '训练正在进行中'})

    _status['training'] = 'running'
    _messages['training'] = []

    data = request.json or {}
    model_type = data.get('model_type', 'improved_dnn')
    epochs = int(data.get('epochs', 100))
    batch_size = int(data.get('batch_size', 128))
    dropout_rate = float(data.get('dropout_rate', 0.3))
    learning_rate = float(data.get('learning_rate', 0.001))
    use_class_weights = data.get('use_class_weights', True)

    def run():
        try:
            start_time = time.time()
            num_classes = len(_state['label_mapping'])
            input_shape = (_state['X_train'].shape[1],)

            add_msg('training', '='*50)
            add_msg('training', f'【步骤2】模型训练 - {model_type}')
            add_msg('training', '='*50)
            add_msg('training', f'模型类型: {model_type}')
            add_msg('training', f'输入维度: {input_shape[0]}')
            add_msg('training', f'输出类别: {num_classes}')
            add_msg('training', f'参数: epochs={epochs}, batch_size={batch_size}, dropout={dropout_rate}, lr={learning_rate}')

            if model_type in ['improved_dnn', 'dnn']:
                dnn_type = 'improved' if model_type == 'improved_dnn' else 'basic'
                model = build_dnn_model(input_shape, num_classes, dnn_type, dropout_rate, learning_rate)
                add_msg('training', f'\n模型结构:')
                model.summary(print_fn=lambda x: add_msg('training', x))

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
                ]

                class_w = _state['class_weights'] if use_class_weights else None

                add_msg('training', f'\n开始训练...')
                history = model.fit(
                    _state['X_train'], _state['y_train'],
                    validation_data=(_state['X_val'], _state['y_val']),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_w,
                    callbacks=callbacks,
                    verbose=1
                )

                y_pred_proba = model.predict(_state['X_val'], verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)

            elif model_type == 'random_forest':
                n_estimators = int(data.get('n_estimators', 100))
                max_depth = int(data.get('max_depth', 20))
                add_msg('training', f'参数: n_estimators={n_estimators}, max_depth={max_depth}')

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced' if use_class_weights else None
                )
                add_msg('training', f'\n开始训练随机森林...')
                model.fit(_state['X_train'], _state['y_train'])

                y_pred = model.predict(_state['X_val'])
                y_pred_proba = model.predict_proba(_state['X_val'])
                history = None

            elif model_type == 'xgboost':
                if not XGB_AVAILABLE:
                    add_msg('training', '错误: XGBoost未安装')
                    _status['training'] = 'error'
                    return
                n_estimators = int(data.get('n_estimators', 100))
                max_depth = int(data.get('max_depth', 6))
                learning_rate_xgb = float(data.get('learning_rate', 0.3))
                add_msg('training', f'参数: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate_xgb}')

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate_xgb,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                add_msg('training', f'\n开始训练XGBoost...')
                model.fit(_state['X_train'], _state['y_train'])

                y_pred = model.predict(_state['X_val'])
                y_pred_proba = model.predict_proba(_state['X_val'])
                history = None
            else:
                add_msg('training', f'错误: 未知模型类型 {model_type}')
                _status['training'] = 'error'
                return

            train_time = time.time() - start_time
            add_msg('training', f'\n训练完成！耗时: {train_time:.2f} 秒')
            add_msg('training', f'\n开始评估验证集...')

            results = compute_full_metrics(_state['y_val'], y_pred, y_pred_proba, _state['label_mapping'])
            results['train_time'] = float(train_time)

            if history:
                results['history'] = {
                    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
                    'loss': [float(x) for x in history.history.get('loss', [])],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                }

            add_msg('training', '\n' + '='*50)
            add_msg('training', '验证集性能指标')
            add_msg('training', '='*50)
            add_msg('training', f"准确率: {results['accuracy']:.4f}")
            add_msg('training', f"精确率(Macro): {results['precision_macro']:.4f}")
            add_msg('training', f"召回率(Macro): {results['recall_macro']:.4f}")
            add_msg('training', f"F1分数(Macro): {results['f1_macro']:.4f}")
            add_msg('training', f"F1分数(Weighted): {results['f1_weighted']:.4f}")

            _state['model'] = model
            _state['model_type'] = model_type
            _state['history'] = history
            _state['val_results'] = results

            add_msg('training', '\n模型训练和评估完成！')
            _status['training'] = 'completed'

        except Exception as e:
            import traceback
            add_msg('training', f'错误: {str(e)}')
            add_msg('training', traceback.format_exc())
            _status['training'] = 'error'

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})


@app.route('/api/train/status')
def train_status():
    last_idx = request.args.get('last_idx', 0, type=int)
    new_messages = _messages['training'][last_idx:]
    return jsonify({
        'status': _status['training'],
        'messages': new_messages,
        'last_idx': len(_messages['training'])
    })


@app.route('/api/test', methods=['POST'])
def test():
    if _state['model'] is None:
        return jsonify({'status': 'error', 'message': '请先训练模型'})
    if _status['testing'] == 'running':
        return jsonify({'status': 'error', 'message': '测试正在进行中'})

    _status['testing'] = 'running'
    _messages['testing'] = []

    def run():
        try:
            add_msg('testing', '='*50)
            add_msg('testing', '【步骤4】测试集评估')
            add_msg('testing', '='*50)
            add_msg('testing', '使用 train_test 文件中划分的测试集')

            X_test_final = _state['X_val']
            y_test = _state['y_val']
            add_msg('testing', f'测试集样本数: {len(y_test)} 条')

            model = _state['model']
            model_type = _state['model_type']

            if model_type in ['improved_dnn', 'dnn']:
                y_pred_proba = model.predict(X_test_final, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test_final)
                y_pred_proba = model.predict_proba(X_test_final)

            results = compute_full_metrics(y_test, y_pred, y_pred_proba, _state['label_mapping'])
            results['test_count'] = int(len(y_test))

            add_msg('testing', '\n' + '='*50)
            add_msg('testing', 'NSL-KDD 测试集性能指标')
            add_msg('testing', '='*50)
            add_msg('testing', f"准确率: {results['accuracy']:.4f}")
            add_msg('testing', f"精确率(Macro): {results['precision_macro']:.4f}")
            add_msg('testing', f"精确率(Weighted): {results['precision_weighted']:.4f}")
            add_msg('testing', f"召回率(Macro): {results['recall_macro']:.4f}")
            add_msg('testing', f"召回率(Weighted): {results['recall_weighted']:.4f}")
            add_msg('testing', f"F1分数(Macro): {results['f1_macro']:.4f}")
            add_msg('testing', f"F1分数(Weighted): {results['f1_weighted']:.4f}")

            _state['test_results'] = results
            add_msg('testing', '\n测试集评估完成！')
            _status['testing'] = 'completed'

        except Exception as e:
            import traceback
            add_msg('testing', f'错误: {str(e)}')
            add_msg('testing', traceback.format_exc())
            _status['testing'] = 'error'

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})


@app.route('/api/test/status')
def test_status():
    last_idx = request.args.get('last_idx', 0, type=int)
    new_messages = _messages['testing'][last_idx:]
    return jsonify({
        'status': _status['testing'],
        'messages': new_messages,
        'last_idx': len(_messages['testing'])
    })


@app.route('/api/results/validation')
def val_results():
    if _state['val_results'] is None:
        return jsonify({'status': 'error', 'message': '请先训练模型'})
    return jsonify({'status': 'success', 'data': _state['val_results'], 'model_type': _state['model_type']})


@app.route('/api/results/test')
def test_results_api():
    if _state['test_results'] is None:
        return jsonify({'status': 'error', 'message': '请先运行测试集评估'})
    return jsonify({'status': 'success', 'data': _state['test_results'], 'model_type': _state['model_type']})


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """多算法对比（加分项）"""
    if _state['scaler'] is None:
        return jsonify({'status': 'error', 'message': '请先进行数据预处理'})
    if _status['comparison'] == 'running':
        return jsonify({'status': 'error', 'message': '对比正在进行中'})

    _status['comparison'] = 'running'
    _messages['comparison'] = []

    data = request.json or {}
    models_to_compare = data.get('models', ['improved_dnn', 'random_forest', 'xgboost'])
    epochs = int(data.get('epochs', 20))

    def run():
        try:
            add_msg('comparison', '='*50)
            add_msg('comparison', '【加分项】多算法性能对比')
            add_msg('comparison', '='*50)
            add_msg('comparison', f'对比算法: {models_to_compare}')

            num_classes = len(_state['label_mapping'])
            input_shape = (_state['X_train'].shape[1],)
            X_train, y_train = _state['X_train'], _state['y_train']
            X_val, y_val = _state['X_val'], _state['y_val']
            results_list = []

            for model_name in models_to_compare:
                add_msg('comparison', f'\n--- 训练 {model_name} ---')
                start_time = time.time()

                try:
                    if model_name in ['improved_dnn', 'dnn']:
                        dnn_type = 'improved' if model_name == 'improved_dnn' else 'basic'
                        model = build_dnn_model(input_shape, num_classes, dnn_type, 0.3, 0.001)
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=256,
                            class_weight=_state['class_weights'],
                            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)],
                            verbose=0
                        )
                        y_pred_proba = model.predict(X_val, verbose=0)
                        y_pred = np.argmax(y_pred_proba, axis=1)

                    elif model_name == 'random_forest':
                        model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced')
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        y_pred_proba = model.predict_proba(X_val)

                    elif model_name == 'xgboost':
                        if not XGB_AVAILABLE:
                            add_msg('comparison', f'  XGBoost不可用，跳过')
                            continue
                        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        y_pred_proba = model.predict_proba(X_val)

                    else:
                        continue

                    train_time = time.time() - start_time
                    metrics = compute_full_metrics(y_val, y_pred, y_pred_proba, _state['label_mapping'])
                    metrics['model_name'] = model_name
                    metrics['train_time'] = float(train_time)
                    results_list.append(metrics)

                    add_msg('comparison', f'  准确率: {metrics["accuracy"]:.4f}')
                    add_msg('comparison', f'  F1(Macro): {metrics["f1_macro"]:.4f}')
                    add_msg('comparison', f'  F1(Weighted): {metrics["f1_weighted"]:.4f}')
                    add_msg('comparison', f'  耗时: {train_time:.2f}s')

                    _state['comparison_results'][model_name] = metrics

                except Exception as e:
                    add_msg('comparison', f'  错误: {str(e)}')

            # 生成对比报告
            if results_list:
                add_msg('comparison', '\n' + '='*50)
                add_msg('comparison', '模型对比总结')
                add_msg('comparison', '='*50)

                best_acc = max(results_list, key=lambda x: x['accuracy'])
                best_f1 = max(results_list, key=lambda x: x['f1_macro'])
                fastest = min(results_list, key=lambda x: x['train_time'])

                add_msg('comparison', f'准确率最高: {best_acc["model_name"]} ({best_acc["accuracy"]:.4f})')
                add_msg('comparison', f'F1(Macro)最高: {best_f1["model_name"]} ({best_f1["f1_macro"]:.4f})')
                add_msg('comparison', f'训练最快: {fastest["model_name"]} ({fastest["train_time"]:.2f}s)')

            _status['comparison'] = 'completed'

        except Exception as e:
            import traceback
            add_msg('comparison', f'错误: {str(e)}')
            add_msg('comparison', traceback.format_exc())
            _status['comparison'] = 'error'

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})


@app.route('/api/compare/status')
def compare_status():
    last_idx = request.args.get('last_idx', 0, type=int)
    new_messages = _messages['comparison'][last_idx:]
    return jsonify({
        'status': _status['comparison'],
        'messages': new_messages,
        'last_idx': len(_messages['comparison']),
        'results': _state['comparison_results']
    })


@app.route('/api/tune', methods=['POST'])
def tune_hyperparameters():
    """超参数调优（模型优化）"""
    if _state['scaler'] is None:
        return jsonify({'status': 'error', 'message': '请先进行数据预处理'})
    if _status['tuning'] == 'running':
        return jsonify({'status': 'error', 'message': '调优正在进行中'})

    _status['tuning'] = 'running'
    _messages['tuning'] = []

    data = request.json or {}
    model_type = data.get('model_type', 'random_forest')

    def run():
        try:
            add_msg('tuning', '='*50)
            add_msg('tuning', '【步骤3】模型优化 - 超参数调优')
            add_msg('tuning', '='*50)
            add_msg('tuning', f'模型: {model_type}')

            X_train, y_train = _state['X_train'], _state['y_train']

            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10]
                }
                model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
                add_msg('tuning', f'参数网格: {param_grid}')
                add_msg('tuning', f'总组合数: {3*4*3}=36')

            elif model_type == 'xgboost':
                if not XGB_AVAILABLE:
                    add_msg('tuning', 'XGBoost不可用')
                    _status['tuning'] = 'error'
                    return
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
                model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
                add_msg('tuning', f'参数网格: {param_grid}')
                add_msg('tuning', f'总组合数: {3*3*3}=27')

            else:
                add_msg('tuning', 'DNN调优请使用训练界面的参数调整功能')
                _status['tuning'] = 'error'
                return

            add_msg('tuning', '\n开始网格搜索（使用3折交叉验证）...')
            start_time = time.time()

            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='f1_weighted',
                n_jobs=-1, verbose=1, return_train_score=False
            )
            grid_search.fit(X_train, y_train)

            tune_time = time.time() - start_time
            add_msg('tuning', f'\n调优完成！耗时: {tune_time:.2f}s')
            add_msg('tuning', f'\n最佳参数: {grid_search.best_params_}')
            add_msg('tuning', f'最佳F1(Weighted)分数: {grid_search.best_score_:.4f}')

            cv_results = grid_search.cv_results_
            top_n = min(10, len(cv_results['mean_test_score']))
            top_indices = np.argsort(-cv_results['mean_test_score'])[:top_n]

            add_msg('tuning', f'\nTop {top_n} 参数组合:')
            for rank, idx in enumerate(top_indices):
                params = cv_results['params'][idx]
                score = cv_results['mean_test_score'][idx]
                std = cv_results['std_test_score'][idx]
                add_msg('tuning', f'  #{rank+1}: F1={score:.4f}±{std:.4f} - {params}')

            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'tune_time': float(tune_time),
                'top_results': [{
                    'rank': rank + 1,
                    'params': cv_results['params'][idx],
                    'score': float(cv_results['mean_test_score'][idx]),
                    'std': float(cv_results['std_test_score'][idx])
                } for rank, idx in enumerate(top_indices)]
            }
            _state['tuning_results'] = tuning_results

            _status['tuning'] = 'completed'

        except Exception as e:
            import traceback
            add_msg('tuning', f'错误: {str(e)}')
            add_msg('tuning', traceback.format_exc())
            _status['tuning'] = 'error'

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})


@app.route('/api/tune/status')
def tune_status():
    last_idx = request.args.get('last_idx', 0, type=int)
    new_messages = _messages['tuning'][last_idx:]
    return jsonify({
        'status': _status['tuning'],
        'messages': new_messages,
        'last_idx': len(_messages['tuning']),
        'results': _state['tuning_results']
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if _state['model'] is None:
        return jsonify({'status': 'error', 'message': '请先训练模型'})

    try:
        data = request.json or {}
        features = data.get('features')

        if features is None:
            return jsonify({'status': 'error', 'message': '缺少features参数'})

        X = np.array(features, dtype=float).reshape(1, -1)
        model = _state['model']
        model_type = _state['model_type']

        # 预处理流程与训练保持一致
        X_scaled = _state['scaler'].transform(X)
        if _state.get('var_selector') is not None:
            X_scaled = _state['var_selector'].transform(X_scaled)
        if _state['feature_selector'] is not None:
            X_scaled = _state['feature_selector'].transform(X_scaled)
        if _state['pca'] is not None:
            X_scaled = _state['pca'].transform(X_scaled)
        X_final = X_scaled

        if model_type in ['improved_dnn', 'dnn']:
            y_pred_proba = model.predict(X_final, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_final)
            y_pred_proba = model.predict_proba(X_final)

        mapping = _state['label_mapping']
        pred_label = mapping.get(int(y_pred[0]), str(y_pred[0]))
        probabilities = {mapping.get(i, str(i)): float(p) for i, p in enumerate(y_pred_proba[0])}

        return jsonify({
            'status': 'success',
            'prediction': pred_label,
            'prediction_class': int(y_pred[0]),
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/reset', methods=['POST'])
def reset():
    global _state, _status, _messages
    _state = {
        'label_encoders': {}, 'scaler': None, 'feature_selector': None, 'var_selector': None, 'pca': None,
        'selected_features': None, 'X_train': None, 'y_train': None, 'X_val': None,
        'y_val': None, 'X_test': None, 'y_test': None, 'label_mapping': None,
        'model': None, 'model_type': None, 'history': None, 'val_results': None,
        'test_results': None, 'df_train': None, 'class_weights': None,
        'preprocess_info': None, 'comparison_results': {}, 'tuning_results': None,
    }
    _status = {'preprocessing': 'idle', 'training': 'idle', 'testing': 'idle',
               'comparison': 'idle', 'tuning': 'idle'}
    _messages = {'preprocessing': [], 'training': [], 'testing': [], 'comparison': [], 'tuning': []}
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
