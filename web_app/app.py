import os
import sys
import json
import time
import base64
import threading
import warnings
import pandas as pd
import numpy as np
from io import BytesIO

import pickle
import joblib
from flask import Flask, render_template, request, jsonify, send_file

# PyTorch 必须在主线程导入，否则 macOS 上会 segfault（exit code 245）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 避免 OpenMP 线程冲突
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'network_security_key'


# 预定义 DNN 模型类（模块级别，避免在线程内定义导致闭包引用问题）
class DNNMultiClass(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128, 64), dropout_rate=0.3):
        super(DNNMultiClass, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 预定义 1D-CNN 模型类（模块级别）
class CNN1DMultiClass(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels=(64, 128, 256),
                 kernel_size=3, dropout_rate=0.3):
        super(CNN1DMultiClass, self).__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


# 预定义 Transformer 模型类（模块级别）
class TransformerMultiClass(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout_rate=0.1):
        super(TransformerMultiClass, self).__init__()
        self.input_dim = input_dim
        self.proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(-1)               # (batch, input_dim, 1)
        x = self.proj(x)                  # (batch, input_dim, d_model)
        x = x + self.pos_embedding         # 加入位置编码
        x = self.encoder(x)               # (batch, input_dim, d_model)
        x = x.mean(dim=1)                 # 平均池化 -> (batch, d_model)
        return self.classifier(x)


# 模型基础名与分类粒度：状态键按 {base}_{granularity} 命名空间隔离，5 类/23 类并存
_BASE_MODELS = ['xgboost', 'dnn', 'cnn1d', 'transformer']
_GRANULARITIES = ['5', '23']
_STATUS_KEYS = ([f'{b}_{g}' for b in _BASE_MODELS for g in _GRANULARITIES] +
                ['preprocessing'] +
                [f'test_{b}_{g}' for b in _BASE_MODELS for g in _GRANULARITIES])

training_status = {k: 'idle' for k in _STATUS_KEYS}
training_messages = {k: [] for k in _STATUS_KEYS}

# 进度追踪（0-100 整数），前端进度条据此更新
training_progress = {k: 0 for k in _STATUS_KEYS}

def add_message(task, msg):
    if task in training_messages:
        training_messages[task].append(msg)
        if len(training_messages[task]) > 100:
            training_messages[task] = training_messages[task][-50:]

def update_progress(task, value):
    if task in training_progress:
        training_progress[task] = max(0, min(100, int(value)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    if training_status['preprocessing'] == 'running':
        return jsonify({'status': 'error', 'message': '预处理正在进行中'})
    
    dataset = request.json.get('dataset', 'full')
    training_status['preprocessing'] = 'running'
    training_messages['preprocessing'] = []
    training_progress['preprocessing'] = 0

    def run_preprocessing():
        try:
            data_file = 'KDDTrain+.txt' if dataset == 'full' else 'KDDTrain+_20Percent.txt'
            add_message('preprocessing', f'开始数据预处理（防数据泄漏版本：先切分，后拟合）...')
            add_message('preprocessing', f'使用数据集: {data_file}')
            update_progress('preprocessing', 5)

            import warnings
            warnings.filterwarnings('ignore')

            from data_preprocessing.data_preprocessing import (
                load_data, explore_data, preprocess_labels,
                split_data, encode_and_scale,
                COLUMN_NAMES, ATTACK_CATEGORIES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
            )

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'Train', data_file)

            if not os.path.exists(data_path):
                add_message('preprocessing', f'错误: 数据文件不存在: {data_path}')
                training_status['preprocessing'] = 'error'
                return

            add_message('preprocessing', f'加载数据: {data_path}')
            df = load_data(data_path)
            update_progress('preprocessing', 15)

            add_message('preprocessing', '数据探索中...')
            df = explore_data(df)
            update_progress('preprocessing', 25)

            add_message('preprocessing', '标签预处理中（生成 二分类 / 5大分类 / 23细分类 三种标签）...')
            df, le_category, le_multiclass = preprocess_labels(df)
            add_message('preprocessing', f'23 细分类共 {len(le_multiclass.classes_)} 个类别')
            update_progress('preprocessing', 40)

            add_message('preprocessing', '数据集划分（训练70%/验证10%/测试20%，以 23 分类做分层采样）...')
            df_train, df_val, df_test = split_data(df, test_size=0.2, val_size=0.1)
            update_progress('preprocessing', 55)

            add_message('preprocessing', '特征编码与标准化（仅训练集fit，验证/测试集只transform）...')
            df_train_p, df_val_p, df_test_p, ohe, scaler, ohe_names = encode_and_scale(df_train, df_val, df_test)
            update_progress('preprocessing', 75)

            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train')

            df_train_p.to_csv(os.path.join(output_dir, 'KDDTrain_preprocessed_train.csv'), index=False)
            df_val_p.to_csv(os.path.join(output_dir, 'KDDTrain_preprocessed_val.csv'), index=False)
            df_test_p.to_csv(os.path.join(output_dir, 'KDDTrain_preprocessed_test.csv'), index=False)

            import joblib
            joblib.dump(ohe, os.path.join(output_dir, 'encoder_onehot.pkl'))
            joblib.dump(scaler, os.path.join(output_dir, 'scaler_standard.pkl'))
            # 保存 23 分类标签编码器
            joblib.dump(le_multiclass, os.path.join(output_dir, 'encoder_multiclass_23.pkl'))
            # 同时保存纯文本类别列表（跨 sklearn 版本兼容）
            class_list_path = os.path.join(output_dir, 'encoder_multiclass_23_classes.txt')
            with open(class_list_path, 'w') as f:
                for c in le_multiclass.classes_:
                    f.write(c + '\n')
            # 保存 5 大类标签编码器（供 5 分类任务推理映射）
            joblib.dump(le_category, os.path.join(output_dir, 'encoder_category_5.pkl'))
            class_list_path_5 = os.path.join(output_dir, 'encoder_category_5_classes.txt')
            with open(class_list_path_5, 'w') as f:
                for c in le_category.classes_:
                    f.write(c + '\n')
            update_progress('preprocessing', 95)

            add_message('preprocessing', f'预处理完成！')
            add_message('preprocessing', f'训练集: {df_train_p.shape}')
            add_message('preprocessing', f'验证集: {df_val_p.shape}')
            add_message('preprocessing', f'测试集: {df_test_p.shape}')
            add_message('preprocessing', f'StandardScaler/OneHotEncoder 仅在训练集上 fit')
            add_message('preprocessing', f'验证集用于训练监控，测试集仅用于最终评估')
            add_message('preprocessing', f'已生成 23 细分类标签: normal + 22 种具体攻击类型')

            training_status['preprocessing'] = 'completed'
            update_progress('preprocessing', 100)

        except Exception as e:
            add_message('preprocessing', f'错误: {str(e)}')
            training_status['preprocessing'] = 'error'
    
    thread = threading.Thread(target=run_preprocessing)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/preprocess/status')
def get_preprocess_status():
    return jsonify({
        'status': training_status['preprocessing'],
        'messages': training_messages['preprocessing'],
        'progress': training_progress.get('preprocessing', 0)
    })

@app.route('/data/distribution')
def get_data_distribution():
    try:
        dataset = request.args.get('dataset', '20percent')
        data_file = 'KDDTrain+.txt' if dataset == 'full' else 'KDDTrain+_20Percent.txt'
        
        raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', data_file)
        processed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain_preprocessed_train.csv')
        
        if not os.path.exists(raw_data_path):
            return jsonify({'status': 'error', 'message': f'原始数据文件不存在: {data_file}'})
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_preprocessing.data_preprocessing import COLUMN_NAMES, ATTACK_CATEGORIES
        
        df = pd.read_csv(raw_data_path, names=COLUMN_NAMES)
        
        label_col = 'label'
        label_dist_raw = df[label_col].value_counts()
        normal_count = int(label_dist_raw.get('normal', 0))
        attack_count = len(df) - normal_count
        label_dist = {'normal': normal_count, 'attack': attack_count}
        
        cat_dist = {}
        for label, count in label_dist_raw.items():
            if label == 'normal':
                cat_name = 'normal'
            else:
                cat_name = ATTACK_CATEGORIES.get(label, 'other')
            cat_dist[cat_name] = cat_dist.get(cat_name, 0) + int(count)
        
        protocol_dist = df['protocol_type'].value_counts().to_dict()
        protocol_dist = {k: int(v) for k, v in protocol_dist.items()}
        
        flag_dist = df['flag'].value_counts().to_dict()
        flag_dist = {k: int(v) for k, v in flag_dist.items()}
        
        service_top10 = df['service'].value_counts().head(10).to_dict()
        service_top10 = {k: int(v) for k, v in service_top10.items()}
        
        feature_count = 0
        if os.path.exists(processed_path):
            df_p = pd.read_csv(processed_path)
            exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
            feature_count = len([c for c in df_p.columns if c not in exclude_cols])
        else:
            feature_count = 41
        
        stats = {
            'total_samples': len(df),
            'feature_count': feature_count,
            'normal_count': normal_count,
            'attack_count': attack_count,
            'normal_ratio': round(normal_count / len(df), 4),
            'attack_ratio': round(attack_count / len(df), 4)
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats,
            'label_distribution': label_dist,
            'category_distribution': cat_dist,
            'protocol_distribution': protocol_dist,
            'flag_distribution': flag_dist,
            'service_top10': service_top10
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train/model', methods=['POST'])
def train_model():
    model_name = request.json.get('model_name')
    granularity = request.json.get('granularity', '23')
    if granularity not in ('5', '23'):
        return jsonify({'status': 'error', 'message': '无效的分类粒度，仅支持 5 或 23'})
    task_key = f'{model_name}_{granularity}'

    if model_name not in ['xgboost', 'dnn', 'cnn1d', 'transformer']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})

    if training_status[task_key] == 'running':
        return jsonify({'status': 'error', 'message': f'{model_name}({granularity}类) 正在训练中'})

    training_status[task_key] = 'running'
    training_messages[task_key] = []
    training_progress[task_key] = 0

    def run_training():
        try:
            add_message(task_key,f'开始训练 {model_name} 模型（防数据泄漏版本）...')
            update_progress(task_key, 3)

            import warnings
            warnings.filterwarnings('ignore')
            import numpy as np
            np.seterr(all='ignore')

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
            val_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_val.csv')
            test_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_test.csv')
            
            for p in [train_path, val_path, test_path]:
                if not os.path.exists(p):
                    add_message(task_key,f'错误: 数据文件不存在: {p}')
                    add_message(task_key,'请先运行数据预处理')
                    training_status[task_key] ='error'
                    return
            
            # 加载训练集、验证集、测试集（预处理时已正确划分）
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)

            exclude_cols = ['label', 'difficulty', 'label_binary',
                            'label_category', 'label_category_encoded',
                            'label_multiclass', 'label_multiclass_encoded']
            feature_cols = [col for col in df_train.columns if col not in exclude_cols]

            # 按分类粒度选标签列：5 类用 label_category_encoded，23 类用 label_multiclass_encoded
            label_col = 'label_category_encoded' if granularity == '5' else 'label_multiclass_encoded'
            X_train = df_train[feature_cols].values.astype(np.float32)
            y_train = df_train[label_col].values.astype(np.int64)
            X_val = df_val[feature_cols].values.astype(np.float32)
            y_val = df_val[label_col].values.astype(np.int64)
            X_test = df_test[feature_cols].values.astype(np.float32)
            y_test = df_test[label_col].values.astype(np.int64)

            # 加载类别列表获取真正的类别总数（跨 sklearn 版本兼容）
            if granularity == '5':
                encoder_class_path = os.path.join(base_dir, 'Train', 'encoder_category_5_classes.txt')
            else:
                encoder_class_path = os.path.join(base_dir, 'Train', 'encoder_multiclass_23_classes.txt')
            if os.path.exists(encoder_class_path):
                with open(encoder_class_path, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                num_classes = len(class_names)
                normal_idx = class_names.index('normal') if 'normal' in class_names else 0
            else:
                # 5 类编码器不存在时，从预处理 CSV 的 label_category 列动态构建（兼容旧产物）
                if granularity == '5':
                    class_names = sorted(df_train['label_category'].dropna().unique().tolist())
                    num_classes = len(class_names)
                    normal_idx = class_names.index('normal') if 'normal' in class_names else 0
                    add_message(task_key, f'未找到 5 类编码器，从训练集动态构建类别: {class_names}')
                else:
                    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
                    normal_idx = 0

            # XGBoost 必须用数据中实际出现的不重复类别数（标签必须0..N-1 连续无空隙）
            xgb_num_class = len(np.unique(y_train))
            if xgb_num_class != num_classes:
                add_message(task_key, f'警告: 数据有空隙，实际 {xgb_num_class} 个不重复标签（全部 {num_classes} 个类别）')

            task_label = '5 大类' if granularity == '5' else '23 细分类'
            add_message(task_key, f'【{task_label}任务】共 {num_classes} 个类别')
            if granularity == '5':
                add_message(task_key, f'  类别: {class_names}')
            add_message(task_key, f'数据加载完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}')
            add_message(task_key, f'特征数量: {len(feature_cols)}')
            add_message(task_key, f'验证集用于训练监控，测试集锁死至最终评估')
            update_progress(task_key, 8)
            
            if model_name == 'xgboost':
                os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')
                import xgboost as xgb

                # XGBoost multi:softprob 要求标签 0..num_classes-1 连续且全部出现。
                # 训练集中缺失的类别（例如 20% 子集缺少 perl，其固定索引为 12）用零权重
                # 虚拟样本补齐，保证模型输出空间恒为 num_classes（23），与神经网络一致，
                # 无需标签重编码，预测空间与编码空间直接对应。
                unique_train = np.unique(y_train)
                if len(unique_train) < num_classes:
                    missing = sorted(set(range(num_classes)) - set(unique_train.tolist()))
                    add_message(task_key, f'训练集缺少 {len(missing)} 个类别，添加零权重虚拟样本补齐: {missing}')
                    X_dummy = np.zeros((len(missing), X_train.shape[1]), dtype=np.float32)
                    y_dummy = np.array(missing, dtype=np.int64)
                    X_train_fit = np.vstack([X_train, X_dummy])
                    y_train_fit = np.concatenate([y_train, y_dummy])
                    sample_weight = np.ones(len(X_train_fit), dtype=np.float32)
                    sample_weight[-len(missing):] = 0.0
                else:
                    X_train_fit = X_train
                    y_train_fit = y_train
                    sample_weight = None
                add_message(task_key, f'XGBoost 标签空间: 0..{num_classes - 1}（{num_classes} 类直接训练，无重编码）')
                update_progress(task_key, 12)

                params = {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'objective': 'multi:softprob',
                    'num_class': num_classes,
                    'eval_metric': 'mlogloss',
                    'use_label_encoder': False,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist'
                }

                add_message(task_key,f'XGBoost ({granularity} 分类) 参数: n_estimators={params["n_estimators"]}, max_depth={params["max_depth"]}, num_class={params["num_class"]}')
                add_message(task_key,'开始训练（eval_set 使用验证集，非测试集）...')
                update_progress(task_key, 20)

                start_time = time.time()
                model = xgb.XGBClassifier(**params)
                # eval_set 使用验证集，非测试集；y_val/y_train 均为 0..num_classes-1 编码空间
                if sample_weight is not None:
                    model.fit(X_train_fit, y_train_fit, sample_weight=sample_weight,
                              eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train_fit, y_train_fit, eval_set=[(X_val, y_val)], verbose=False)
                train_time = time.time() - start_time
                update_progress(task_key, 85)

                add_message(task_key,f'训练完成，耗时: {train_time:.2f} 秒')
                
            elif model_name == 'dnn':
                device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
                add_message(task_key,f'使用设备: {device}')

                input_dim = X_train.shape[1]
                model = DNNMultiClass(input_dim, num_classes).to(device)
                # 多分类使用 CrossEntropyLoss
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.LongTensor(y_val).to(device)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

                add_message(task_key,f'【DNN 多分类】输出维度={num_classes}, loss=CrossEntropyLoss')
                add_message(task_key,'开始训练（epoch 监控使用验证集，非测试集）...')
                update_progress(task_key, 12)
                start_time = time.time()
                
                for epoch in range(50):
                    model.train()
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    update_progress(task_key, 12 + int((epoch + 1) / 50 * 76))

                    if (epoch + 1) % 5 == 0:
                        # 在验证集上计算 loss 和准确率（非测试集）
                        model.eval()
                        with torch.no_grad():
                            val_logits = model(X_val_tensor)
                            val_loss = criterion(val_logits, y_val_tensor).item()
                            val_acc = (torch.argmax(val_logits, dim=1) == y_val_tensor).float().mean().item()
                        add_message(task_key,f'Epoch [{epoch+1}/50] - Train Loss: {epoch_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
                
                train_time = time.time() - start_time
                update_progress(task_key, 88)
                add_message(task_key,f'训练完成，耗时: {train_time:.2f} 秒')
            
            elif model_name == 'cnn1d':
                device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
                add_message(task_key,f'使用设备: {device}')

                input_dim = X_train.shape[1]
                model = CNN1DMultiClass(input_dim, num_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.LongTensor(y_val).to(device)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

                add_message(task_key,f'【1D-CNN 多分类】输入序列长度={input_dim}, 输出维度={num_classes}, loss=CrossEntropyLoss')
                add_message(task_key,'开始训练（epoch 监控使用验证集，非测试集）...')
                update_progress(task_key, 12)
                start_time = time.time()

                for epoch in range(50):
                    model.train()
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    update_progress(task_key, 12 + int((epoch + 1) / 50 * 76))

                    if (epoch + 1) % 5 == 0:
                        model.eval()
                        with torch.no_grad():
                            val_logits = model(X_val_tensor)
                            val_loss = criterion(val_logits, y_val_tensor).item()
                            val_acc = (torch.argmax(val_logits, dim=1) == y_val_tensor).float().mean().item()
                        add_message(task_key,f'Epoch [{epoch+1}/50] - Train Loss: {epoch_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

                train_time = time.time() - start_time
                update_progress(task_key, 88)
                add_message(task_key,f'训练完成，耗时: {train_time:.2f} 秒')

            elif model_name == 'transformer':
                device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
                add_message(task_key,f'使用设备: {device}')

                input_dim = X_train.shape[1]
                model = TransformerMultiClass(input_dim, num_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.LongTensor(y_val).to(device)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

                add_message(task_key,f'【Transformer 多分类】token数={input_dim}, d_model=64, 输出维度={num_classes}, loss=CrossEntropyLoss')
                add_message(task_key,'开始训练（epoch 监控使用验证集，非测试集）...')
                update_progress(task_key, 12)
                start_time = time.time()

                for epoch in range(50):
                    model.train()
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    update_progress(task_key, 12 + int((epoch + 1) / 50 * 76))

                    if (epoch + 1) % 5 == 0:
                        model.eval()
                        with torch.no_grad():
                            val_logits = model(X_val_tensor)
                            val_loss = criterion(val_logits, y_val_tensor).item()
                            val_acc = (torch.argmax(val_logits, dim=1) == y_val_tensor).float().mean().item()
                        add_message(task_key,f'Epoch [{epoch+1}/50] - Train Loss: {epoch_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

                train_time = time.time() - start_time
                update_progress(task_key, 88)
                add_message(task_key,f'训练完成，耗时: {train_time:.2f} 秒')
            
            # ========== 最终评估：仅在测试集上做单次评估 ==========
            add_message(task_key,'最终评估：在测试集上做单次评估（测试集首次参与）...')
            update_progress(task_key, 92)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

            # 所有模型（xgboost/dnn/cnn1d/transformer）均为 23 分类任务
            if model_name in ('dnn', 'cnn1d', 'transformer'):
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                model.eval()
                with torch.no_grad():
                    logits = model(X_test_tensor).cpu().numpy()
                # 多分类：取 argmax 作为预测
                y_pred = np.argmax(logits, axis=1)
                # 概率用 softmax
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                y_prob_matrix = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                eval_classes = num_classes
            else:  # xgboost
                y_pred = model.predict(X_test)
                y_prob_matrix = model.predict_proba(X_test)
                # XGBoost 直接在 0..num_classes-1 空间训练，预测与标签空间一致
                eval_classes = model.n_classes_ if hasattr(model, 'n_classes_') else num_classes

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob_matrix, multi_class='ovr', average='weighted', labels=list(range(eval_classes)))
            except Exception:
                auc = float('nan')
            cm = confusion_matrix(y_test, y_pred, labels=list(range(eval_classes))).tolist()

            add_message(task_key,f'测试集评估结果:')
            add_message(task_key,f'  准确率: {accuracy:.4f}')
            add_message(task_key,f'  精确率 (weighted): {precision:.4f}')
            add_message(task_key,f'  召回率 (weighted): {recall:.4f}')
            add_message(task_key,f'  F1-Score (weighted): {f1:.4f}')
            add_message(task_key,f'  AUC: {auc:.4f}')
            add_message(task_key,f'  混淆矩阵: {cm}')

            # 每类详细指标（分类报告）：展示稀有攻击(r2l/u2r 等)的召回情况
            try:
                from sklearn.metrics import classification_report
                _names = class_names if 'class_names' in locals() and class_names else None
                report = classification_report(y_test, y_pred, labels=list(range(eval_classes)),
                                               target_names=_names, zero_division=0)
                add_message(task_key, '每类详细指标 (precision/recall/f1):')
                for _line in report.split('\n'):
                    if _line.strip():
                        add_message(task_key, _line)
            except Exception as _e:
                add_message(task_key, f'分类报告生成失败: {_e}')
            
            metrics = {
                'test_acc': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'train_time': train_time
            }
            
            import joblib
            
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            if model_name in ('dnn', 'cnn1d', 'transformer'):
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_{model_name}_{granularity}.pth'))
            else:
                joblib.dump(model, os.path.join(model_dir, f'model_{model_name}_{granularity}.pkl'))
            
            pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, f'results_{model_name}_{granularity}_metrics.csv'), index=False)
            
            add_message(task_key,f'模型已保存至: {model_dir}')
            add_message(task_key,f'{model_name} 模型训练完成!')

            training_status[task_key] ='completed'
            update_progress(task_key, 100)

        except Exception as e:
            import traceback
            add_message(task_key,f'错误: {str(e)}')
            add_message(task_key,traceback.format_exc())
            training_status[task_key] ='error'
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/train/status/<model_name>')
def get_train_status(model_name):
    if model_name not in ['xgboost', 'dnn', 'cnn1d', 'transformer']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    granularity = request.args.get('granularity', '23')
    if granularity not in ('5', '23'):
        return jsonify({'status': 'error', 'message': '无效的分类粒度'})
    task_key = f'{model_name}_{granularity}'

    return jsonify({
        'status': training_status.get(task_key, 'idle'),
        'messages': training_messages.get(task_key, []),
        'progress': training_progress.get(task_key, 0)
    })

@app.route('/results/compare')
def get_comparison_results():
    try:
        granularity = request.args.get('granularity', '23')
        if granularity not in ('5', '23'):
            return jsonify({'status': 'error', 'message': '无效的分类粒度'})
        results = {}

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        for model_name in ['xgboost', 'dnn', 'cnn1d', 'transformer']:
            # 优先找带粒度后缀的新文件，回退到旧的无后缀文件（兼容已有 23 类产物）
            metric_path = os.path.join(models_dir, model_name, f'results_{model_name}_{granularity}_metrics.csv')
            if not os.path.exists(metric_path) and granularity == '23':
                metric_path = os.path.join(models_dir, model_name, f'results_{model_name}_metrics.csv')
            if os.path.exists(metric_path):
                df = pd.read_csv(metric_path)
                results[model_name] = {
                    'accuracy': float(df['test_acc'].values[0]),
                    'precision': float(df['precision'].values[0]),
                    'recall': float(df['recall'].values[0]),
                    'f1': float(df['f1'].values[0]),
                    'auc': float(df['auc'].values[0]),
                    'train_time': float(df['train_time'].values[0])
                }
        
        if not results:
            return jsonify({'status': 'error', 'message': '没有找到任何模型结果，请先训练模型'})
        
        return jsonify({'status': 'success', 'results': results})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/results/model/<model_name>')
def get_model_results(model_name):
    try:
        granularity = request.args.get('granularity', '23')
        if granularity not in ('5', '23'):
            return jsonify({'status': 'error', 'message': '无效的分类粒度'})
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        metric_path = os.path.join(models_dir, model_name, f'results_{model_name}_{granularity}_metrics.csv')
        if not os.path.exists(metric_path) and granularity == '23':
            metric_path = os.path.join(models_dir, model_name, f'results_{model_name}_metrics.csv')
        
        if not os.path.exists(metric_path):
            return jsonify({'status': 'error', 'message': f'{model_name} 模型结果不存在'})
        
        df = pd.read_csv(metric_path)
        
        return jsonify({
            'status': 'success',
            'results': {
                'accuracy': float(df['test_acc'].values[0]),
                'precision': float(df['precision'].values[0]),
                'recall': float(df['recall'].values[0]),
                'f1': float(df['f1'].values[0]),
                'auc': float(df['auc'].values[0]),
                'train_time': float(df['train_time'].values[0]),
                'confusion_matrix': eval(df['confusion_matrix'].values[0]) if 'confusion_matrix' in df.columns else []
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/features/importance/<model_name>')
def get_feature_importance(model_name):
    try:
        granularity = request.args.get('granularity', '23')
        if granularity not in ('5', '23'):
            return jsonify({'status': 'error', 'message': '无效的分类粒度'})
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        if model_name in ('dnn', 'cnn1d', 'transformer'):
            return jsonify({'status': 'error', 'message': f'{model_name} 模型不支持特征重要性分析'})
        
        model_path = os.path.join(models_dir, model_name, f'model_{model_name}_{granularity}.pkl')
        if not os.path.exists(model_path) and granularity == '23':
            model_path = os.path.join(models_dir, model_name, f'model_{model_name}.pkl')
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': f'{model_name}({granularity}类) 模型不存在'})
        
        import joblib
        model = joblib.load(model_path)
        
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain_preprocessed_train.csv')
        df = pd.read_csv(data_path)
        
        exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[-20:]
        
        top_features = []
        for i in indices:
            top_features.append({
                'feature': feature_cols[i],
                'importance': float(importance[i])
            })
        
        top_features.sort(key=lambda x: -x['importance'])
        
        return jsonify({'status': 'success', 'features': top_features})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset/status')
def reset_status():
    global training_status, training_messages, training_progress
    training_status = {k: 'idle' for k in _STATUS_KEYS}
    training_messages = {k: [] for k in _STATUS_KEYS}
    training_progress = {k: 0 for k in _STATUS_KEYS}
    return jsonify({'status': 'success'})

@app.route('/test/model', methods=['POST'])
def test_model():
    """使用新的测试集评估已训练好的模型"""
    model_name = request.json.get('model_name')
    test_file = request.json.get('test_file', 'train_test')  # 默认使用 train_test 文件
    granularity = request.json.get('granularity', '23')
    if granularity not in ('5', '23'):
        return jsonify({'status': 'error', 'message': '无效的分类粒度，仅支持 5 或 23'})
    
    if model_name not in ['xgboost', 'dnn', 'cnn1d', 'transformer']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    task_key = f'test_{model_name}_{granularity}'
    if training_status[task_key] == 'running':
        return jsonify({'status': 'error', 'message': f'{model_name}({granularity}类) 正在测试中'})
    
    training_status[task_key] = 'running'
    training_messages[task_key] = []
    training_progress[task_key] = 0

    def run_testing():
        try:
            add_message(task_key, f'开始测试 {model_name} 模型...')
            update_progress(task_key, 3)

            import warnings
            warnings.filterwarnings('ignore')
            import numpy as np
            np.seterr(all='ignore')

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 加载测试数据
            test_path = os.path.join(base_dir, 'Train', test_file)
            if not os.path.exists(test_path):
                add_message(task_key, f'错误: 测试数据文件不存在: {test_path}')
                training_status[task_key] = 'error'
                return
            
            add_message(task_key, f'加载测试数据: {test_path}')
            
            # 使用预处理模块的列名定义
            from data_preprocessing.data_preprocessing import COLUMN_NAMES, ATTACK_CATEGORIES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
            import joblib
            
            df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
            add_message(task_key, f'测试数据形状: {df_test.shape}')
            update_progress(task_key, 10)
            
            # 加载预处理器（训练时保存的）
            ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
            scaler_path = os.path.join(base_dir, 'Train', 'scaler_standard.pkl')
            
            if not os.path.exists(ohe_path) or not os.path.exists(scaler_path):
                add_message(task_key, '错误: 预处理器文件不存在，请先运行数据预处理')
                training_status[task_key] = 'error'
                return
            
            ohe = joblib.load(ohe_path)
            scaler = joblib.load(scaler_path)
            add_message(task_key, '已加载预处理器')
            
            # 标签处理（多分类）
            df_test['label_binary'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
            
            # 按分类粒度为测试数据生成标签
            if granularity == '5':
                class_file = os.path.join(base_dir, 'Train', 'encoder_category_5_classes.txt')
            else:
                class_file = os.path.join(base_dir, 'Train', 'encoder_multiclass_23_classes.txt')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                label_to_idx = {name: i for i, name in enumerate(class_names)}
                if granularity == '5':
                    # 5 类：原始攻击先经 ATTACK_CATEGORIES 归到 5 大类，再映射到索引
                    df_test['label_multiclass_encoded'] = df_test['label'].map(
                        lambda x: ATTACK_CATEGORIES.get(x, 'normal')).map(label_to_idx).fillna(
                        label_to_idx.get('normal', 0)).astype(int)
                else:
                    # 未知标签(训练集23类之外的新攻击)映射为 -1，评估时剔除，与 eval_*_train_test.py 对齐
                    df_test['label_multiclass_encoded'] = df_test['label'].map(label_to_idx).fillna(-1).astype(int)
            else:
                if granularity == '5':
                    # 5 类编码器不存在：动态从训练集 label_category 列构建映射
                    train_csv = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
                    cat_classes = sorted(pd.read_csv(train_csv, usecols=['label_category'])[
                        'label_category'].dropna().unique().tolist())
                    label_to_idx = {c: i for i, c in enumerate(cat_classes)}
                    df_test['label_multiclass_encoded'] = df_test['label'].map(
                        lambda x: ATTACK_CATEGORIES.get(x, 'normal')).map(label_to_idx).fillna(
                        label_to_idx.get('normal', 0)).astype(int)
                    add_message(task_key, f'未找到 5 类编码器，动态构建类别: {cat_classes}')
                else:
                    df_test['label_multiclass_encoded'] = df_test['label_binary']
            
            y_test = df_test['label_multiclass_encoded'].values
            unique_labels, label_counts = np.unique(y_test, return_counts=True)
            add_message(task_key, f'测试集多分类标签分布: {len(unique_labels)} 个类别, 样本数={len(y_test)}')
            update_progress(task_key, 25)
            
            # One-Hot 编码（使用训练时 fit 的编码器）
            ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
            ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
            df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
            df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
            df_test_enc = pd.concat([df_rest, df_ohe], axis=1)
            
            # 补齐列（训练集可能有一些测试集没有的类别）
            train_feature_cols = None
            train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
            if os.path.exists(train_csv_path):
                df_train_sample = pd.read_csv(train_csv_path, nrows=1)
                exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded',
                                'label_multiclass', 'label_multiclass_encoded']
                train_feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]
            
            if train_feature_cols:
                for col in train_feature_cols:
                    if col not in df_test_enc.columns:
                        df_test_enc[col] = 0.0
                df_test_enc = df_test_enc[train_feature_cols + ['label_binary']]
            
            # 标准化（使用训练时 fit 的标准化器）
            numeric_cols = [col for col in NUMERIC_FEATURES if col in df_test_enc.columns]
            df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])
            add_message(task_key, f'测试数据预处理完成')
            update_progress(task_key, 45)
            
            # 准备特征矩阵（优先使用训练集特征列，确保维度一致）
            if train_feature_cols:
                feature_cols = train_feature_cols
                for col in feature_cols:
                    if col not in df_test_enc.columns:
                        df_test_enc[col] = 0.0
                X_test = df_test_enc[feature_cols].values.astype(np.float32)
            else:
                exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded',
                                'label_multiclass', 'label_multiclass_encoded']
                feature_cols = [col for col in df_test_enc.columns if col not in exclude_cols]
                X_test = df_test_enc[feature_cols].values.astype(np.float32)
            
            add_message(task_key, f'测试特征矩阵: {X_test.shape}')
            
            # 加载模型
            model_dir = os.path.join(base_dir, 'models', model_name)
            if model_name in ('dnn', 'cnn1d', 'transformer'):
                model_path = os.path.join(model_dir, f'model_{model_name}_{granularity}.pth')
                if not os.path.exists(model_path) and granularity == '23':
                    model_path = os.path.join(model_dir, f'model_{model_name}.pth')
            else:
                model_path = os.path.join(model_dir, f'model_{model_name}_{granularity}.pkl')
                if not os.path.exists(model_path) and granularity == '23':
                    model_path = os.path.join(model_dir, f'model_{model_name}.pkl')
            
            if not os.path.exists(model_path):
                add_message(task_key, f'错误: 模型文件不存在，请先训练 {model_name} 模型')
                training_status[task_key] = 'error'
                return
            
            add_message(task_key, f'加载模型: {model_path}')
            update_progress(task_key, 55)
            
            # 模型预测
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
            
            if model_name in ('dnn', 'cnn1d', 'transformer'):
                device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
                state_dict = torch.load(model_path, map_location=device)

                if model_name == 'dnn':
                    input_dim = state_dict['network.0.weight'].shape[1]
                    num_classes = state_dict['network.12.weight'].shape[0]
                    model = DNNMultiClass(input_dim=input_dim, num_classes=num_classes).to(device)
                elif model_name == 'cnn1d':
                    # CNN1D 的 conv 权重不依赖 input_dim，只需推断 num_classes
                    num_classes = state_dict['classifier.3.weight'].shape[0]
                    model = CNN1DMultiClass(input_dim=1, num_classes=num_classes).to(device)
                else:  # transformer
                    # input_dim 从 pos_embedding shape 推断
                    input_dim = state_dict['pos_embedding'].shape[1]
                    num_classes = state_dict['classifier.1.weight'].shape[0]
                    model = TransformerMultiClass(input_dim=input_dim, num_classes=num_classes).to(device)

                model.load_state_dict(state_dict)
                model.eval()

                X_test_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    logits = model(X_test_tensor).cpu().numpy()
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                y_prob_matrix = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # 多分类预测
                y_pred = np.argmax(y_prob_matrix, axis=1)

            elif model_name == 'xgboost':
                model = joblib.load(model_path)
                # 新版 XGBoost 在训练阶段已通过零权重虚拟样本补齐到 num_classes(=23) 类，
                # 预测空间与编码空间直接一致，无需任何标签重编码。
                y_pred = model.predict(X_test).astype(np.int64)
                y_prob_matrix = model.predict_proba(X_test)
                num_classes = model.n_classes_ if hasattr(model, 'n_classes_') else y_prob_matrix.shape[1]

            add_message(task_key, f'预测完成')
            update_progress(task_key, 75)

            # 与 eval_*_train_test.py 对齐：剔除测试集中未知标签(-1)的样本，
            # 这些样本属于训练集 23 类之外的新攻击类型，无法做多分类评估。
            valid_mask = y_test >= 0
            n_dropped = int((~valid_mask).sum())
            if n_dropped > 0:
                add_message(task_key, f'注: 测试集中 {n_dropped} 个样本标签不在训练集 23 类中，评估时已剔除')
                y_test = y_test[valid_mask]
                y_pred = y_pred[valid_mask]
                y_prob_matrix = y_prob_matrix[valid_mask]

            # 所有模型均为 23 分类任务，统一使用多分类评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # 多分类 AUC: OvR weighted
            try:
                n_cols = y_prob_matrix.shape[1]
                auc = roc_auc_score(y_test, y_prob_matrix, multi_class='ovr',
                                    average='weighted', labels=list(range(n_cols)))
            except Exception:
                try:
                    # 降级：手动计算 OvR AUC
                    from sklearn.preprocessing import label_binarize
                    all_labels = list(range(y_prob_matrix.shape[1]))
                    y_test_bin = label_binarize(y_test, classes=all_labels)
                    aucs = []
                    for c in range(y_prob_matrix.shape[1]):
                        if c < y_test_bin.shape[1] and len(np.unique(y_test_bin[:, c])) > 1:
                            aucs.append(roc_auc_score(y_test_bin[:, c], y_prob_matrix[:, c]))
                    auc = np.mean(aucs) if aucs else float('nan')
                except Exception:
                    auc = float('nan')
            cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes))).tolist()
            add_message(task_key, f'测试集评估结果（多分类 {num_classes} 类）:')
            add_message(task_key, f'  准确率: {accuracy:.4f}')
            add_message(task_key, f'  精确率 (weighted): {precision:.4f}')
            add_message(task_key, f'  召回率 (weighted): {recall:.4f}')
            add_message(task_key, f'  F1-Score (weighted): {f1:.4f}')
            add_message(task_key, f'  AUC (weighted OvR): {auc:.4f}')
            add_message(task_key, f'  混淆矩阵: {len(cm)}x{len(cm)} (已保存)')
            
            # 保存测试结果
            test_results = {
                'test_acc': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'test_file': test_file,
                'test_samples': len(y_test)
            }
            
            results_path = os.path.join(model_dir, f'results_{model_name}_{granularity}_external_test.csv')
            pd.DataFrame([test_results]).to_csv(results_path, index=False)
            add_message(task_key, f'测试结果已保存: {results_path}')
            add_message(task_key, f'{model_name} 模型测试完成!')

            training_status[task_key] = 'completed'
            update_progress(task_key, 100)

        except Exception as e:
            import traceback
            add_message(task_key, f'错误: {str(e)}')
            add_message(task_key, traceback.format_exc())
            training_status[task_key] = 'error'
    
    thread = threading.Thread(target=run_testing)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/test/status/<model_name>')
def get_test_status(model_name):
    granularity = request.args.get('granularity', '23')
    if granularity not in ('5', '23'):
        return jsonify({'status': 'error', 'message': '无效的分类粒度'})
    task_key = f'test_{model_name}_{granularity}'
    if task_key not in training_status:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    return jsonify({
        'status': training_status[task_key],
        'messages': training_messages[task_key],
        'progress': training_progress.get(task_key, 0)
    })

@app.route('/test/results/<model_name>')
def get_test_results(model_name):
    try:
        granularity = request.args.get('granularity', '23')
        if granularity not in ('5', '23'):
            return jsonify({'status': 'error', 'message': '无效的分类粒度'})
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.join(base_dir, 'models', model_name, f'results_{model_name}_{granularity}_external_test.csv')
        if not os.path.exists(results_path) and granularity == '23':
            results_path = os.path.join(base_dir, 'models', model_name, f'results_{model_name}_external_test.csv')
        
        if not os.path.exists(results_path):
            return jsonify({'status': 'error', 'message': f'{model_name} 测试结果不存在'})
        
        df = pd.read_csv(results_path)
        
        return jsonify({
            'status': 'success',
            'results': {
                'accuracy': float(df['test_acc'].values[0]),
                'precision': float(df['precision'].values[0]),
                'recall': float(df['recall'].values[0]),
                'f1': float(df['f1'].values[0]),
                'auc': float(df['auc'].values[0]),
                'test_samples': int(df['test_samples'].values[0]),
                'test_file': df['test_file'].values[0] if 'test_file' in df.columns else 'unknown',
                'confusion_matrix': eval(df['confusion_matrix'].values[0]) if 'confusion_matrix' in df.columns else []
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=True)