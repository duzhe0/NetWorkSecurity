import os
import sys
import json
import time
import base64
import threading
import pandas as pd
import numpy as np
from io import BytesIO

import pickle
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'network_security_key'

training_status = {
    'xgboost': 'idle',
    'random_forest': 'idle',
    'dnn': 'idle',
    'isolation_forest': 'idle',
    'autoencoder': 'idle',
    'preprocessing': 'idle',
    'test_xgboost': 'idle',
    'test_random_forest': 'idle',
    'test_dnn': 'idle',
    'test_isolation_forest': 'idle',
    'test_autoencoder': 'idle'
}

training_messages = {
    'xgboost': [],
    'random_forest': [],
    'dnn': [],
    'isolation_forest': [],
    'autoencoder': [],
    'preprocessing': [],
    'test_xgboost': [],
    'test_random_forest': [],
    'test_dnn': [],
    'test_isolation_forest': []
}

def add_message(task, msg):
    if task in training_messages:
        training_messages[task].append(msg)
        if len(training_messages[task]) > 100:
            training_messages[task] = training_messages[task][-50:]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    if training_status['preprocessing'] == 'running':
        return jsonify({'status': 'error', 'message': '预处理正在进行中'})
    
    dataset = request.json.get('dataset', '20percent')
    training_status['preprocessing'] = 'running'
    training_messages['preprocessing'] = []
    
    def run_preprocessing():
        try:
            data_file = 'KDDTrain+.txt' if dataset == 'full' else 'KDDTrain+_20Percent.txt'
            add_message('preprocessing', f'开始数据预处理（防数据泄漏版本：先切分，后拟合）...')
            add_message('preprocessing', f'使用数据集: {data_file}')
            
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
            
            add_message('preprocessing', '数据探索中...')
            df = explore_data(df)
            
            add_message('preprocessing', '标签预处理中...')
            df, label_encoder = preprocess_labels(df)
            
            add_message('preprocessing', '数据集划分（训练70%/验证10%/测试20%）...')
            df_train, df_val, df_test = split_data(df, test_size=0.2, val_size=0.1)
            
            add_message('preprocessing', '特征编码与标准化（仅训练集fit，验证/测试集只transform）...')
            df_train_p, df_val_p, df_test_p, ohe, scaler, ohe_names = encode_and_scale(df_train, df_val, df_test)
            
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train')
            
            df_train_p.to_csv(os.path.join(output_dir, 'KDDTrain_preprocessed_train.csv'), index=False)
            df_val_p.to_csv(os.path.join(output_dir, 'KDDTrain_preprocessed_val.csv'), index=False)
            df_test_p.to_csv(os.path.join(output_dir, 'KDDTrain_preprocessed_test.csv'), index=False)
            
            import joblib
            joblib.dump(ohe, os.path.join(output_dir, 'encoder_onehot.pkl'))
            joblib.dump(scaler, os.path.join(output_dir, 'scaler_standard.pkl'))
            
            add_message('preprocessing', f'预处理完成！')
            add_message('preprocessing', f'训练集: {df_train_p.shape}')
            add_message('preprocessing', f'验证集: {df_val_p.shape}')
            add_message('preprocessing', f'测试集: {df_test_p.shape}')
            add_message('preprocessing', f'StandardScaler/OneHotEncoder 仅在训练集上 fit')
            add_message('preprocessing', f'验证集用于训练监控，测试集仅用于最终评估')
            
            training_status['preprocessing'] = 'completed'
            
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
        'messages': training_messages['preprocessing']
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
    
    if model_name not in ['xgboost', 'random_forest', 'dnn', 'isolation_forest', 'autoencoder']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    if training_status[model_name] == 'running':
        return jsonify({'status': 'error', 'message': f'{model_name} 正在训练中'})
    
    training_status[model_name] = 'running'
    training_messages[model_name] = []
    
    def run_training():
        try:
            add_message(model_name, f'开始训练 {model_name} 模型（防数据泄漏版本）...')
            
            import warnings
            warnings.filterwarnings('ignore')
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
            val_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_val.csv')
            test_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_test.csv')
            
            for p in [train_path, val_path, test_path]:
                if not os.path.exists(p):
                    add_message(model_name, f'错误: 数据文件不存在: {p}')
                    add_message(model_name, '请先运行数据预处理')
                    training_status[model_name] = 'error'
                    return
            
            # 加载训练集、验证集、测试集（预处理时已正确划分）
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)
            
            exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
            feature_cols = [col for col in df_train.columns if col not in exclude_cols]
            
            X_train = df_train[feature_cols].values
            y_train = df_train['label_binary'].values
            X_val = df_val[feature_cols].values
            y_val = df_val['label_binary'].values
            X_test = df_test[feature_cols].values
            y_test = df_test['label_binary'].values
            
            add_message(model_name, f'数据加载完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}')
            add_message(model_name, f'特征数量: {len(feature_cols)}')
            add_message(model_name, f'验证集用于训练监控，测试集锁死至最终评估')
            
            if model_name == 'xgboost':
                os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')
                import xgboost as xgb
                
                params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                add_message(model_name, f'XGBoost参数: {params}')
                add_message(model_name, '开始训练（eval_set 使用验证集，非测试集）...')
                
                start_time = time.time()
                model = xgb.XGBClassifier(**params)
                # 关键修复：eval_set 使用验证集，而非测试集
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                train_time = time.time() - start_time
                
                add_message(model_name, f'训练完成，耗时: {train_time:.2f} 秒')
                
            elif model_name == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                
                params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                add_message(model_name, f'随机森林参数: {params}')
                add_message(model_name, '开始训练...')
                
                start_time = time.time()
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                add_message(model_name, f'训练完成，耗时: {train_time:.2f} 秒')
                
            elif model_name == 'dnn':
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                
                class DNN(nn.Module):
                    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
                        super(DNN, self).__init__()
                        layers = []
                        prev_dim = input_dim
                        for hidden_dim in hidden_dims:
                            layers.append(nn.Linear(prev_dim, hidden_dim))
                            layers.append(nn.BatchNorm1d(hidden_dim))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout_rate))
                            prev_dim = hidden_dim
                        layers.append(nn.Linear(prev_dim, 1))
                        layers.append(nn.Sigmoid())
                        self.network = nn.Sequential(*layers)
                    
                    def forward(self, x):
                        return self.network(x)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                add_message(model_name, f'使用设备: {device}')
                
                input_dim = X_train.shape[1]
                model = DNN(input_dim).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
                # 关键修复：使用验证集 tensor，非测试集
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
                
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                
                add_message(model_name, '开始训练（epoch 监控使用验证集，非测试集）...')
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
                    
                    if (epoch + 1) % 10 == 0:
                        # 在验证集上计算 loss（非测试集）
                        model.eval()
                        with torch.no_grad():
                            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
                        add_message(model_name, f'Epoch [{epoch+1}/50] - Train Loss: {epoch_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f}')
                
                train_time = time.time() - start_time
                add_message(model_name, f'训练完成，耗时: {train_time:.2f} 秒')
            
            elif model_name == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                import numpy as np

                # 仅使用正常流量训练（经典异常检测方法）
                X_train_normal = X_train[y_train == 0]
                add_message(model_name, f'Isolation Forest 参数:')
                add_message(model_name, f'  n_estimators: 500')
                add_message(model_name, f'  max_samples: 256')
                add_message(model_name, f'  contamination: 0.05 (由阈值优化决定)')
                add_message(model_name, f'  仅使用正常流量训练: {len(X_train_normal)} 样本')
                add_message(model_name, '开始训练（异常检测，仅学习正常模式）...')

                start_time = time.time()
                model = IsolationForest(
                    n_estimators=500,
                    max_samples=256,
                    contamination=0.05,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train_normal)
                train_time = time.time() - start_time

                add_message(model_name, f'训练完成，耗时: {train_time:.2f} 秒')

                # 使用正常训练数据分数分布设定阈值（P15分位对零日攻击更敏感）
                from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
                add_message(model_name, '阈值设定：使用正常训练数据分数分布的 P15 分位...')
                
                normal_scores = model.decision_function(X_train_normal)
                threshold_percentile = 15
                best_threshold = np.percentile(normal_scores, threshold_percentile)
                add_message(model_name, f'正常训练数据分数分布 P{threshold_percentile} 分位: {best_threshold:.4f}')

                # 验证集评估
                val_scores = model.decision_function(X_val)
                val_pred = (val_scores < best_threshold).astype(int)
                val_acc = accuracy_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred)
                val_prec = precision_score(y_val, val_pred, zero_division=0)
                val_rec = recall_score(y_val, val_pred, zero_division=0)
                add_message(model_name, f'验证集准确率: {val_acc:.4f}, 精确率: {val_prec:.4f}, 召回率: {val_rec:.4f}, F1: {val_f1:.4f}')

                # 保存阈值
                import pickle
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', model_name)
                os.makedirs(model_dir, exist_ok=True)
                with open(os.path.join(model_dir, 'isolation_forest_params.pkl'), 'wb') as f:
                    pickle.dump({'threshold': best_threshold}, f)
            
            elif model_name == 'autoencoder':
                import numpy as np
                import torch
                import torch.nn as nn
                import torch.optim as optim
                import pickle
                from sklearn.metrics import accuracy_score, f1_score
                add_message(model_name, 'AutoEncoder 异常检测模型 - 仅使用正常流量训练...')
                
                start_time = time.time()
                
                X_train_normal = X_train[y_train == 0]
                add_message(model_name, f'仅使用正常流量训练: {len(X_train_normal)} 样本')
                
                class AutoEncoder(nn.Module):
                    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
                        super(AutoEncoder, self).__init__()
                        layers = []
                        prev_dim = input_dim
                        for hidden_dim in hidden_dims:
                            layers.append(nn.Linear(prev_dim, hidden_dim))
                            layers.append(nn.BatchNorm1d(hidden_dim))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout_rate))
                            prev_dim = hidden_dim
                        self.encoder = nn.Sequential(*layers)
                        
                        layers_dec = []
                        for hidden_dim in reversed(hidden_dims[:-1]):
                            layers_dec.append(nn.Linear(prev_dim, hidden_dim))
                            layers_dec.append(nn.BatchNorm1d(hidden_dim))
                            layers_dec.append(nn.ReLU())
                            layers_dec.append(nn.Dropout(dropout_rate))
                            prev_dim = hidden_dim
                        layers_dec.append(nn.Linear(prev_dim, input_dim))
                        self.decoder = nn.Sequential(*layers_dec)
                    
                    def forward(self, x):
                        encoded = self.encoder(x)
                        decoded = self.decoder(encoded)
                        return decoded
                
                input_dim = X_train_normal.shape[1]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = AutoEncoder(input_dim=input_dim).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                
                X_train_tensor = torch.FloatTensor(X_train_normal).to(device)
                from torch.utils.data import DataLoader, TensorDataset
                train_dataset = TensorDataset(X_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                
                add_message(model_name, '[训练中...]')
                best_val_loss = float('inf')
                patience = 0
                for epoch in range(100):
                    model.train()
                    total_loss = 0
                    for batch in train_loader:
                        optimizer.zero_grad()
                        x = batch[0]
                        outputs = model(x)
                        loss = criterion(outputs, x)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, X_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                    else:
                        patience += 1
                    
                    if (epoch + 1) % 20 == 0:
                        add_message(model_name, f'Epoch [{epoch+1}/100] - Train Loss: {total_loss/len(train_loader):.6f} - Val Loss: {val_loss:.6f}')
                    
                    if patience >= 10:
                        add_message(model_name, f'早停: Epoch {epoch+1}')
                        break
                
                # 使用验证集优化阈值（平衡准确率与F1）
                add_message(model_name, '[阈值设定: 在验证集上优化阈值...]')
                with torch.no_grad():
                    X_train_normal_tensor = torch.FloatTensor(X_train_normal).to(device)
                    train_normal_outputs = model(X_train_normal_tensor)
                    normal_errors = torch.mean((train_normal_outputs - X_train_normal_tensor) ** 2, dim=1).cpu().numpy()
                    
                    val_outputs = model(X_val_tensor)
                    val_errors = torch.mean((val_outputs - X_val_tensor) ** 2, dim=1).cpu().numpy()
                
                thresholds = np.linspace(normal_errors.min(), normal_errors.max(), 200)
                best_acc = 0
                best_threshold = 0
                best_f1 = 0
                
                for thresh in thresholds:
                    val_pred = (val_errors > thresh).astype(int)
                    acc = accuracy_score(y_val, val_pred)
                    f1 = f1_score(y_val, val_pred, zero_division=0)
                    if acc > best_acc:
                        best_acc = acc
                        best_threshold = thresh
                    if f1 > best_f1:
                        best_f1 = f1
                
                add_message(model_name, f'最优阈值: {best_threshold:.6f}')
                add_message(model_name, f'验证集准确率: {best_acc:.4f}, F1: {best_f1:.4f}')
                
                # 保存模型和阈值
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', model_name)
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'model_autoencoder.pth'))
                with open(os.path.join(model_dir, 'autoencoder_params.pkl'), 'wb') as f:
                    pickle.dump({'threshold': best_threshold, 'input_dim': input_dim}, f)
                
                train_time = time.time() - start_time
                add_message(model_name, f'训练完成，耗时: {train_time:.2f} 秒')
            
            # ========== 最终评估：仅在测试集上做单次评估 ==========
            add_message(model_name, '最终评估：在测试集上做单次评估（测试集首次参与）...')
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            if model_name == 'dnn':
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                model.eval()
                with torch.no_grad():
                    y_prob = model(X_test_tensor).cpu().numpy().flatten()
                    y_pred = (y_prob > 0.5).astype(int)
            elif model_name == 'isolation_forest':
                # 加载阈值
                params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', model_name, 'isolation_forest_params.pkl')
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                best_threshold = params['threshold']
                
                scores = model.decision_function(X_test)
                y_pred = (scores < best_threshold).astype(int)
                y_prob = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            elif model_name == 'autoencoder':
                # 加载阈值
                params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', model_name, 'autoencoder_params.pkl')
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                best_threshold = params['threshold']
                
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_errors = torch.mean((test_outputs - X_test_tensor) ** 2, dim=1).cpu().numpy()
                y_pred = (test_errors > best_threshold).astype(int)
                y_prob = (test_errors - test_errors.min()) / (test_errors.max() - test_errors.min() + 1e-10)
            else:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            add_message(model_name, f'测试集评估结果:')
            add_message(model_name, f'  准确率: {accuracy:.4f}')
            add_message(model_name, f'  精确率: {precision:.4f}')
            add_message(model_name, f'  召回率: {recall:.4f}')
            add_message(model_name, f'  F1-Score: {f1:.4f}')
            add_message(model_name, f'  AUC: {auc:.4f}')
            add_message(model_name, f'  混淆矩阵: {cm}')
            
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
            
            if model_name == 'dnn':
                torch.save(model.state_dict(), os.path.join(model_dir, 'model_dnn.pth'))
            elif model_name == 'autoencoder':
                pass
            else:
                joblib.dump(model, os.path.join(model_dir, f'model_{model_name}.pkl'))
            
            pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, f'results_{model_name}_metrics.csv'), index=False)
            
            add_message(model_name, f'模型已保存至: {model_dir}')
            add_message(model_name, f'{model_name} 模型训练完成!')
            
            training_status[model_name] = 'completed'
            
        except Exception as e:
            import traceback
            add_message(model_name, f'错误: {str(e)}')
            add_message(model_name, traceback.format_exc())
            training_status[model_name] = 'error'
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/train/status/<model_name>')
def get_train_status(model_name):
    if model_name not in ['xgboost', 'random_forest', 'dnn', 'isolation_forest', 'autoencoder']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    return jsonify({
        'status': training_status[model_name],
        'messages': training_messages[model_name]
    })

@app.route('/results/compare')
def get_comparison_results():
    try:
        results = {}
        
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        
        for model_name in ['xgboost', 'random_forest', 'dnn', 'isolation_forest']:
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
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
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
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        
        if model_name == 'dnn':
            return jsonify({'status': 'error', 'message': 'DNN模型不支持特征重要性分析'})
        
        model_path = os.path.join(models_dir, model_name, f'model_{model_name}.pkl')
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': f'{model_name} 模型不存在'})
        
        import joblib
        model = joblib.load(model_path)
        
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain_preprocessed.csv')
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
    global training_status, training_messages
    training_status = {
        'xgboost': 'idle',
        'random_forest': 'idle',
        'dnn': 'idle',
        'isolation_forest': 'idle',
        'preprocessing': 'idle',
        'test_xgboost': 'idle',
        'test_random_forest': 'idle',
        'test_dnn': 'idle',
        'test_isolation_forest': 'idle'
    }
    training_messages = {
        'xgboost': [],
        'random_forest': [],
        'dnn': [],
        'isolation_forest': [],
        'preprocessing': [],
        'test_xgboost': [],
        'test_random_forest': [],
        'test_dnn': [],
        'test_isolation_forest': []
    }
    return jsonify({'status': 'success'})

@app.route('/test/model', methods=['POST'])
def test_model():
    """使用新的测试集评估已训练好的模型"""
    model_name = request.json.get('model_name')
    test_file = request.json.get('test_file', 'train_test')  # 默认使用 train_test 文件
    
    if model_name not in ['xgboost', 'random_forest', 'dnn', 'isolation_forest', 'autoencoder']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    task_key = f'test_{model_name}'
    if training_status[task_key] == 'running':
        return jsonify({'status': 'error', 'message': f'{model_name} 正在测试中'})
    
    training_status[task_key] = 'running'
    training_messages[task_key] = []
    
    def run_testing():
        try:
            add_message(task_key, f'开始测试 {model_name} 模型...')
            
            import warnings
            warnings.filterwarnings('ignore')
            
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
            
            # 标签处理
            df_test['label_binary'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
            y_test = df_test['label_binary'].values
            add_message(task_key, f'测试集标签分布: normal={(y_test==0).sum()}, attack={(y_test==1).sum()}')
            
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
                exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
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
            
            # 准备特征矩阵
            exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
            feature_cols = [col for col in df_test_enc.columns if col not in exclude_cols]
            X_test = df_test_enc[feature_cols].values
            
            add_message(task_key, f'测试特征矩阵: {X_test.shape}')
            
            # 加载模型
            model_dir = os.path.join(base_dir, 'models', model_name)
            if model_name == 'dnn':
                model_path = os.path.join(model_dir, 'model_dnn.pth')
            else:
                model_path = os.path.join(model_dir, f'model_{model_name}.pkl')
            
            if not os.path.exists(model_path):
                add_message(task_key, f'错误: 模型文件不存在，请先训练 {model_name} 模型')
                training_status[task_key] = 'error'
                return
            
            add_message(task_key, f'加载模型: {model_path}')
            
            # 模型预测
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
            
            if model_name == 'dnn':
                import torch
                import torch.nn as nn
                
                class DNN(nn.Module):
                    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
                        super(DNN, self).__init__()
                        layers = []
                        prev_dim = input_dim
                        for hidden_dim in hidden_dims:
                            layers.append(nn.Linear(prev_dim, hidden_dim))
                            layers.append(nn.BatchNorm1d(hidden_dim))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout_rate))
                            prev_dim = hidden_dim
                        layers.append(nn.Linear(prev_dim, 1))
                        layers.append(nn.Sigmoid())
                        self.network = nn.Sequential(*layers)
                    
                    def forward(self, x):
                        return self.network(x)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = DNN(input_dim=X_test.shape[1]).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    y_prob = model(X_test_tensor).cpu().numpy().flatten()
                y_pred = (y_prob > 0.5).astype(int)
                
            elif model_name == 'isolation_forest':
                import numpy as np
                model = joblib.load(model_path)
                
                params_path = os.path.join(model_dir, 'isolation_forest_params.pkl')
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                best_threshold = params['threshold']
                
                scores = model.decision_function(X_test)
                y_pred = (scores < best_threshold).astype(int)
                y_prob = 1 - (scores - scores.min()) / (scores.max() - scores.min())
                
            elif model_name == 'autoencoder':
                import numpy as np
                
                params_path = os.path.join(model_dir, 'autoencoder_params.pkl')
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                best_threshold = params['threshold']
                input_dim = params['input_dim']
                
                class AutoEncoder(nn.Module):
                    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
                        super(AutoEncoder, self).__init__()
                        layers = []
                        prev_dim = input_dim
                        for hidden_dim in hidden_dims:
                            layers.append(nn.Linear(prev_dim, hidden_dim))
                            layers.append(nn.BatchNorm1d(hidden_dim))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout_rate))
                            prev_dim = hidden_dim
                        self.encoder = nn.Sequential(*layers)
                        
                        layers_dec = []
                        for hidden_dim in reversed(hidden_dims[:-1]):
                            layers_dec.append(nn.Linear(prev_dim, hidden_dim))
                            layers_dec.append(nn.BatchNorm1d(hidden_dim))
                            layers_dec.append(nn.ReLU())
                            layers_dec.append(nn.Dropout(dropout_rate))
                            prev_dim = hidden_dim
                        layers_dec.append(nn.Linear(prev_dim, input_dim))
                        self.decoder = nn.Sequential(*layers_dec)
                    
                    def forward(self, x):
                        encoded = self.encoder(x)
                        decoded = self.decoder(encoded)
                        return decoded
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = AutoEncoder(input_dim=input_dim).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_errors = torch.mean((test_outputs - X_test_tensor) ** 2, dim=1).cpu().numpy()
                y_pred = (test_errors > best_threshold).astype(int)
                y_prob = (test_errors - test_errors.min()) / (test_errors.max() - test_errors.min() + 1e-10)
                
            else:
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            add_message(task_key, f'预测完成')
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            add_message(task_key, f'测试集评估结果:')
            add_message(task_key, f'  准确率: {accuracy:.4f}')
            add_message(task_key, f'  精确率: {precision:.4f}')
            add_message(task_key, f'  召回率: {recall:.4f}')
            add_message(task_key, f'  F1-Score: {f1:.4f}')
            add_message(task_key, f'  AUC: {auc:.4f}')
            add_message(task_key, f'  混淆矩阵: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]')
            
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
            
            results_path = os.path.join(model_dir, f'results_{model_name}_external_test.csv')
            pd.DataFrame([test_results]).to_csv(results_path, index=False)
            add_message(task_key, f'测试结果已保存: {results_path}')
            add_message(task_key, f'{model_name} 模型测试完成!')
            
            training_status[task_key] = 'completed'
            
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
    task_key = f'test_{model_name}'
    if task_key not in training_status:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    return jsonify({
        'status': training_status[task_key],
        'messages': training_messages[task_key]
    })

@app.route('/test/results/<model_name>')
def get_test_results(model_name):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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