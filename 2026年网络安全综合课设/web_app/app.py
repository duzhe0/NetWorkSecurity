import os
import sys
import json
import time
import base64
import threading
import pandas as pd
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'network_security_key'

training_status = {
    'xgboost': 'idle',
    'random_forest': 'idle',
    'dnn': 'idle',
    'preprocessing': 'idle'
}

training_messages = {
    'xgboost': [],
    'random_forest': [],
    'dnn': [],
    'preprocessing': []
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
    
    training_status['preprocessing'] = 'running'
    training_messages['preprocessing'] = []
    
    def run_preprocessing():
        try:
            add_message('preprocessing', '开始数据预处理...')
            
            import warnings
            warnings.filterwarnings('ignore')
            
            from data_preprocessing.data_preprocessing import (
                load_data, explore_data, preprocess_labels,
                encode_categorical_features, scale_numeric_features,
                COLUMN_NAMES, ATTACK_CATEGORIES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
            )
            
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain+_20Percent.txt')
            
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
            
            add_message('preprocessing', '类别型特征编码中...')
            df_encoded = encode_categorical_features(df)
            
            exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
            feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
            
            add_message('preprocessing', '数值型特征标准化中...')
            df_processed, scaler = scale_numeric_features(df_encoded, feature_cols)
            
            output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain_preprocessed.csv')
            df_processed.to_csv(output_path, index=False)
            
            add_message('preprocessing', f'预处理完成！数据已保存至: {output_path}')
            add_message('preprocessing', f'原始数据: {df.shape}')
            add_message('preprocessing', f'预处理后: {df_processed.shape}')
            
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
        raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain+_20Percent.txt')
        processed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain_preprocessed.csv')
        
        if not os.path.exists(raw_data_path):
            return jsonify({'status': 'error', 'message': '原始数据文件不存在，请将 KDDTrain+_20Percent.txt 放入 Train 目录'})
        
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
    
    if model_name not in ['xgboost', 'random_forest', 'dnn']:
        return jsonify({'status': 'error', 'message': '无效的模型名称'})
    
    if training_status[model_name] == 'running':
        return jsonify({'status': 'error', 'message': f'{model_name} 正在训练中'})
    
    training_status[model_name] = 'running'
    training_messages[model_name] = []
    
    def run_training():
        try:
            add_message(model_name, f'开始训练 {model_name} 模型...')
            
            import warnings
            warnings.filterwarnings('ignore')
            
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train', 'KDDTrain_preprocessed.csv')
            
            if not os.path.exists(data_path):
                add_message(model_name, '错误: 预处理数据文件不存在')
                training_status[model_name] = 'error'
                return
            
            add_message(model_name, f'加载数据: {data_path}')
            df = pd.read_csv(data_path)
            
            exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols].values
            y = df['label_binary'].values
            
            add_message(model_name, f'数据加载完成: {len(X)} 样本, {len(feature_cols)} 特征')
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            add_message(model_name, f'数据划分完成: 训练集 {len(X_train)}, 测试集 {len(X_test)}')
            
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
                add_message(model_name, '开始训练...')
                
                start_time = time.time()
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
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
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
                
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                
                add_message(model_name, '开始训练...')
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
                        add_message(model_name, f'Epoch [{epoch+1}/50] - Loss: {epoch_loss/len(train_loader):.4f}')
                
                train_time = time.time() - start_time
                add_message(model_name, f'训练完成，耗时: {train_time:.2f} 秒')
            
            add_message(model_name, '模型评估中...')
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            if model_name == 'dnn':
                model.eval()
                with torch.no_grad():
                    y_prob = model(X_test_tensor).cpu().numpy().flatten()
                    y_pred = (y_prob > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            add_message(model_name, f'评估结果:')
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
    if model_name not in ['xgboost', 'random_forest', 'dnn']:
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
        
        for model_name in ['xgboost', 'random_forest', 'dnn']:
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
        'preprocessing': 'idle'
    }
    training_messages = {
        'xgboost': [],
        'random_forest': [],
        'dnn': [],
        'preprocessing': []
    }
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)