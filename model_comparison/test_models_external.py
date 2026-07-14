#!/usr/bin/env python3
"""
独立模型测试脚本
使用新的测试集（train_test）评估已训练好的模型

使用方法:
    python3 test_models_external.py --model xgboost
    python3 test_models_external.py --model dnn
    python3 test_models_external.py --model isolation_forest
    python3 test_models_external.py --model autoencoder
    python3 test_models_external.py --model all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.data_preprocessing import (
    COLUMN_NAMES, ATTACK_CATEGORIES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def load_preprocessors(base_dir):
    """加载预处理器（训练时保存的）"""
    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_standard.pkl')
    
    if not os.path.exists(ohe_path):
        raise FileNotFoundError(f"OneHotEncoder 不存在: {ohe_path}\n请先运行数据预处理")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"StandardScaler 不存在: {scaler_path}\n请先运行数据预处理")
    
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
    
    return ohe, scaler


def preprocess_test_data(test_path, base_dir, ohe, scaler):
    """预处理测试数据（使用训练时的预处理器）"""
    print(f"\n[加载测试数据] {test_path}")
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"  测试数据形状: {df_test.shape}")
    
    # 标签处理
    df_test['label_binary'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    y_test = df_test['label_binary'].values
    print(f"  标签分布: normal={(y_test==0).sum()}, attack={(y_test==1).sum()}")
    
    # One-Hot 编码
    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
    df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
    df_test_enc = pd.concat([df_rest, df_ohe], axis=1)
    
    # 获取训练集的特征列顺序
    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    if os.path.exists(train_csv_path):
        df_train_sample = pd.read_csv(train_csv_path, nrows=1)
        exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
        train_feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]
        
        # 补齐列
        for col in train_feature_cols:
            if col not in df_test_enc.columns:
                df_test_enc[col] = 0.0
        
        df_test_enc = df_test_enc[train_feature_cols + ['label_binary']]
    
    # 标准化
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_test_enc.columns]
    df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])
    
    # 准备特征矩阵
    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category', 'label_category_encoded']
    feature_cols = [col for col in df_test_enc.columns if col not in exclude_cols]
    X_test = df_test_enc[feature_cols].values
    
    print(f"  测试特征矩阵: {X_test.shape}")
    
    return X_test, y_test, feature_cols


def test_xgboost(X_test, y_test, base_dir):
    """测试 XGBoost 模型"""
    model_path = os.path.join(base_dir, 'models', 'xgboost', 'model_xgboost.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先训练 XGBoost 模型")
    
    print(f"\n[加载模型] {model_path}")
    model = joblib.load(model_path)
    
    print("[预测中...]")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return y_pred, y_prob, model_path


def test_isolation_forest(X_test, y_test, base_dir):
    """测试 Isolation Forest 模型"""
    import pickle
    import numpy as np
    
    model_path = os.path.join(base_dir, 'models', 'isolation_forest', 'model_isolation_forest.pkl')
    params_path = os.path.join(base_dir, 'models', 'isolation_forest', 'isolation_forest_params.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先训练 Isolation Forest 模型")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"参数文件不存在: {params_path}\n请先训练 Isolation Forest 模型")
    
    print(f"\n[加载模型] {model_path}")
    model = joblib.load(model_path)
    
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    best_threshold = params['threshold']
    print(f"  阈值: {best_threshold:.4f}")
    
    print("[预测中...]")
    scores = model.decision_function(X_test)
    y_pred = (scores < best_threshold).astype(int)
    y_prob = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    
    return y_pred, y_prob, model_path


def test_autoencoder(X_test, y_test, base_dir):
    """测试 AutoEncoder 模型"""
    import pickle
    import numpy as np
    import torch
    import torch.nn as nn
    
    model_path = os.path.join(base_dir, 'models', 'autoencoder', 'model_autoencoder.pth')
    params_path = os.path.join(base_dir, 'models', 'autoencoder', 'autoencoder_params.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先训练 AutoEncoder 模型")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"参数文件不存在: {params_path}\n请先训练 AutoEncoder 模型")
    
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
    
    print(f"\n[加载模型] {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    print(f"  阈值: {best_threshold:.6f}")
    
    model = AutoEncoder(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("[预测中...]")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_errors = torch.mean((test_outputs - X_test_tensor) ** 2, dim=1).cpu().numpy()
    y_pred = (test_errors > best_threshold).astype(int)
    y_prob = (test_errors - test_errors.min()) / (test_errors.max() - test_errors.min() + 1e-10)
    
    return y_pred, y_prob, model_path


def test_dnn(X_test, y_test, base_dir):
    """测试 DNN 模型"""
    import torch
    import torch.nn as nn
    
    model_path = os.path.join(base_dir, 'models', 'dnn', 'model_dnn.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先训练 DNN 模型")
    
    # 定义 DNN 模型结构（必须与训练时一致）
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
    
    print(f"\n[加载模型] {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    model = DNN(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("[预测中...]")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_prob = model(X_test_tensor).cpu().numpy().flatten()
    y_pred = (y_prob > 0.5).astype(int)
    
    return y_pred, y_prob, model_path


def evaluate_and_save(model_name, y_test, y_pred, y_prob, base_dir, test_file):
    """计算评估指标并保存结果"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"\n{'='*60}")
    print(f"[{model_name.upper()}] 测试结果")
    print(f"{'='*60}")
    print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1-Score:           {f1:.4f}")
    print(f"  AUC:                {auc:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"    TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"    FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # 保存结果
    results = {
        'test_acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'test_file': test_file,
        'test_samples': len(y_test)
    }
    
    model_dir = os.path.join(base_dir, 'models', model_name)
    results_path = os.path.join(model_dir, f'results_{model_name}_external_test.csv')
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"\n  结果已保存: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='使用外部测试集评估模型')
    parser.add_argument('--model', type=str, default='all',
                        choices=['xgboost', 'dnn', 'isolation_forest', 'autoencoder', 'all'],
                        help='要测试的模型名称，默认 all')
    parser.add_argument('--test_file', type=str, default='train_test',
                        help='测试数据文件名（位于 Train 目录），默认 train_test')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'Train', args.test_file)
    
    print("="*60)
    print("KDD Cup 99 网络入侵检测 - 外部测试集评估")
    print("="*60)
    
    # 检查测试文件
    if not os.path.exists(test_path):
        print(f"\n[错误] 测试数据文件不存在: {test_path}")
        return
    
    # 加载预处理器
    print("\n[加载预处理器]")
    ohe, scaler = load_preprocessors(base_dir)
    
    # 预处理测试数据
    X_test, y_test, feature_cols = preprocess_test_data(test_path, base_dir, ohe, scaler)
    
    # 测试模型
    models_to_test = ['xgboost', 'dnn', 'isolation_forest', 'autoencoder'] if args.model == 'all' else [args.model]
    
    all_results = {}
    for model_name in models_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"[测试模型] {model_name.upper()}")
            print(f"{'='*60}")
            
            if model_name == 'xgboost':
                y_pred, y_prob, model_path = test_xgboost(X_test, y_test, base_dir)
            elif model_name == 'dnn':
                y_pred, y_prob, model_path = test_dnn(X_test, y_test, base_dir)
            elif model_name == 'isolation_forest':
                y_pred, y_prob, model_path = test_isolation_forest(X_test, y_test, base_dir)
            elif model_name == 'autoencoder':
                y_pred, y_prob, model_path = test_autoencoder(X_test, y_test, base_dir)
            
            results = evaluate_and_save(model_name, y_test, y_pred, y_prob, base_dir, args.test_file)
            all_results[model_name] = results
            
        except FileNotFoundError as e:
            print(f"\n[警告] {e}")
        except Exception as e:
            print(f"\n[错误] {model_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总对比
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("测试结果汇总对比")
        print(f"{'='*60}")
        print(f"\n{'模型':<18} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 72)
        for model_name, r in all_results.items():
            print(f"{model_name:<18} {r['test_acc']:.4f}    {r['precision']:.4f}    {r['recall']:.4f}    {r['f1']:.4f}    {r['auc']:.4f}")
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()