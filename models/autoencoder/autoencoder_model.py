#!/usr/bin/env python3
"""
AutoEncoder 异常检测模型
仅在正常流量上训练，重构误差大的即为异常
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


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


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    val_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_val.csv')
    test_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_test.csv')

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

    print(f"训练集: {len(X_train)} 样本 ({(y_train==0).sum()} normal, {(y_train==1).sum()} attack)")
    print(f"验证集: {len(X_val)} 样本 ({(y_val==0).sum()} normal, {(y_val==1).sum()} attack)")
    print(f"测试集: {len(X_test)} 样本 ({(y_test==0).sum()} normal, {(y_test==1).sum()} attack)")
    print(f"特征数量: {len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_autoencoder(X_train, y_train, X_val, y_val):
    X_train_normal = X_train[y_train == 0]
    print(f"\n仅使用正常流量训练: {len(X_train_normal)} 样本")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    input_dim = X_train_normal.shape[1]
    model = AutoEncoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    X_train_tensor = torch.FloatTensor(X_train_normal).to(device)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    X_val_tensor = torch.FloatTensor(X_val).to(device)

    print("\n[训练中...]")
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
            print(f"Epoch [{epoch+1}/100] - Train Loss: {total_loss/len(train_loader):.6f} - Val Loss: {val_loss:.6f}")

        if patience >= 10:
            print(f"早停: Epoch {epoch+1}")
            break

    # 计算重构误差作为异常分数
    print("\n[计算重构误差...]")
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_errors = torch.mean((val_outputs - X_val_tensor) ** 2, dim=1).cpu().numpy()

    # 使用正常数据分数分布设定阈值
    X_train_normal_tensor = torch.FloatTensor(X_train_normal).to(device)
    with torch.no_grad():
        train_normal_outputs = model(X_train_normal_tensor)
        normal_errors = torch.mean((train_normal_outputs - X_train_normal_tensor) ** 2, dim=1).cpu().numpy()

    threshold_percentile = 15
    best_threshold = np.percentile(normal_errors, threshold_percentile)
    print(f"正常训练数据误差分布 P{threshold_percentile} 分位: {best_threshold:.6f}")

    val_pred = (val_errors > best_threshold).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    print(f"验证集准确率: {val_acc:.4f}, F1: {val_f1:.4f}")

    return model, best_threshold, normal_errors


def evaluate_model(model, X, y, threshold, dataset_name='测试集'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.FloatTensor(X).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        errors = torch.mean((outputs - X_tensor) ** 2, dim=1).cpu().numpy()

    y_pred = (errors > threshold).astype(int)
    y_prob = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0

    cm = confusion_matrix(y, y_pred)

    print(f"\n{'-'*50}")
    print(f"[{dataset_name}] 评估结果")
    print(f"{'-'*50}")
    print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1-Score:           {f1:.4f}")
    print(f"  AUC:                {auc:.4f}")
    print(f"\n  混淆矩阵:")
    print(f"    TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"    FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'threshold': threshold
    }


def main():
    print("="*60)
    print("KDD Cup 99 网络入侵检测 - AutoEncoder 异常检测")
    print("="*60)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()

        model, best_threshold, _ = train_autoencoder(X_train, y_train, X_val, y_val)

        print("\n" + "="*60)
        print("评估结果")
        print("="*60)

        train_results = evaluate_model(model, X_train, y_train, best_threshold, '训练集')
        val_results = evaluate_model(model, X_val, y_val, best_threshold, '验证集')
        test_results = evaluate_model(model, X_test, y_test, best_threshold, '测试集')

        model_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_dir, 'model_autoencoder.pth'))
        joblib.dump({'threshold': best_threshold, 'input_dim': X_train.shape[1]},
                    os.path.join(model_dir, 'autoencoder_params.pkl'))

        metrics = {
            'test_acc': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1'],
            'auc': test_results['auc'],
            'confusion_matrix': str(test_results['confusion_matrix']),
            'train_time': 0
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, 'results_autoencoder_metrics.csv'), index=False)

        print(f"\n模型已保存: {model_dir}/model_autoencoder.pth")
        print(f"参数已保存: {model_dir}/autoencoder_params.pkl")
        print(f"评估结果已保存: {model_dir}/results_autoencoder_metrics.csv")

        print("\n" + "="*60)
        print("AutoEncoder 训练完成!")
        print("="*60)

    except Exception as e:
        print(f"\n[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
