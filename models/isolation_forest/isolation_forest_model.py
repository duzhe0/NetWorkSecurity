#!/usr/bin/env python3
"""
Isolation Forest 异常检测模型
仅在正常流量上训练，偏离正常模式即为异常

特点：
- 仅使用正常流量训练（经典异常检测方法）
- 对未知攻击类型有较好的泛化能力
- 验证集上优化阈值，平衡准确率与召回率
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def load_data():
    """加载预处理后的训练集、验证集、测试集"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    val_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_val.csv')
    test_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据不存在: {train_path}\n请先运行数据预处理")

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


def train_isolation_forest(X_train, y_train, X_val, y_val):
    """
    训练 Isolation Forest 异常检测模型
    仅在正常流量上训练，使模型学习正常模式

    Returns:
        model: 训练好的模型
        best_threshold: 最优阈值
    """
    print("\n" + "="*60)
    print("训练 Isolation Forest 异常检测模型")
    print("="*60)

    # 仅使用正常流量训练
    X_train_normal = X_train[y_train == 0]
    print(f"\n仅使用正常流量训练: {len(X_train_normal)} 样本")

    print(f"\n模型参数:")
    print(f"  n_estimators: 500")
    print(f"  max_samples: 256")
    print(f"  contamination: 0.05 (设为较低值，由阈值优化决定)")
    print(f"  n_jobs: -1")

    model = IsolationForest(
        n_estimators=500,
        max_samples=256,
        contamination=0.05,
        n_jobs=-1,
        random_state=42
    )

    print("\n[训练中...]")
    model.fit(X_train_normal)

    # 使用训练集正常数据的分数分布设定阈值（P5分位更严格，减少误报）
    print("\n[阈值设定: 使用正常训练数据分数分布的 P5 分位...]")
    normal_scores = model.decision_function(X_train_normal)

    threshold_percentile = 5
    best_threshold = np.percentile(normal_scores, threshold_percentile)
    print(f"正常训练数据分数分布 P{threshold_percentile} 分位: {best_threshold:.4f}")

    # 验证集评估
    val_scores = model.decision_function(X_val)
    val_pred = (val_scores < best_threshold).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    val_prec = precision_score(y_val, val_pred, zero_division=0)
    val_rec = recall_score(y_val, val_pred, zero_division=0)
    print(f"验证集准确率: {val_acc:.4f}, 精确率: {val_prec:.4f}, 召回率: {val_rec:.4f}, F1: {val_f1:.4f}")

    # 也提供更激进的阈值选项供参考
    print("\n不同分位数阈值参考:")
    for p in [5, 10, 15, 20, 25]:
        thresh = np.percentile(normal_scores, p)
        val_pred_p = (val_scores < thresh).astype(int)
        acc_p = accuracy_score(y_val, val_pred_p)
        f1_p = f1_score(y_val, val_pred_p)
        print(f"  P{p}分位: 阈值={thresh:.4f}, 验证集Acc={acc_p:.4f}, F1={f1_p:.4f}")

    return model, best_threshold


def evaluate_model(model, X, y, threshold, dataset_name='测试集'):
    """评估模型效果"""
    scores = model.decision_function(X)
    y_pred = (scores < threshold).astype(int)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # 用异常分数的归一化作为概率估计
    y_prob = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
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


def plot_scores_distribution(model, X_train, y_train, X_test, y_test, threshold):
    """绘制分数分布"""
    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(train_scores[y_train == 0], label='Normal', kde=True, color='green', alpha=0.6)
    sns.histplot(train_scores[y_train == 1], label='Attack', kde=True, color='red', alpha=0.6)
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Train Score Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(test_scores[y_test == 0], label='Normal', kde=True, color='green', alpha=0.6)
    sns.histplot(test_scores[y_test == 1], label='Attack', kde=True, color='red', alpha=0.6)
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Test Score Distribution')
    plt.legend()

    plt.tight_layout()
    model_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(model_dir, 'results_isolation_forest_score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n分数分布图已保存: results_isolation_forest_score_distribution.png")


def main():
    print("="*60)
    print("KDD Cup 99 网络入侵检测 - Isolation Forest 异常检测")
    print("="*60)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data()

        model, best_threshold = train_isolation_forest(X_train, y_train, X_val, y_val)

        print("\n" + "="*60)
        print("评估结果")
        print("="*60)

        train_results = evaluate_model(model, X_train, y_train, best_threshold, '训练集')
        val_results = evaluate_model(model, X_val, y_val, best_threshold, '验证集')
        test_results = evaluate_model(model, X_test, y_test, best_threshold, '测试集')

        plot_scores_distribution(model, X_train, y_train, X_test, y_test, best_threshold)

        # 保存模型和结果
        model_dir = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(model, os.path.join(model_dir, 'model_isolation_forest.pkl'))
        joblib.dump({'threshold': best_threshold, 'feature_cols': feature_cols},
                    os.path.join(model_dir, 'isolation_forest_params.pkl'))

        # 保存 metrics（与 Web 应用一致的格式）
        metrics = {
            'test_acc': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1'],
            'auc': test_results['auc'],
            'confusion_matrix': str(test_results['confusion_matrix']),
            'train_time': 0
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, 'results_isolation_forest_metrics.csv'), index=False)

        print(f"\n模型已保存: {model_dir}/model_isolation_forest.pkl")
        print(f"参数已保存: {model_dir}/isolation_forest_params.pkl")
        print(f"评估结果已保存: {model_dir}/results_isolation_forest_metrics.csv")

        print("\n" + "="*60)
        print("Isolation Forest 训练完成!")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
    except Exception as e:
        print(f"\n[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
