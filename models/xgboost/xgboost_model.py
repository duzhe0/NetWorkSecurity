import os
# macOS 需要设置 libomp 路径才能加载 XGBoost
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_preprocessed_data():
    """分别加载预处理后的训练集、验证集、测试集"""
    print("=" * 60)
    print("XGBoost 模型训练")
    print("=" * 60)

    data_dir = "../../Train/"

    df_train = pd.read_csv(os.path.join(data_dir, "KDDTrain_preprocessed_train.csv"))
    df_val   = pd.read_csv(os.path.join(data_dir, "KDDTrain_preprocessed_val.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "KDDTrain_preprocessed_test.csv"))

    print(f"\n[数据加载] 训练集形状: {df_train.shape}")
    print(f"[数据加载] 验证集形状: {df_val.shape}")
    print(f"[数据加载] 测试集形状: {df_test.shape}")

    # 特征列：排除标签列和辅助列
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train['label_binary'].values

    X_val = df_val[feature_cols].values
    y_val = df_val['label_binary'].values

    X_test = df_test[feature_cols].values
    y_test = df_test['label_binary'].values

    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集样本: {len(X_train)}, 验证集样本: {len(X_val)}, 测试集样本: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_xgboost(X_train, y_train, X_val, y_val):
    """训练XGBoost模型，使用验证集作为early stopping的eval_set"""
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    # XGBoost参数设置
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

    print(f"\nXGBoost参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 训练模型，eval_set使用验证集（不是测试集）
    print("\n开始训练...")
    start_time = time.time()

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")

    return model, train_time


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """评估模型性能（仅用测试集做最终评估）"""
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # 训练集评估
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n[训练集] 准确率: {train_acc:.4f}")

    # 测试集评估
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)

    print(f"\n[测试集] 评估结果:")
    print(f"  准确率 (Accuracy): {test_acc:.4f}")
    print(f"  精确率 (Precision): {test_precision:.4f}")
    print(f"  召回率 (Recall): {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n[混淆矩阵]")
    print(cm)

    # 分类报告
    print(f"\n[分类报告]")
    print(classification_report(y_test, y_test_pred, target_names=['normal', 'attack']))

    # 评估指标字典
    metrics = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
    }

    return metrics, y_test_pred, y_test_prob


def plot_results(y_test, y_pred, y_prob, model, feature_cols):
    """可视化结果"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    # 1. 混淆矩阵可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['normal', 'attack'],
                yticklabels=['normal', 'attack'])
    plt.title('XGBoost 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 2. ROC曲线
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_xgboost_evaluation.png', dpi=150, bbox_inches='tight')
    print("混淆矩阵和ROC曲线已保存: models/xgboost/results_xgboost_evaluation.png")
    plt.close()

    # 3. 特征重要性
    plt.figure(figsize=(10, 8))
    importance = model.feature_importances_
    indices = np.argsort(importance)[-20:]  # 取Top 20

    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('XGBoost 特征重要性排名 (Top 20)')
    plt.tight_layout()
    plt.savefig('results_xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
    print("特征重要性图已保存: models/xgboost/results_xgboost_feature_importance.png")
    plt.close()


def save_results(model, metrics, train_time):
    """保存模型和结果到当前目录（models/xgboost/）"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    # 保存模型
    joblib.dump(model, 'model_xgboost.pkl')
    print("模型已保存: models/xgboost/model_xgboost.pkl")

    # 保存评估结果
    results_df = pd.DataFrame([metrics])
    results_df['train_time'] = train_time
    results_df.to_csv('results_xgboost_metrics.csv', index=False)
    print("评估指标已保存: models/xgboost/results_xgboost_metrics.csv")


def main():
    """主函数"""
    # 1. 分别加载训练集、验证集、测试集
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_preprocessed_data()

    # 2. 训练XGBoost模型（eval_set使用验证集）
    model, train_time = train_xgboost(X_train, y_train, X_val, y_val)

    # 3. 用测试集做最终评估
    metrics, y_pred, y_prob = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 4. 可视化结果
    plot_results(y_test, y_pred, y_prob, model, feature_cols)

    # 5. 保存结果
    save_results(model, metrics, train_time)

    print("\n" + "=" * 60)
    print("XGBoost 模型训练完成!")
    print("=" * 60)

    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
