import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
    """加载预处理后的数据"""
    print("=" * 60)
    print("随机森林 模型训练")
    print("=" * 60)

    data_path = "Train/KDDTrain_preprocessed.csv"
    df = pd.read_csv(data_path)
    print(f"\n[数据加载] 数据集形状: {df.shape}")

    # 特征列：排除标签列和辅助列
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y_binary = df['label_binary'].values  # 二分类标签

    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")

    return X, y_binary, feature_cols


def split_data(X, y, test_size=0.3):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n[数据划分] 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    """训练随机森林模型"""
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    # 随机森林参数设置
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    print(f"\n随机森林参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 训练模型
    print("\n开始训练...")
    start_time = time.time()

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")

    # 交叉验证
    print("\n5折交叉验证...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return model, train_time


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """评估模型性能"""
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['normal', 'attack'],
                yticklabels=['normal', 'attack'])
    plt.title('随机森林 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 2. ROC曲线
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'随机森林 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_randomforest_evaluation.png', dpi=150, bbox_inches='tight')
    print("混淆矩阵和ROC曲线已保存: results_randomforest_evaluation.png")
    plt.close()

    # 3. 特征重要性
    plt.figure(figsize=(10, 8))
    importance = model.feature_importances_
    indices = np.argsort(importance)[-20:]  # 取Top 20

    plt.barh(range(len(indices)), importance[indices], align='center', color='green')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('随机森林 特征重要性排名 (Top 20)')
    plt.tight_layout()
    plt.savefig('results_randomforest_feature_importance.png', dpi=150, bbox_inches='tight')
    print("特征重要性图已保存: results_randomforest_feature_importance.png")
    plt.close()


def save_results(model, metrics, train_time):
    """保存模型和结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    # 保存模型
    joblib.dump(model, 'model_randomforest.pkl')
    print("模型已保存: model_randomforest.pkl")

    # 保存评估结果
    results_df = pd.DataFrame([metrics])
    results_df['train_time'] = train_time
    results_df.to_csv('results_randomforest_metrics.csv', index=False)
    print("评估指标已保存: results_randomforest_metrics.csv")


def main():
    """主函数"""
    # 1. 加载预处理数据
    X, y, feature_cols = load_preprocessed_data()

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. 训练随机森林模型
    model, train_time = train_random_forest(X_train, y_train, X_test, y_test)

    # 4. 评估模型
    metrics, y_pred, y_prob = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 5. 可视化结果
    plot_results(y_test, y_pred, y_prob, model, feature_cols)

    # 6. 保存结果
    save_results(model, metrics, train_time)

    print("\n" + "=" * 60)
    print("随机森林 模型训练完成!")
    print("=" * 60)

    return model, metrics


if __name__ == '__main__':
    model, metrics = main()