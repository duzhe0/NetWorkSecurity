import os

# macOS 需要设置 libomp 路径才能加载 XGBoost
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============== 常量定义（与预处理模块一致）===============
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]


def load_preprocessed_data():
    """分别加载预处理后的训练集、验证集、测试集"""
    print("=" * 60)
    print("XGBoost 模型训练")
    print("=" * 60)

    data_dir = "../../Train/"

    df_train = pd.read_csv(os.path.join(data_dir, "KDDTrain_preprocessed_train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "KDDTrain_preprocessed_val.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "KDDTrain_preprocessed_test.csv"))

    print(f"\n[数据加载] 训练集形状: {df_train.shape}")
    print(f"[数据加载] 验证集形状: {df_val.shape}")
    print(f"[数据加载] 测试集形状: {df_test.shape}")

    # 特征列：排除所有标签列和辅助列（含新增的 23 分类标签列）
    exclude_cols = ['label', 'difficulty', 'label_binary',
                    'label_category', 'label_category_encoded',
                    'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train['label_multiclass_encoded'].values.astype(np.int64)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val['label_multiclass_encoded'].values.astype(np.int64)

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test['label_multiclass_encoded'].values.astype(np.int64)

    # 从文件获取真正的总类别数（总共 23 类）
    class_file = os.path.join(data_dir, "encoder_multiclass_23_classes.txt")
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        num_classes = len(class_names)
    else:
        # 兼容旧版本：从所有数据中取最大标签 + 1（确保覆盖所有可能出现的标签）
        max_label = max(df_train['label_multiclass_encoded'].max(),
                        df_val['label_multiclass_encoded'].max(),
                        df_test['label_multiclass_encoded'].max())
        num_classes = int(max_label + 1)

    print(f"\n【{num_classes} 分类任务】共 {num_classes} 个类别 (normal + {num_classes - 1} 种攻击)")
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集样本: {len(X_train)}, 验证集样本: {len(X_val)}, 测试集样本: {len(X_test)}")
    print(f"训练集实际类别数: {len(np.unique(y_train))}, 唯一值: {sorted(np.unique(y_train))}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes


def train_xgboost(X_train, y_train, X_val, y_val, num_classes, weight_scheme='none'):
    """训练XGBoost模型（多分类：multi:softprob），使用验证集作为early stopping的eval_set
    
    weight_scheme: 'none' | 'balanced' | 'sqrt' | 'log1p'
    """
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    # 计算类别平衡权重
    if weight_scheme == 'none':
        sample_weights_real = None
        print(f"\n[类别加权] 方案: none（无权重）")
    else:
        sample_weights_raw = compute_sample_weight('balanced', y_train)
        if weight_scheme == 'balanced':
            sample_weights_real = sample_weights_raw.copy()
        elif weight_scheme == 'sqrt':
            sample_weights_real = np.sqrt(sample_weights_raw)
        elif weight_scheme == 'log1p':
            sample_weights_real = np.log1p(sample_weights_raw)
        else:
            raise ValueError(f"未知的 weight_scheme: {weight_scheme}")
        print(f"\n[类别加权] 方案: {weight_scheme}")
        print(f"[类别加权] balanced 原始权重: {sample_weights_raw.min():.4f} ~ {sample_weights_raw.max():.4f}")
        print(f"[类别加权] 实际使用权重: {sample_weights_real.min():.4f} ~ {sample_weights_real.max():.4f}")

    # 【关键修复】检查训练集是否包含所有类别，缺失则补充虚拟样本（权重为0）
    unique_train = np.unique(y_train)
    if len(unique_train) < num_classes:
        missing_classes = set(range(num_classes)) - set(unique_train)
        print(f"训练集中缺少类别: {sorted(missing_classes)}，将添加虚拟样本（权重为0）以覆盖所有类别")

        # 创建虚拟样本（特征全0）
        num_missing = len(missing_classes)
        X_dummy = np.zeros((num_missing, X_train.shape[1]), dtype=np.float32)
        y_dummy = np.array(list(missing_classes), dtype=np.int64)

        # 合并数据
        X_train_aug = np.vstack([X_train, X_dummy])
        y_train_aug = np.concatenate([y_train, y_dummy])

        # 设置样本权重
        if sample_weights_real is not None:
            sample_weight = np.concatenate([
                sample_weights_real.astype(np.float32),
                np.zeros(num_missing, dtype=np.float32)
            ])
        else:
            sample_weight = None

        print(f"增强后训练集大小: {len(X_train_aug)}，其中虚拟样本 {num_missing} 个")
    else:
        X_train_aug = X_train
        y_train_aug = y_train
        sample_weight = sample_weights_real.astype(np.float32) if sample_weights_real is not None else None

    # XGBoost参数设置（多分类）
    params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'num_class': num_classes,  # 使用全局类别数
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    print(f"\nXGBoost参数（多分类）:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 训练模型，eval_set使用验证集（不是测试集）
    print("\n开始训练...")
    start_time = time.time()

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_aug, y_train_aug,
              sample_weight=sample_weight,
              eval_set=[(X_val, y_val)],
              verbose=False)

    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")

    # 注意：模型已经训练完毕，后续评估仍使用原始的 X_train, y_train（不含虚拟样本）
    return model, train_time


def evaluate_model(model, X_train, X_test, y_train, y_test, num_classes):
    """评估模型性能（多分类）"""
    print("\n" + "=" * 60)
    print("模型评估 - 多分类")
    print("=" * 60)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob_matrix = model.predict_proba(X_test)

    # 训练集评估
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n[训练集] 准确率: {train_acc:.4f}")

    # 测试集评估（多分类使用 weighted average）
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_test_prob_matrix, multi_class='ovr',
                                 average='weighted', labels=list(range(num_classes)))
    except Exception:
        test_auc = float('nan')

    print(f"\n[测试集] 评估结果 (多分类, {num_classes} 类):")
    print(f"  准确率 (Accuracy): {test_acc:.4f}")
    print(f"  精确率 (Precision weighted): {test_precision:.4f}")
    print(f"  召回率 (Recall weighted): {test_recall:.4f}")
    print(f"  F1-Score weighted: {test_f1:.4f}")
    print(f"  AUC (weighted OvR): {test_auc:.4f}")

    # 混淆矩阵（多分类 N×N）
    cm = confusion_matrix(y_test, y_test_pred, labels=list(range(num_classes)))
    print(f"\n[混淆矩阵] {num_classes}x{num_classes}")

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

    return metrics, y_test_pred, y_test_prob_matrix


def evaluate_external_test(model, num_classes, class_names):
    """在外部 train_test 集上评估模型（零信息泄露：仅用训练好的 encoder+scaler 做 transform）"""
    print("\n" + "=" * 60)
    print("外部评估 (train_test)")
    print("=" * 60)

    import pickle as _pickle
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 加载预处理工具
    ohe_path = os.path.join(base_dir, 'Train', 'encoder_onehot.pkl')
    scaler_path = os.path.join(base_dir, 'Train', 'scaler_robust.pkl')
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)

    # 加载 train_test 原始数据
    test_path = os.path.join(base_dir, 'Train', 'train_test')
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    print(f"[外部数据] train_test 原始: {df_test.shape}")

    # 生成标签：未知攻击 → 最后一类（unknown_attack = 23）
    if class_names:
        unknown_idx = len(class_names) - 1
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        df_test['label_multiclass_encoded'] = df_test['label'].map(label_to_idx).fillna(unknown_idx).astype(int)
    else:
        df_test['label_multiclass_encoded'] = 0

    y_test_orig = df_test['label_multiclass_encoded'].values

    # 从训练集获取特征列顺序
    train_csv_path = os.path.join(base_dir, 'Train', 'KDDTrain_preprocessed_train.csv')
    df_train_sample = pd.read_csv(train_csv_path, nrows=1)
    exclude_cols = ['label', 'difficulty', 'label_binary', 'label_category',
                    'label_category_encoded', 'label_multiclass', 'label_multiclass_encoded']
    feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]

    # One-Hot 编码（仅 transform）
    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_array = ohe.transform(df_test[CATEGORICAL_FEATURES])
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df_test.index)
    df_rest = df_test.drop(columns=CATEGORICAL_FEATURES)
    df_test_enc = pd.concat([df_rest, df_ohe], axis=1)

    # 补齐训练集有的列
    for col in feature_cols:
        if col not in df_test_enc.columns:
            df_test_enc[col] = 0.0
    df_test_enc = df_test_enc[feature_cols]

    # 标准化（仅 transform）
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_test_enc.columns]
    # Binary Indicator
    ZERO_INFLATED_COLS = ['src_bytes', 'dst_bytes', 'duration']
    BINARY_INDICATOR_COLS = []
    for col in ZERO_INFLATED_COLS:
        if col in df_test_enc.columns:
            indicator_name = f'is_zero_{col}'
            df_test_enc[indicator_name] = (df_test_enc[col] == 0).astype(int)
            BINARY_INDICATOR_COLS.append(indicator_name)
    # Log1p transform
    LOGP1_COLS = ['src_bytes', 'dst_bytes', 'duration']
    for col in LOGP1_COLS:
        if col in df_test_enc.columns:
            df_test_enc[col] = np.log1p(df_test_enc[col])
    numeric_cols.extend(BINARY_INDICATOR_COLS)
    df_test_enc[numeric_cols] = scaler.transform(df_test_enc[numeric_cols])

    X_test = df_test_enc[feature_cols].values.astype(np.float32)
    print(f"[外部数据] 预处理完成，特征矩阵: {X_test.shape}")

    # 获取 XGBoost 标签映射
    label_map_path = os.path.join(base_dir, 'models', 'xgboost', 'xgboost_label_map.pkl')
    if os.path.exists(label_map_path):
        with open(label_map_path, 'rb') as f:
            saved = _pickle.load(f)
        label_map = saved['label_map']
        print(f"[外部数据] 从 xgboost_label_map.pkl 加载标签映射: {len(label_map)} 类")
    else:
        # 模型以 24 类训练，标签索引 0..23 恒等映射
        label_map = {i: i for i in range(num_classes)}
        print(f"[外部数据] 使用恒等映射（num_class={num_classes}）")

    # 映射到模型空间（恒等映射，未知攻击已在上游映射为 23）
    y_test = np.array([label_map.get(int(v), 23) for v in y_test_orig], dtype=np.int64)
    # 安全检查：理论上不应再有 -1，但保留兜底
    valid_mask = y_test >= 0
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"[外部数据] 剔除 {n_dropped} 个非法标签样本（不应出现）")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    y_true_v = y_test[valid_mask]
    y_pred_v = y_pred[valid_mask]
    y_prob_v = y_prob[valid_mask]

    ext_acc = accuracy_score(y_true_v, y_pred_v)
    ext_f1 = f1_score(y_true_v, y_pred_v, average='weighted', zero_division=0)

    print(f"[外部评估] Acc: {ext_acc:.4f}, F1: {ext_f1:.4f}")

    ext_cm = confusion_matrix(y_true_v, y_pred_v, labels=list(range(num_classes)))
    return {'ext_acc': ext_acc, 'ext_f1': ext_f1}, ext_cm


def plot_results(y_test, y_pred, y_prob, model, feature_cols, num_classes):
    """可视化结果（多分类）"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)

    # 1. 混淆矩阵可视化（多分类 N×N）
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'XGBoost 混淆矩阵 ({num_classes} 类)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 2. 多分类 ROC（macro OvR，使用概率矩阵）
    plt.subplot(1, 2, 2)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=list(range(num_classes)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr',
                                  average='weighted', labels=list(range(num_classes)))
    except Exception:
        fpr, tpr = [0, 1], [0, 1]
        auc_score = float('nan')
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线（多分类 OvR）')
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
    """主函数：多方案对比 + 双轨评估"""
    # 1. 分别加载训练集、验证集、测试集
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, num_classes = load_preprocessed_data()

    # 加载类别名
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    class_file = os.path.join(base_dir, 'Train', 'encoder_multiclass_23_classes.txt')
    class_names = []
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]

    # 方案列表
    schemes = ['none', 'balanced', 'sqrt', 'log1p']

    all_results = []

    print("\n" + "=" * 70)
    print("XGBoost 多方案类别加权对比实验")
    print("=" * 70)

    for scheme in schemes:
        print("\n" + "#" * 70)
        print(f"# 方案: {scheme}")
        print("#" * 70)

        # 2. 训练
        model, train_time = train_xgboost(X_train, y_train, X_val, y_val, num_classes,
                                          weight_scheme=scheme)

        # 3. 内部测试集评估
        metrics, y_pred, y_prob = evaluate_model(model, X_train, X_test, y_train, y_test, num_classes)
        np.save(f'cm_internal_{scheme}.npy', metrics['confusion_matrix'])

        # 4. 外部 train_test 评估
        ext_metrics, ext_cm = evaluate_external_test(model, num_classes, class_names)
        np.save(f'cm_external_{scheme}.npy', ext_cm)

        # 5. 保存模型（按方案命名）
        joblib.dump(model, f'model_xgboost_{scheme}.pkl')
        print(f"[保存] model_xgboost_{scheme}.pkl")

        # 汇总记录
        result = {
            'scheme': scheme,
            'internal_acc': metrics['test_acc'],
            'internal_f1': metrics['f1'],
            'internal_auc': metrics['auc'],
            'external_acc': ext_metrics['ext_acc'],
            'external_f1': ext_metrics['ext_f1'],
            'train_time': train_time
        }
        all_results.append(result)

    # 打印汇总表
    print("\n" + "=" * 90)
    print("XGBoost 多方案双轨评估汇总")
    print("=" * 90)
    print(f"{'方案':<12} {'内部Acc':>10} {'内部F1':>10} {'内部AUC':>10} {'外部Acc':>10} {'外部F1':>10} {'耗时(s)':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['scheme']:<12} {r['internal_acc']:>10.4f} {r['internal_f1']:>10.4f} "
              f"{r['internal_auc']:>10.4f} {r['external_acc']:>10.4f} {r['external_f1']:>10.4f} "
              f"{r['train_time']:>8.1f}")

    # 保存汇总 CSV
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv('results_xgboost_schemes_summary.csv', index=False)
    print("\n[保存] 汇总表已保存: results_xgboost_schemes_summary.csv")

    print("\n" + "=" * 60)
    print("XGBoost 多方案对比完成!")
    print("=" * 60)

    return all_results


if __name__ == '__main__':
    all_results = main()