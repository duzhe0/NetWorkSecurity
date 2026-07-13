#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NSL-KDD 入侵检测系统：训练 + 测试集评估
复用现有模块（DataPreprocessor / IntrusionDetectionModel / ModelEvaluator），
在运行时扩展 attack_types 映射以覆盖 NSL-KDD 测试集中的新攻击类型。
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 屏蔽 TensorFlow 过多的 CPU 指令集告警
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 将源码目录加入路径
SRC_DIR = r'c:\Users\qq208\Desktop\网络安全综合课设'
sys.path.insert(0, SRC_DIR)

from data_preprocessing import DataPreprocessor
from model import IntrusionDetectionModel
from evaluation import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------
# 扩展 attack_types：覆盖 NSL-KDD 测试集全部攻击类型
# ------------------------------------------------------------
EXTENDED_ATTACK_TYPES = {
    'normal': 'normal',
    # DoS
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'procmon': 'dos',
    'udpstorm': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'worm': 'dos',
    # Probe
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    # R2L
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
    'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
    'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'httptunnel': 'r2l',
    'named': 'r2l', 'sendmail': 'r2l', 'snmpget': 'r2l', 'xsnoop': 'r2l',
    'xlock': 'r2l',
    # U2R
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}


def main():
    # 数据文件
    train_file = os.path.join(SRC_DIR, 'KDDTrain+.txt')
    test_file = os.path.join(SRC_DIR, 'KDDTest+.txt')

    if not os.path.exists(train_file):
        print(f"[ERROR] 训练集不存在: {train_file}")
        return
    if not os.path.exists(test_file):
        print(f"[ERROR] 测试集不存在: {test_file}")
        print("请先下载 NSL-KDD 测试集 KDDTest+.txt")
        return

    print("=" * 70)
    print("NSL-KDD 入侵检测系统 - 训练 + 测试集评估")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 数据预处理（训练集）
    # --------------------------------------------------------
    print("\n[1/5] 数据预处理 - 训练集 ...")
    preprocessor = DataPreprocessor()
    # 运行时扩展攻击类型映射
    preprocessor.attack_types = EXTENDED_ATTACK_TYPES

    X_train_full, y_train_full, df_train = preprocessor.preprocess(
        train_file,
        use_feature_selection=True,
        use_pca=False,
        n_features=30
    )
    label_mapping = preprocessor.get_label_mapping()
    print(f"  训练集样本数: {len(df_train)}")
    print(f"  特征数(特征选择后): {X_train_full.shape[1]}")
    print(f"  标签映射: {label_mapping}")

    # 类别分布
    unique, counts = np.unique(y_train_full, return_counts=True)
    print("  训练集类别分布:")
    for u, c in zip(unique, counts):
        print(f"    {label_mapping.get(int(u), u)}: {c}")

    # 划分训练/验证集 (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42,
        stratify=y_train_full
    )
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")

    # --------------------------------------------------------
    # 2. 构建深度学习模型
    # --------------------------------------------------------
    print("\n[2/5] 构建改进型 DNN 模型 ...")
    num_classes = len(np.unique(y_train_full))
    model = IntrusionDetectionModel(
        input_shape=(X_train.shape[1],),
        num_classes=num_classes,
        model_type='improved_dnn'
    )
    model.build_improved_dnn_model(dropout_rate=0.3)
    print(model.get_model_summary())

    # --------------------------------------------------------
    # 3. 训练模型
    # --------------------------------------------------------
    print("\n[3/5] 开始训练模型 ...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=256,
        use_class_weights=True
    )
    print("  训练完成！")

    # --------------------------------------------------------
    # 4. 在验证集上评估
    # --------------------------------------------------------
    print("\n[4/5] 验证集评估 (训练集 20% 内部划分) ...")
    evaluator = ModelEvaluator(model, label_mapping)
    val_results = evaluator.evaluate(X_val, y_val)

    print("\n  --- 验证集性能指标 ---")
    print(f"  准确率 (Accuracy):              {val_results['accuracy']:.4f}")
    print(f"  精确率 (Precision, Macro):      {val_results['precision_macro']:.4f}")
    print(f"  精确率 (Precision, Weighted):   {val_results['precision_weighted']:.4f}")
    print(f"  召回率 (Recall, Macro):         {val_results['recall_macro']:.4f}")
    print(f"  召回率 (Recall, Weighted):      {val_results['recall_weighted']:.4f}")
    print(f"  F1分数 (F1, Macro):             {val_results['f1_macro']:.4f}")
    print(f"  F1分数 (F1, Weighted):          {val_results['f1_weighted']:.4f}")

    # --------------------------------------------------------
    # 5. 在 NSL-KDD 测试集上评估
    # --------------------------------------------------------
    print("\n[5/5] NSL-KDD 测试集评估 (KDDTest+.txt) ...")
    # 使用训练好的 preprocessor 处理测试集（复用 scaler / feature_selector / label_encoder）
    df_test = preprocessor.load_data(test_file)
    df_test = preprocessor.handle_missing_values(df_test)
    df_test = preprocessor.map_attack_types(df_test)

    # 测试集可能含有训练集未见的攻击 -> 映射为 'other'，这里丢弃无法映射的样本
    # 但因为我们已扩展映射，绝大多数都会落到 5 类之一
    other_count = (df_test['main_category'] == 'other').sum()
    if other_count > 0:
        print(f"  [提示] 测试集中有 {other_count} 条样本攻击类型未在映射中，将被剔除。")
        unknown_attacks = df_test[df_test['main_category'] == 'other']['label'].unique()
        print(f"  未映射的攻击类型: {list(unknown_attacks)}")
        df_test = df_test[df_test['main_category'] != 'other'].reset_index(drop=True)

    # 编码分类变量（复用训练集的 encoder）
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_test_enc = df_test.copy()
    for col in categorical_cols:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            # 对未见的类别用第一个类别兜底，避免 transform 报错
            df_test_enc[col] = df_test_enc[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df_test_enc[col] = 0

    # 编码标签（复用训练集的 label encoder）
    le_label = preprocessor.label_encoders['main_category']
    y_test = df_test['main_category'].astype(str).map(
        lambda x: le_label.transform([x])[0] if x in le_label.classes_ else -1
    ).values
    valid_mask = y_test >= 0
    df_test_enc = df_test_enc[valid_mask].reset_index(drop=True)
    y_test = y_test[valid_mask]

    # 提取特征
    feature_cols = [c for c in preprocessor.column_names if c not in ['label', 'difficulty']]
    X_test_raw = df_test_enc[feature_cols]

    # 标准化（复用训练集 scaler）
    X_test_scaled = preprocessor.scaler.transform(X_test_raw)

    # 特征选择（复用训练集 selector）
    if preprocessor.feature_selector is not None:
        X_test_final = preprocessor.feature_selector.transform(X_test_scaled)
    else:
        X_test_final = X_test_scaled

    print(f"  测试集样本数: {len(y_test)}")
    unique_t, counts_t = np.unique(y_test, return_counts=True)
    print("  测试集类别分布:")
    for u, c in zip(unique_t, counts_t):
        print(f"    {label_mapping.get(int(u), u)}: {c}")

    # 预测与评估
    y_pred_test = model.predict_classes(X_test_final)
    y_pred_proba_test = model.predict(X_test_final)

    test_acc = accuracy_score(y_test, y_pred_test)
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, confusion_matrix
    )
    test_prec_macro = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_prec_w = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_rec_macro = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_rec_w = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_f1_w = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

    print("\n" + "=" * 70)
    print("  ★ NSL-KDD 测试集 (KDDTest+.txt) 性能指标 ★")
    print("=" * 70)
    print(f"  准确率 (Accuracy):              {test_acc:.4f}")
    print(f"  精确率 (Precision, Macro):      {test_prec_macro:.4f}")
    print(f"  精确率 (Precision, Weighted):   {test_prec_w:.4f}")
    print(f"  召回率 (Recall, Macro):         {test_rec_macro:.4f}")
    print(f"  召回率 (Recall, Weighted):      {test_rec_w:.4f}")
    print(f"  F1分数 (F1, Macro):             {test_f1_macro:.4f}")
    print(f"  F1分数 (F1, Weighted):          {test_f1_w:.4f}")

    # 各类别详细指标
    present_labels = sorted(np.unique(y_test))
    target_names = [label_mapping.get(int(i), str(i)) for i in present_labels]
    print("\n  --- 各类别详细分类报告 ---")
    cr_text = classification_report(
        y_test, y_pred_test,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    )
    print(cr_text)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_test, labels=present_labels)
    print("  --- 混淆矩阵 ---")
    header = "          " + "  ".join(f"{t:>8s}" for t in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"  {target_names[i]:>8s} {row_str}")

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 验证集图表
    evaluator.plot_confusion_matrix(
        y_val, val_results['y_pred'],
        save_path=os.path.join(output_dir, 'val_confusion_matrix.png')
    )
    evaluator.plot_training_history(
        history,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    evaluator.plot_roc_curves(
        y_val, val_results['y_pred_proba'],
        save_path=os.path.join(output_dir, 'val_roc_curves.png')
    )
    evaluator.plot_metrics_comparison(
        val_results,
        save_path=os.path.join(output_dir, 'val_metrics_comparison.png')
    )

    # 测试集图表
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 测试集混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('NSL-KDD 测试集 混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 测试集 ROC 曲线
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    n_classes = num_classes
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        if i in np.unique(y_test):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba_test[:, i])
            roc_auc = auc(fpr, tpr)
            lbl = label_mapping.get(int(i), f'类别 {i}')
            plt.plot(fpr, tpr, linewidth=2, label=f'{lbl} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title('NSL-KDD 测试集 ROC 曲线', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 测试集指标对比
    metrics_names = ['准确率', '精确率(Macro)', '召回率(Macro)', 'F1(Macro)']
    metrics_values = [test_acc, test_prec_macro, test_rec_macro, test_f1_macro]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values,
                   color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], alpha=0.7)
    plt.ylim([0, 1])
    plt.ylabel('数值', fontsize=12)
    plt.title('NSL-KDD 测试集 性能指标', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}',
                 ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存评估报告
    report_path = os.path.join(output_dir, 'nsl_kdd_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("NSL-KDD 入侵检测系统 - 评估报告\n")
        f.write("=" * 70 + "\n\n")
        f.write("【数据集信息】\n")
        f.write(f"训练集: KDDTrain+.txt ({len(df_train)} 样本)\n")
        f.write(f"测试集: KDDTest+.txt ({len(y_test)} 样本)\n")
        f.write(f"特征数: {X_train.shape[1]}\n")
        f.write(f"类别数: {num_classes}\n")
        f.write(f"标签映射: {label_mapping}\n\n")

        f.write("【验证集性能 (训练集 20% 内部划分)】\n")
        f.write(f"准确率 (Accuracy):              {val_results['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision, Macro):      {val_results['precision_macro']:.4f}\n")
        f.write(f"精确率 (Precision, Weighted):   {val_results['precision_weighted']:.4f}\n")
        f.write(f"召回率 (Recall, Macro):         {val_results['recall_macro']:.4f}\n")
        f.write(f"召回率 (Recall, Weighted):      {val_results['recall_weighted']:.4f}\n")
        f.write(f"F1分数 (F1, Macro):             {val_results['f1_macro']:.4f}\n")
        f.write(f"F1分数 (F1, Weighted):          {val_results['f1_weighted']:.4f}\n\n")

        f.write("【NSL-KDD 测试集性能 (KDDTest+.txt)】\n")
        f.write(f"准确率 (Accuracy):              {test_acc:.4f}\n")
        f.write(f"精确率 (Precision, Macro):      {test_prec_macro:.4f}\n")
        f.write(f"精确率 (Precision, Weighted):   {test_prec_w:.4f}\n")
        f.write(f"召回率 (Recall, Macro):         {test_rec_macro:.4f}\n")
        f.write(f"召回率 (Recall, Weighted):      {test_rec_w:.4f}\n")
        f.write(f"F1分数 (F1, Macro):             {test_f1_macro:.4f}\n")
        f.write(f"F1分数 (F1, Weighted):          {test_f1_w:.4f}\n\n")

        f.write("【各类别详细分类报告】\n")
        f.write(cr_text + "\n")

        f.write("【混淆矩阵】\n")
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>8d}" for v in row)
            f.write(f"  {target_names[i]:>8s} {row_str}\n")

    # 保存模型
    model_path = os.path.join(output_dir, 'nsl_kdd_model')
    model.save_model(model_path)

    print(f"\n结果已保存到: {output_dir}")
    print(f"  - val_confusion_matrix.png      (验证集混淆矩阵)")
    print(f"  - training_history.png          (训练历史)")
    print(f"  - val_roc_curves.png            (验证集ROC曲线)")
    print(f"  - val_metrics_comparison.png    (验证集指标对比)")
    print(f"  - test_confusion_matrix.png     (测试集混淆矩阵)")
    print(f"  - test_roc_curves.png           (测试集ROC曲线)")
    print(f"  - test_metrics_comparison.png   (测试集指标对比)")
    print(f"  - nsl_kdd_evaluation_report.txt (详细评估报告)")
    print(f"  - nsl_kdd_model.h5              (训练好的模型)")
    print("=" * 70)
    print("程序执行完成！")


if __name__ == '__main__':
    main()
