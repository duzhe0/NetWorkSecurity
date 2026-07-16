#!/usr/bin/env python3
"""
混淆矩阵可视化（支持多模型对比和单模型独立展示）

用法:
    python visualize_confusion_matrices.py                              # 默认：XGBoost vs DNN 内部测试集
    python visualize_confusion_matrices.py --source external            # train_test 外部测试集
    python visualize_confusion_matrices.py --model two_stage --source external  # Two-Stage 单模型
"""

import os
import sys
import ast
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 数据加载（可复用）
# ============================================================

def load_all_class_names():
    """加载完整的 23 分类标签名列表"""
    class_file = os.path.join(BASE_DIR, 'Train', 'encoder_multiclass_23_classes.txt')
    with open(class_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def load_confusion_matrix(model_name, source='internal', threshold=None, cm_suffix=None):
    """
    从 CSV 加载并解析混淆矩阵。
    source='internal' → results_{model}_metrics.csv (KDDTrain 测试集)
    source='external' → results_{model}_external_test.csv (train_test 外部集)
    也支持从 .npy 文件加载（优先 CSV）。
    """
    th_suffix = f"_th{str(threshold).replace('.', '')}" if threshold else (cm_suffix or "")
    if source == 'internal':
        filename = f'results_{model_name}_metrics.csv'
    else:
        filename = f'results_{model_name}_external_test{th_suffix}.csv'

    csv_path = os.path.join(BASE_DIR, 'models', model_name, filename)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cm_str = df['confusion_matrix'].values[0]
        cm = np.array(ast.literal_eval(cm_str), dtype=int)
        acc = float(df['test_acc'].values[0])
        n_samples = int(df['test_samples'].values[0]) if 'test_samples' in df.columns else int(cm.sum())
        test_file = df['test_file'].values[0] if 'test_file' in df.columns else 'KDDTrain_preprocessed_test'
    else:
        # Fallback: .npy
        npy_name = f'cm_{source}{th_suffix}.npy'
        npy_path = os.path.join(BASE_DIR, 'models', model_name, npy_name)
        if os.path.exists(npy_path):
            cm = np.load(npy_path)
            acc = cm.diagonal().sum() / cm.sum()
            n_samples = int(cm.sum())
            test_file = f'{source}_test'
        else:
            raise FileNotFoundError(f"结果文件不存在: {csv_path} 或 {npy_path}")

    print(f"[{model_name}] {source} | 矩阵: {cm.shape} | 样本: {n_samples} | Acc: {acc:.4f}")
    return cm, acc, n_samples, test_file


def deduce_labels(cm, full_class_names):
    """
    根据混淆矩阵维度推断类标签。
    内部测试集：23×23，用完整列表。
    外部测试集：可能 < 23 类（某些攻击类型在外部数据中不存在），
    用矩阵行号映射回原始索引。
    """
    n = cm.shape[0]
    if n == len(full_class_names):
        # 完整 23 类
        return full_class_names
    else:
        # 部分类别：按行和降序推断哪些类别有样本
        row_sums = cm.sum(axis=1)
        non_empty = np.where(row_sums > 0)[0]
        # 保守处理：直接用序号标注，提示用户这类对应原始标签
        labels = []
        for i in range(n):
            label = full_class_names[i] if i < len(full_class_names) else f"class_{i}"
            if row_sums[i] == 0 and cm[:, i].sum() == 0:
                label += " ◇"    # 标记此类别完全缺失
            labels.append(label)
        return labels


# ============================================================
# 可视化（可复用核心）
# ============================================================

def plot_confusion_matrices(cm_xgb, cm_dnn, class_labels, source_label, output_name):
    """并排绘制两个模型的混淆矩阵热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(34, max(14, len(class_labels) * 0.65)))

    models = [
        ('XGBoost', cm_xgb, axes[0]),
        ('DNN',     cm_dnn,  axes[1]),
    ]

    vmax = max(cm_xgb.max(), cm_dnn.max())
    class_totals = cm_dnn.sum(axis=1)

    short_labels = [
        f"{i:02d} {name[:10]}\n({int(tot)})"
        for i, (name, tot) in enumerate(zip(class_labels, class_totals))
    ]

    for model_name, cm, ax in models:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    annot[i, j] = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"
                else:
                    annot[i, j] = ""

        sns.heatmap(
            cm, annot=annot, fmt='', cmap='YlOrRd',
            vmin=0, vmax=vmax,
            xticklabels=short_labels, yticklabels=short_labels,
            ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'label': '样本数', 'shrink': 0.8},
            annot_kws={'fontsize': 6}
        )

        acc = cm.diagonal().sum() / cm.sum()
        ax.set_title(f'{model_name}  ({cm.shape[0]}×{cm.shape[0]})\n'
                     f'准确率: {acc:.4f}  |  总样本: {int(cm.sum())}',
                     fontsize=13, fontweight='bold', pad=18)
        ax.set_xlabel('预测标签', fontsize=11)
        ax.set_ylabel('真实标签', fontsize=11)
        ax.tick_params(axis='both', labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.suptitle(f'XGBoost vs DNN 混淆矩阵对比 — {source_label}',
                 fontsize=17, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_name)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def plot_single_confusion_matrix(cm, class_labels, model_name, source_label, output_name):
    """绘制单个模型的混淆矩阵热力图"""
    n = cm.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(max(18, n * 0.75), max(10, n * 0.65)))

    class_totals = cm.sum(axis=1)
    short_labels = [
        f"{i:02d} {name[:12]}\n({int(tot)})"
        for i, (name, tot) in enumerate(zip(class_labels, class_totals))
    ]

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                annot[i, j] = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"
            else:
                annot[i, j] = ""

    sns.heatmap(
        cm, annot=annot, fmt='', cmap='YlOrRd',
        vmin=0,
        xticklabels=short_labels, yticklabels=short_labels,
        ax=ax, linewidths=0.5, linecolor='white',
        cbar_kws={'label': '样本数', 'shrink': 0.8},
        annot_kws={'fontsize': 6}
    )

    acc = cm.diagonal().sum() / cm.sum()
    ax.set_title(f'{model_name} 混淆矩阵 ({n}×{n})\n'
                 f'准确率: {acc:.4f}  |  总样本: {int(cm.sum())}',
                 fontsize=14, fontweight='bold', pad=18)
    ax.set_xlabel('预测标签', fontsize=11)
    ax.set_ylabel('真实标签', fontsize=11)
    ax.tick_params(axis='both', labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.suptitle(f'{model_name} 混淆矩阵 — {source_label}',
                 fontsize=17, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_name)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def print_misclass_analysis(cm_xgb, cm_dnn, class_labels):
    """打印误分类分析"""
    xgb_err = cm_xgb.sum(axis=1) - cm_xgb.diagonal()
    dnn_err = cm_dnn.sum(axis=1) - cm_dnn.diagonal()
    print(f"\n[误分类对比] XGBoost 总误分: {int(xgb_err.sum())}  |  DNN 总误分: {int(dnn_err.sum())}")

    worsened = []
    for i in range(len(class_labels)):
        if dnn_err[i] > xgb_err[i]:
            delta = int(dnn_err[i] - xgb_err[i])
            worsened.append((class_labels[i], int(xgb_err[i]), int(dnn_err[i]), delta))

    if worsened:
        print(f"\n  DNN 误分类恶化的类别 ({len(worsened)} 个):")
        for name, xe, de, d in sorted(worsened, key=lambda x: -x[3])[:10]:
            print(f"    {name:<16} XGBoost={xe:>4} → DNN={de:>4}  (恶化 +{d})")
    else:
        print("  所有类别 DNN 均不劣于 XGBoost")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='混淆矩阵可视化')
    parser.add_argument('--source', type=str, default='internal',
                        choices=['internal', 'external'],
                        help='internal = KDDTrain 测试集 | external = train_test 外部集')
    parser.add_argument('--model', type=str, default=None,
                        choices=['xgboost', 'dnn', 'cnn1d', 'transformer', 'two_stage'],
                        help='单模型模式（不指定则默认 XGBoost vs DNN 对比）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='阈值，如 0.6 或 0.7')
    parser.add_argument('--cm_suffix', type=str, default=None,
                        help='自定义 CM 文件后缀，如 _s107_s2075')
    args = parser.parse_args()

    source_label = 'KDDTrain_preprocessed_test (内部测试集 20%)' if args.source == 'internal' \
                   else 'train_test (外部独立测试集)'

    print(f"[数据源] {source_label}")
    print(f"[项目根] {BASE_DIR}")
    print("=" * 60)

    full_names = load_all_class_names()

    if args.model:
        # 单模型模式
        model_label = args.model.upper().replace('_', ' ')
        if args.cm_suffix:
            th_suffix = args.cm_suffix
        elif args.threshold:
            th_suffix = f"_th{str(args.threshold).replace('.', '')}"
        else:
            th_suffix = ""
        output_name = f'confusion_matrix_{args.model}_{args.source}{th_suffix}.png'

        cm, acc, n, _ = load_confusion_matrix(args.model, args.source, args.threshold, args.cm_suffix)
        class_labels = deduce_labels(cm, full_names)
        print(f"[类别] 共 {len(class_labels)} 个")

        out = plot_single_confusion_matrix(cm, class_labels, model_label, source_label, output_name)
        print(f"\n[保存] {out}")
    else:
        # 默认：XGBoost vs DNN 对比
        output_name = 'confusion_matrix_comparison.png' if args.source == 'internal' \
                      else 'confusion_matrix_comparison_external.png'

        cm_xgb, acc_xgb, n_xgb, f_xgb = load_confusion_matrix('xgboost', args.source)
        cm_dnn, acc_dnn, n_dnn, f_dnn = load_confusion_matrix('dnn', args.source)

        class_labels = deduce_labels(cm_dnn, full_names)
        print(f"[类别] 共 {len(class_labels)} 个: {class_labels}")

        print_misclass_analysis(cm_xgb, cm_dnn, class_labels)
        out = plot_confusion_matrices(cm_xgb, cm_dnn, class_labels, source_label, output_name)

        print(f"\n[保存] {out}")

    print("=" * 60)
    print("可视化完成!")


if __name__ == '__main__':
    main()
