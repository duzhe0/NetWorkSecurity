#!/usr/bin/env python3
"""
混淆矩阵分析报告生成器

用法:
    python generate_cm_report.py --model xgboost --source internal --version 1.3b
    python generate_cm_report.py --model dnn --source external --version 1.3b
    
输入: 从 metrics CSV 或 npy 文件读取混淆矩阵
输出: AIMemory/{version}_{model}_{source}_cm.md
"""

import os
import sys
import ast
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIMEMORY_DIR = os.path.join(BASE_DIR, 'AIMemory')


def load_class_names():
    """加载完整类别名列表（含 unknown_attack）"""
    path = os.path.join(BASE_DIR, 'Train', 'encoder_multiclass_23_classes.txt')
    if not os.path.exists(path):
        return [f'class_{i}' for i in range(24)]
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_confusion_matrix(model_name, source, scheme=None, threshold=None):
    """
    从模型目录加载混淆矩阵。
    优先读 scheme 指定的 .npy，fallback 读通用 .npy，再 fallback 读 CSV。
    """
    model_dir = os.path.join(BASE_DIR, 'models', model_name)
    th_suffix = f"_th{str(threshold).replace('.', '')}" if threshold else ""

    # 优先按 scheme 读 .npy
    if scheme:
        npy_name = f'cm_{source}_{scheme}.npy'
        npy_path = os.path.join(model_dir, npy_name)
        if os.path.exists(npy_path):
            cm = np.load(npy_path)
            labels = load_class_names()
            n = cm.shape[0]
            return cm, labels[:n]

    # threshold-specific .npy
    if threshold:
        npy_path = os.path.join(model_dir, f'cm_{source}{th_suffix}.npy')
        if os.path.exists(npy_path):
            cm = np.load(npy_path)
            labels = load_class_names()
            n = cm.shape[0]
            return cm, labels[:n]

    # 通用 .npy
    npy_name = f'cm_{source}.npy'
    npy_path = os.path.join(model_dir, npy_name)
    if os.path.exists(npy_path):
        cm = np.load(npy_path)
        labels = load_class_names()
        n = cm.shape[0]
        return cm, labels[:n]

    # Fallback: metrics CSV (with threshold suffix)
    if threshold:
        csv_path = os.path.join(model_dir, f'results_{model_name}_external_test{th_suffix}.csv')
    else:
        csv_name = f'results_{model_name}_metrics.csv' if source == 'internal' else f'results_{model_name}_external_test.csv'
        csv_path = os.path.join(model_dir, csv_name)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cm_str = df['confusion_matrix'].values[0]
        cm = np.array(ast.literal_eval(cm_str), dtype=int)
        labels = load_class_names()
        n = cm.shape[0]
        return cm, labels[:n]

    raise FileNotFoundError(f"混淆矩阵未找到: {npy_path} 或 {csv_path}")


def compute_per_class_metrics(cm, class_names):
    """计算每类的 precision / recall / f1"""
    n = cm.shape[0]
    rows = []
    for i in range(n):
        tp = cm[i, i]
        total_true = cm[i, :].sum()
        total_pred = cm[:, i].sum()
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            'class_idx': i,
            'class_name': class_names[i] if i < len(class_names) else f'class_{i}',
            'total_true': int(total_true),
            'total_pred': int(total_pred),
            'correct': int(tp),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    return rows


def find_major_misclass_flows(cm, class_names, min_samples=50):
    """找出主要误判流向（≥ min_samples 的 off-diagonal 样本）"""
    flows = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] >= min_samples:
                flows.append({
                    'true_class': class_names[i] if i < len(class_names) else f'class_{i}',
                    'pred_class': class_names[j] if j < len(class_names) else f'class_{j}',
                    'samples': int(cm[i, j])
                })
    flows.sort(key=lambda x: -x['samples'])
    return flows


def generate_report(model_name, source, version, cm, class_names):
    """生成 markdown 分析报告"""
    source_label = 'KDDTrain+ 内部测试集 (20% split)' if source == 'internal' else '外部独立测试集 (Train/train_test)'
    total = int(cm.sum())
    acc = cm.diagonal().sum() / total if total > 0 else 0
    total_errors = total - int(cm.diagonal().sum())

    per_class = compute_per_class_metrics(cm, class_names)

    n_visible = sum(1 for r in per_class if r['total_true'] > 0)
    missing = [r['class_name'] for r in per_class if r['total_true'] == 0]

    lines = []
    lines.append(f"# {model_name.upper()} 混淆矩阵分析报告")
    lines.append(f"")
    lines.append(f"> 实验版本: **{version}** | 数据源: **{source_label}** | 生成时间: 自动")
    lines.append(f"")
    lines.append("---")
    lines.append(f"")
    lines.append(f"## 一、整体概览")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 测试集样本数 | {total:,} |")
    lines.append(f"| 混淆矩阵形状 | {cm.shape[0]}×{cm.shape[1]} |")
    lines.append(f"| 准确率 | **{acc:.4f}** |")
    lines.append(f"| 总误分数 | {total_errors:,} |")
    lines.append(f"| 实际出现类别 | {n_visible} / {cm.shape[0]} |")
    if missing:
        lines.append(f"| 缺失类别 | {', '.join(missing)} |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 二、逐类详细指标")
    lines.append(f"")
    lines.append(f"| 类别 | 真实数 | 预测数 | 正确数 | 精确率 | 召回率 | F1 |")
    lines.append(f"|------|--------|--------|--------|--------|--------|-----|")

    for r in sorted(per_class, key=lambda x: -x['total_true']):
        name = r['class_name']
        if r['total_true'] == 0:
            lines.append(f"| {name} | **0** | {r['total_pred']} | 0 | 0.0000 | -- | 0.0000 |")
        else:
            p_bold = f"**{r['precision']:.4f}**" if r['precision'] >= 0.9 else f"{r['precision']:.4f}"
            r_bold = f"**{r['recall']:.4f}**" if r['recall'] >= 0.9 else f"{r['recall']:.4f}"
            f_bold = f"**{r['f1']:.4f}**" if r['f1'] >= 0.9 else f"{r['f1']:.4f}"
            lines.append(f"| {name} | {r['total_true']} | {r['total_pred']} | {r['correct']} | {p_bold} | {r_bold} | {f_bold} |")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # 误判流向
    flows = find_major_misclass_flows(cm, class_names)
    if flows:
        lines.append(f"## 三、主要误判流向（≥50 样本）")
        lines.append(f"")
        lines.append(f"| 真实类别 | 误判为 | 样本数 |")
        lines.append(f"|---------|--------|--------|")
        for f in flows[:15]:
            lines.append(f"| **{f['true_class']}** → {f['pred_class']} | {f['pred_class']} | **{f['samples']:,}** |")
    else:
        lines.append(f"## 三、主要误判流向")
        lines.append(f"")
        lines.append(f"无可报告的大规模误判（所有误判均 < 50 样本）。")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # 结论
    lines.append(f"## 四、结论")
    lines.append(f"")

    # 找最差 5 类
    worst = sorted([r for r in per_class if r['total_true'] > 0], key=lambda x: x['f1'])[:5]
    best = sorted([r for r in per_class if r['total_true'] > 0], key=lambda x: -x['f1'])[:3]

    lines.append(f"### 关键发现")
    lines.append(f"")
    lines.append(f"1. **准确率**: {acc:.4f}（{'内部' if source == 'internal' else '外部'}测试集）")
    lines.append(f"2. **最强类别** (F1 ≥ 0.90):")
    for r in best:
        if r['f1'] >= 0.9:
            lines.append(f"   - {r['class_name']}: F1={r['f1']:.4f}")
    if not best[0]['f1'] >= 0.9:
        lines.append(f"   - 无（所有类别 F1 < 0.90）")

    lines.append(f"3. **最弱类别**:")
    for r in worst:
        lines.append(f"   - {r['class_name']}: F1={r['f1']:.4f}, 召回率={r['recall']:.4f}")
    lines.append(f"4. **总误分**: {total_errors:,} / {total:,} 样本")

    lines.append(f"")
    if source == 'external':
        lines.append(f"> 注意：外部测试集含训练集未出现的攻击类型（已映射为 unknown_attack），")
        lines.append(f"> 模型对这些类别的识别能力是零样本泛化的核心考验。")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"> 关联文件: 混淆矩阵 numpy → `models/{model_name}/cm_{source}.npy` | 版本: {version}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='混淆矩阵分析报告生成器')
    parser.add_argument('--model', type=str, required=True,
                        choices=['xgboost', 'dnn', 'cnn1d', 'transformer', 'two_stage'],
                        help='模型名称')
    parser.add_argument('--source', type=str, required=True,
                        choices=['internal', 'external'],
                        help='internal=内部测试集 | external=外部测试集')
    parser.add_argument('--version', type=str, required=True,
                        help='实验版本号，如 1.3b')
    parser.add_argument('--scheme', type=str, default=None,
                        help='权重方案名，如 none/balanced/sqrt/log1p')
    parser.add_argument('--threshold', type=float, default=None,
                        help='阈值，如 0.6 或 0.7')
    args = parser.parse_args()

    th_str = f" (th={args.threshold})" if args.threshold else ""
    print(f"[生成报告] 模型={args.model}, 数据源={args.source}, 版本={args.version}{th_str}")
    print("=" * 60)

    cm, class_names = load_confusion_matrix(args.model, args.source, args.scheme, args.threshold)
    print(f"[混淆矩阵] 形状={cm.shape}, 总样本={int(cm.sum())}")

    report = generate_report(args.model, args.source, args.version, cm, class_names)

    os.makedirs(AIMEMORY_DIR, exist_ok=True)
    th_suffix = f"_th{str(args.threshold).replace('.', '')}" if args.threshold else ""
    filename = f"{args.version}_{args.model}_{args.source}_cm{th_suffix}.md"
    filepath = os.path.join(AIMEMORY_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[保存] {filepath}")
    print("=" * 60)
    print("报告生成完成!")


if __name__ == '__main__':
    main()
