"""
EDA 图表生成脚本 (v2 — KDDTrain+)
输出1: 23类分布柱状图
输出2: 关键数值特征的多峰分布图 + 二值列堆叠图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

DATA_PATH = r'd:\NetWorkSecurity\code\NetWorkSecurity\data\KDDTrain+.txt'
OUTPUT_DIR = r'd:\NetWorkSecurity\code\NetWorkSecurity\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KDDTrain+ 标准 22 种攻击
ATTACKS_22 = [
    'back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap',
    'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune',
    'nmap', 'perl', 'phf', 'pod', 'portsweep',
    'rootkit', 'satan', 'smurf', 'spy', 'teardrop',
    'warezclient', 'warezmaster'
]

CATEGORY_MAP = {
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS','teardrop':'DoS',
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'guess_passwd':'R2L','warezmaster':'R2L','warezclient':'R2L','imap':'R2L',
    'ftp_write':'R2L','multihop':'R2L','phf':'R2L','spy':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
}

KEY_FEATURES = [
    'src_bytes', 'dst_bytes', 'count', 'srv_count', 'duration',
    'serror_rate', 'same_srv_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_serror_rate'
]
BINARY_FEATURES = ['logged_in', 'land', 'root_shell', 'is_guest_login', 'su_attempted', 'wrong_fragment']

cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]
df = pd.read_csv(DATA_PATH, header=None, names=cols + ['label','difficulty'])
print(f'数据加载完成: {df.shape}')

# ============================================================
# 输出1：23类柱状图
# ============================================================
print('\n===== 输出1 =====')
all_classes = ['normal'] + ATTACKS_22
counts = {c: len(df[df['label']==c]) for c in all_classes}
sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_items)

colors = []
for l in labels:
    if l == 'normal':
        colors.append('#2E86AB')
    else:
        cat = CATEGORY_MAP.get(l, 'Other')
        cmap = {'DoS':'#E74C3C','Probe':'#F39C12','R2L':'#27AE60','U2R':'#8E44AD'}
        colors.append(cmap.get(cat, '#95A5A6'))

fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_title('NSL-KDD (KDDTrain+): Class Distribution (23-class)', fontsize=14)
ax.grid(axis='x', alpha=0.3, linestyle='--')
for bar, v in zip(bars, values):
    ax.text(bar.get_width() + max(values)*0.005, bar.get_y() + bar.get_height()/2,
            str(v), va='center', fontsize=8)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='Normal'),
    Patch(facecolor='#E74C3C', label='DoS'),
    Patch(facecolor='#F39C12', label='Probe'),
    Patch(facecolor='#27AE60', label='R2L'),
    Patch(facecolor='#8E44AD', label='U2R'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  01_class_distribution.png saved')

# ============================================================
# 输出2：逐特征多峰分布图
# ============================================================
print('\n===== 输出2 =====')
sufficient = [c for c in all_classes if counts[c] >= 5]

for feat in KEY_FEATURES:
    fd = df[feat].dropna()
    if fd.nunique() <= 2:
        fig, ax = plt.subplots(figsize=(14, 7))
        box_data = []
        box_labels = []
        for c in sufficient:
            vals = df[df['label']==c][feat].dropna()
            if len(vals) >= 5:
                box_data.append(vals)
                box_labels.append(c)
        bp = ax.boxplot(box_data, labels=box_labels, vert=False, patch_artist=True)
        cmap = {'DoS':'#E74C3C','Probe':'#F39C12','R2L':'#27AE60','U2R':'#8E44AD','normal':'#2E86AB'}
        for patch, label in zip(bp['boxes'], box_labels):
            cat = 'normal' if label == 'normal' else CATEGORY_MAP.get(label, 'Other')
            patch.set_facecolor(cmap.get(cat, '#95A5A6'))
        ax.set_title(f'Feature: {feat} (Boxplot by Class)', fontsize=12)
        ax.set_xlabel(feat, fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    else:
        skewness = fd.skew()
        use_log = skewness > 5 and (fd > 0).all()
        fig, ax = plt.subplots(figsize=(14, 7))
        palette = sns.color_palette('husl', len(sufficient))
        for i, c in enumerate(sufficient):
            vals = df[df['label']==c][feat].dropna()
            if len(vals) < 5:
                continue
            data_plot = np.log1p(vals) if use_log else vals
            label_text = f'{c} (n={len(vals)})' if len(vals) < 500 else f'{c}'
            sns.kdeplot(data=data_plot, ax=ax, color=palette[i],
                       linewidth=1.2, alpha=0.7, label=label_text)
        xlabel = f'log({feat}+1)' if use_log else feat
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        title = f'Feature: {feat} (Distribution by Class)'
        title += ' [log scale]' if use_log else ''
        if skewness > 5:
            title += f' (skew={skewness:.1f})'
        ax.set_title(title, fontsize=12)
        ax.legend(loc='best', fontsize=6, ncol=2)
        ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_{feat}_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {feat} done')

# 二值列
print('\n===== 二值列 =====')
for feat in BINARY_FEATURES:
    if df[feat].nunique() <= 1:
        print(f'  跳过 {feat}: 常量')
        continue
    fig, ax = plt.subplots(figsize=(14, 7))
    proportions = []
    class_names = []
    for c in sufficient:
        sub = df[df['label']==c][feat]
        if len(sub) >= 5:
            p1 = sub.mean()
            proportions.append([1-p1, p1])
            class_names.append(c)
    proportions = np.array(proportions)
    x = np.arange(len(class_names))
    ax.bar(x, proportions[:,0], 0.6, label='0', color='#AED6F1')
    ax.bar(x, proportions[:,1], 0.6, bottom=proportions[:,0], label='1', color='#E74C3C')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_ylabel('Proportion')
    ax.set_title(f'Feature: {feat} (0/1 by Class)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_{feat}_binary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {feat} done')

print(f'\n全部完成 → {OUTPUT_DIR}')
