"""
输出3: 文字版数据统计表（按23类分组）
"""
import pandas as pd
import numpy as np

DATA_PATH = r'd:\NetWorkSecurity\code\NetWorkSecurity\data\KDDTrain+.txt'
OUTPUT_PATH = r'd:\NetWorkSecurity\code\NetWorkSecurity\output\03_data_statistics.md'

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

CLASS23 = ['normal',
    'neptune','satan','ipsweep','portsweep','smurf','nmap','back','teardrop','warezclient','pod',
    'guess_passwd','buffer_overflow','warezmaster','land','imap','rootkit','loadmodule',
    'ftp_write','multihop','phf','perl','spy'
]

# 对每列、每类计算统计量
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write('# 数据统计表 (KDDTrain+, 按23类分组)\n\n')
    f.write(f'总行数: {len(df)}, 总列数: {len(cols)}\n\n')
    
    for col in cols:
        f.write(f'---\n## {col}\n\n')
        
        if df[col].dtype in ['float64', 'int64', 'Int64']:
            vals = df[col].dropna()
            nunique = vals.nunique()
            is_binary = (nunique == 2 and set(vals.unique()) <= {0, 1})
            is_almost_constant = (vals == 0).mean() > 0.99
            
            if is_binary:
                # 二值列：只展示1的比例
                f.write(f'类型: 二值 | 全局1的比例: {vals.mean():.4f}\n\n')
                f.write('| 类别 | 样本量 | 1的比例 | 0的比例 |\n')
                f.write('|------|--------|---------|---------|\n')
                for c in CLASS23:
                    sub = df[df['label']==c][col].dropna()
                    p1 = sub.mean()
                    f.write(f'| {c} | {len(sub)} | {p1:.4f} | {1-p1:.4f} |\n')
            
            elif is_almost_constant:
                # 近常量
                zr = (vals == 0).mean()
                f.write(f'类型: 近常量 | 全局零值比例: {zr:.4f}\n\n')
                f.write('| 类别 | 样本量 | 均值 | 标准差 | 最大值 |\n')
                f.write('|------|--------|------|--------|--------|\n')
                for c in CLASS23:
                    sub = df[df['label']==c][col].dropna()
                    f.write(f'| {c} | {len(sub)} | {sub.mean():.4f} | {sub.std():.4f} | {sub.max()} |\n')
            
            elif nunique <= 20:
                # 离散计数
                f.write(f'类型: 离散计数 | 值域: {int(vals.min())}~{int(vals.max())}\n\n')
                f.write('| 类别 | 样本量 | 均值 | 标准差 | 零值比例 |\n')
                f.write('|------|--------|------|--------|----------|\n')
                for c in CLASS23:
                    sub = df[df['label']==c][col].dropna()
                    zr = (sub == 0).mean()
                    f.write(f'| {c} | {len(sub)} | {sub.mean():.4f} | {sub.std():.4f} | {zr:.4f} |\n')
            
            else:
                # 连续值：展示均值、方差、偏度、零值比例
                skew_global = vals.skew()
                f.write(f'类型: 连续 | 全局偏度: {skew_global:.2f}\n\n')
                f.write('| 类别 | 样本量 | 均值 | 标准差 | 偏度 | 零值比例 | 最小值 | 最大值 |\n')
                f.write('|------|--------|------|--------|------|----------|--------|--------|\n')
                for c in CLASS23:
                    sub = df[df['label']==c][col].dropna()
                    zr = (sub == 0).mean()
                    f.write(f'| {c} | {len(sub)} | {sub.mean():.2f} | {sub.std():.2f} | {sub.skew():.2f} | {zr:.4f} | {int(sub.min()) if sub.min()==int(sub.min()) else sub.min():.2f} | {int(sub.max()) if sub.max()==int(sub.max()) else sub.max():.2f} |\n')
        
        else:
            # 类别列
            f.write(f'类型: 类别 | 唯一值数: {df[col].nunique()}\n\n')
            f.write('| 类别 | 样本量 | 最常见值 | 最常见值占比 |\n')
            f.write('|------|--------|----------|--------------|\n')
            for c in CLASS23:
                sub = df[df['label']==c][col].dropna()
                if len(sub) > 0:
                    top = sub.value_counts().iloc[0]
                    top_val = sub.value_counts().index[0]
                    top_ratio = top / len(sub)
                    f.write(f'| {c} | {len(sub)} | {top_val} | {top_ratio:.4f} |\n')
                else:
                    f.write(f'| {c} | 0 | - | - |\n')
        
        f.write('\n')

print(f'输出3已保存: {OUTPUT_PATH}')
