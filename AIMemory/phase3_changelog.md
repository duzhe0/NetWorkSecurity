# Phase 3 改动记录

> Phase 1: 24类标签 + 类别加权多方案对比
> Phase 2: 零膨胀Binary Indicator + Log1p偏态压缩 + RobustScaler
> Phase 3: 扩展Binary Indicator + Focal Loss + Entity Embedding

---

## 改动 4.1: 扩展 Binary Indicator（更多零膨胀列）

- **状态**: ✅ 已完成
- **文件**: `data_preprocessing/data_preprocessing.py`
- **改后**: 对以下 5 列也加 Binary Indicator
  | 原始列 | 新列 |
  |--------|------|
  | `num_failed_logins` | `is_zero_num_failed_logins` |
  | `num_shells` | `is_zero_num_shells` |
  | `num_access_files` | `is_zero_num_access_files` |
  | `num_file_creations` | `is_zero_num_file_creations` |
  | `num_root` | `is_zero_num_root` |
- **结果**: ZERO_INFLATED_COLS 从 3 列扩展到 8 列，Binary Indicator 3→8
- **信息量**: num_failed_logins 99.9%为0, num_shells 100%, num_access_files 99.7%, num_file_creations 99.8%, num_root 99.5%

### 改动 4.2: hot 列 Log1p 变换

- **状态**: ✅ 已完成
- **文件**: `data_preprocessing/data_preprocessing.py`
- **改后**: hot 也做 log1p（原始 0~77 → 0~4.36）
- **结果**: LOGP1_COLS 从 3 列扩展到 4 列（加入 hot）

### 改动 4.3: Focal Loss 替代 CrossEntropyLoss（DNN/CNN1D/Transformer）

- **状态**: ✅ 已完成
- **涉及文件**: `models/losses.py`（新增）, `models/dnn/dnn_model.py`, `models/cnn1d/cnn1d_model.py`, `models/transformer/transformer_model.py`
- **实现**: `FocalLoss(alpha=class_weights, gamma=2.0)`
- **效果总结**: FocalLoss + balanced权重 表现极差（内部Acc 21-72%），需用 sqrt 或 none

### 改动 4.4: Entity Embedding 替代 service OneHot（DNN/CNN1D/Transformer）

- **状态**: ✅ 已完成
- **涉及文件**: `models/embedding_utils.py`（新增 ServiceEmbeddingModel 包装器）, 3个DL模型文件
- **实现**: `nn.Embedding(vocab_size=69, embedding_dim=10)`, 68维OneHot → 10维稠密向量
- **特征维度**: 125列 → Embedding后 66 维（125 - 1 service_encoded - 68 OneHot + 10 embedding）
- **OOV处理**: val=1样本, test=2样本映射到UNK索引

---

## 预处理输出

| 项目 | 值 |
|------|-----|
| 特征列数 | 125 |
| 数值型特征 | 34 |
| Binary Indicator | 8 |
| OneHot特征 | 82 |
| service_encoded | 1（整数编码，供 Embedding） |
| 训练集 | 88180 |
| 验证集 | 12598 |
| 测试集 | 25195 |

---

## Phase 3 四模型双轨评估结果

### DNN（FocalLoss + Entity Embedding）

| 方案 | 内部Acc | 内部F1 | 内部AUC | 外部Acc | 外部F1 | 耗时(s) |
|------|---------|--------|---------|---------|--------|---------|
| none | 0.9960 | 0.9957 | 1.0000 | 0.7072 | 0.6125 | 149.5 |
| balanced | 0.7167 | 0.7805 | 0.9818 | 0.4653 | 0.5076 | 149.4 |
| **sqrt** | **0.9799** | **0.9822** | **0.9998** | **0.7147** | **0.6351** | 148.3 |
| log1p | 0.8836 | 0.9089 | 0.9981 | 0.6762 | 0.6583 | 148.4 |

### CNN1D（FocalLoss + Entity Embedding）

| 方案 | 内部Acc | 内部F1 | 内部AUC | 外部Acc | 外部F1 | 耗时(s) |
|------|---------|--------|---------|---------|--------|---------|
| none | 0.9970 | 0.9968 | 1.0000 | 0.7217 | 0.6250 | 195.8 |
| balanced | 0.2422 | 0.2683 | 0.8675 | 0.1352 | 0.1268 | 193.1 |
| **sqrt** | **0.9546** | **0.9638** | **0.9997** | **0.6901** | **0.6202** | 193.6 |
| log1p | 0.8945 | 0.9171 | 0.9989 | 0.6845 | 0.6496 | 194.6 |

### Transformer（FocalLoss + Entity Embedding）

| 方案 | 内部Acc | 内部F1 | 内部AUC | 外部Acc | 外部F1 | 耗时(s) |
|------|---------|--------|---------|---------|--------|---------|
| none | 0.9950 | 0.9947 | 0.9999 | 0.7020 | 0.6151 | 317.6 |
| balanced | 0.2124 | 0.2482 | 0.6846 | 0.1430 | 0.1575 | 324.3 |
| **sqrt** | **0.9263** | **0.9423** | **0.9981** | **0.6926** | **0.6436** | 324.1 |
| log1p | 0.9147 | 0.9310 | 0.9991 | 0.6689 | 0.6308 | 323.6 |

### XGBoost（无FocalLoss/Embedding，仅4.1+4.2）

| 方案 | 内部Acc | 内部F1 | 内部AUC | 外部Acc | 外部F1 | 耗时(s) |
|------|---------|--------|---------|---------|--------|---------|
| none | 0.9986 | 0.9985 | 1.0000 | 0.7219 | 0.6194 | 8.2 |
| **balanced** | **0.9987** | **0.9987** | **1.0000** | **0.7325** | **0.6409** | 6.9 |
| sqrt | 0.9990 | 0.9989 | 1.0000 | 0.7217 | 0.6191 | 6.0 |
| log1p | 0.9988 | 0.9988 | 1.0000 | 0.7213 | 0.6186 | 6.1 |

---

## 关键发现

1. **XGBoost 统治**: 内部 99.86-99.90%，外部 72-73%，远优于 DL 模型。balanced 反而最好（外部 73.25%）
2. **FocalLoss效果**: DL模型中 'none' 方案内部最好（99.5-99.7%），但外部泛化不如 sqrt。'balanced' 方案在 FocalLoss 下完全崩盘（内部 21-72%）
3. **Entity Embedding 效果**: 特征从 125 维压缩到 66 维（-47%），DNN/CNN1D 精度保持高位（内部 99%+），说明 10 维 Embedding 有效替代了 68 维 OneHot
4. **最佳方案**: 
   - 内部评估: XGBoost sqrt（99.90%）或 DL none（99.5-99.7%）
   - 外部泛化: XGBoost balanced（73.25%）> CNN1D none（72.17%）> DNN sqrt（71.47%）
5. **训练效率**: XGBoost (~8s) 远快于 DNN (~150s), CNN1D (~195s), Transformer (~320s)
6. **推荐**: 生产环境用 XGBoost; 研究场景 DL 模型用 none 或 sqrt 加权


---

## Phase 4.1: 置信度阈值拒绝机制（解决 unknown_attack）

### 问题
外部测试 Acc 只有 70-73%，因为 56.8% 的错误来自训练集从未见过的 unknown_attack（3750样本）。
强制模型对没见过的东西分类没意义。

### 方案
- 对所有4个模型的 evaluate_external_test 函数加 `conf_threshold` 参数（默认0.5）
- `max_prob < threshold` → 标记为"未知"，不参与分类准确率计算
- 报告：全量Acc、拒绝数、高置信Acc

### 结果

| 模型 | 全量Acc | th=0.5 拒绝% | th=0.5 高置信Acc | th=0.7 高置信Acc | th=0.9 拒绝% | th=0.9 高置信Acc |
|------|:-----:|:--------:|:------------:|:------------:|:--------:|:------------:|
| DNN none | 70.72% | 3.1% | 72.77% | 76.49% | 15.8% | **80.74%** |
| CNN1D none | 72.17% | 1.1% | 72.91% | 77.20% | 15.9% | **82.01%** |
| Transformer none | 70.20% | 1.2% | 70.96% | 75.43% | 17.1% | **81.74%** |
| XGBoost none | 72.19% | 0.2% | 72.29% | 72.71% | 2.0% | 73.40% |

### 结论
- **DL模型**: threshold=0.9 时高置信 Acc **80-82%**（+10pp），但拒绝 16% 样本。性价比高
- **XGBoost**: 几乎不拒绝（0.2-2%），因为它对所有预测都过于自信。这个机制对它帮助不大
- **推荐**: DL 模型用 threshold=0.7-0.9，既保留大部分样本又能提升精度


---

## Phase 4.2: 噪声过采样稀有类（解决稀有类样本不足）

### 问题
训练集 13 个类别样本数 < 200（最少 spy=2, perl=3），导致 balanced 方案崩盘、外部稀有类 recall=0。

### 方案
- 在预处理中加 `oversample_rare_classes()` 函数
- 稀有类（<200样本）通过随机复制+高斯噪声（sigma=0.05）增强到 200 样本
- 数值列加噪声，Binary和OneHot列保持原值
- **仅对训练集**做增强，验证/测试集不动
- 训练集: 88180 → 90516 (+2336 条)

### 稀有类增强明细

| 类别 | 增强前 | 增强后 |
|------|:-----:|:-----:|
| spy | 2 | 200 |
| perl | 3 | 200 |
| phf | 3 | 200 |
| ftp_write | 5 | 200 |
| multihop | 5 | 200 |
| loadmodule | 6 | 200 |
| rootkit | 7 | 200 |
| imap | 8 | 200 |
| land | 12 | 200 |
| warezmaster | 14 | 200 |
| buffer_overflow | 21 | 200 |
| guess_passwd | 37 | 200 |
| pod | 141 | 200 |

### 效果：none 方案对比（SMOTE前 → 后）

| 模型 | 内部Acc | 外部Acc |
|------|:----:|:----:|
| DNN | 99.60% → 99.68% | 70.72% → 71.96% |
| CNN1D | 99.70% → 99.65% | 72.17% → 71.68% |
| Transformer | 99.50% → 99.76% | 70.20% → **73.30%** |
| XGBoost | 99.86% → 99.87% | 72.19% → 72.13% |

### 效果：balanced 方案对比（SMOTE前 → 后，改善巨大）

| 模型 | 内部Acc | 外部Acc |
|------|:----:|:----:|
| DNN | 71.67% → **88.76%** (+17pp) | 46.53% → 66.35% |
| CNN1D | 24.22% → **68.83%** (+44pp) | 13.52% → 45.30% |
| Transformer | 21.24% → **90.05%** (+69pp) | 14.30% → 68.50% |
| XGBoost | 99.87% → 99.88% | 73.25% → 72.40% |

### 结论
- **balanced方案救活了**：DL模型从崩盘（21-72%）恢复到可用水平（69-90%）
- **外部 Acc 改善有限**：因为外部测试的主导因素是 unknown_attack（3750样本，模型没见过），过采样对此无帮助
- **Transformer 受益最大**：balanced +69pp，none 外部也 +3.1pp
- **XGBoost 不受影响**：它本来就处理得好，过采样对它没变化
- **推荐组合**：SMOTE + 置信度拒绝（Phase 4.1+4.2）一起用，DL 模型能达到更好的高置信 Acc
