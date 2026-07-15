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
