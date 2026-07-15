import numpy as np
import torch
import torch.nn as nn


SERVICE_EMBEDDING_DIM = 10  # Embedding 输出维度


class ServiceEmbeddingModel(nn.Module):
    """
    包装任意基础模型，自动处理 service Embedding。

    用法:
        base = DNN(input_dim=66, num_classes=24)
        model = ServiceEmbeddingModel(base, vocab_size=69, feature_cols=feature_cols)
        # model 直接接受 125 维输入，内部自动做 Embedding 替换
    """

    def __init__(self, base_model, vocab_size=69, embedding_dim=SERVICE_EMBEDDING_DIM, feature_cols=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.base_model = base_model
        self.feature_cols = feature_cols

        # 预计算索引（避免 forward 中重复计算）
        if feature_cols is not None:
            try:
                self._service_idx = feature_cols.index('service_encoded')
            except ValueError:
                self._service_idx = None
            self._service_onehot_indices = [
                i for i, c in enumerate(feature_cols)
                if c.startswith('service_') and c != 'service_encoded'
            ]
            self._other_indices = [
                i for i in range(len(feature_cols))
                if i != self._service_idx and i not in self._service_onehot_indices
            ]
        else:
            self._service_idx = None
            self._service_onehot_indices = []
            self._other_indices = []

    def forward(self, x):
        if self._service_idx is None or self.feature_cols is None:
            return self.base_model(x)

        service_ids = x[:, self._service_idx].long()
        emb = self.embedding(service_ids)
        x_other = x[:, self._other_indices]
        return self.base_model(torch.cat([x_other, emb], dim=1))

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim


def compute_embedded_input_dim(feature_cols, embedding_dim=SERVICE_EMBEDDING_DIM):
    """
    计算输入特征嵌入后的实际维度。

    125 列 → 去掉 1 个 service_encoded + 68 个 service_* → 56 列 + embedding_dim
    """
    removed = 1  # service_encoded
    removed += sum(1 for c in feature_cols if c.startswith('service_') and c != 'service_encoded')
    return len(feature_cols) - removed + embedding_dim
