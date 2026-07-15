import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    相比 CrossEntropyLoss(weight=alpha) 多了一个 (1-p_t)^gamma 因子：
    - 模型对某个样本预测置信度越高（p_t 越大），该样本的损失打折越多
    - 让模型更关注那些"一直分不对"的难样本

    Parameters:
        alpha: class weights, shape (num_classes,) or None
        gamma: focusing parameter, default 2.0 (from original paper)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
