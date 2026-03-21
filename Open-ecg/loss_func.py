import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    监督对比损失 (Stage 1使用)
    结合硬负样本：原始k类 + 硬负样本k类 = 2k类
    参考: loss_func(1).py
    """

    def __init__(self, alpha=0.6, temperature=0.1):
        super().__init__()
        self.alpha = alpha  # 对比损失权重
        self.temp = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def nt_xent_loss(self, anchor, labels):
        """Normalized Temperature-scaled Cross Entropy Loss"""
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            # 创建正样本mask：同类为1，异类为0（包括硬负样本）
            mask = torch.eq(labels, labels.transpose(0, 1)).float()
            # 去除对角线（自身）
            mask = mask - torch.eye(labels.size(0), device=labels.device)

        # 计算相似度
        anchor = F.normalize(anchor, dim=-1)
        sim_matrix = torch.matmul(anchor, anchor.transpose(0, 1)) / self.temp

        # 数值稳定性处理
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # 计算log概率
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 只计算正样本对的损失
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        pos_log_prob = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -1 * pos_log_prob.mean()
        return loss

    def forward(self, embeddings, logits, labels):
        """
        Args:
            embeddings: [B, D] 特征向量 (30维)
            logits: [B, 2*num_classes] 分类输出
            labels: [B] 类别标签 (LongTensor, 值为0到2k-1)
        """
        # 确保labels是1D LongTensor
        if labels.dim() > 1:
            labels = labels.squeeze()

        ce_loss = (1 - self.alpha) * self.ce_loss(logits, labels)
        cl_loss = self.alpha * self.nt_xent_loss(embeddings, labels)
        return ce_loss + cl_loss


class MultiHypersphereLoss(nn.Module):
    """
    多超球面学习损失 (Stage 2使用)
    与 train.py 中当前手写的 Stage 2 loss 完全一致：
    - 最小化样本到其类 prototype 的平方欧氏距离
    - 仅对当前 batch 中实际出现的类别求平均
    - 不再加入 prototype 正则项
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels, class_prototypes):
        """
        Args:
            embeddings: [B, D] 特征向量
            labels: [B] 类别标签 (0到k-1)
            class_prototypes: list of k tensors, 每个是 [D] 的原型向量
        """
        loss = 0.0
        valid_classes = 0

        for j in range(len(class_prototypes)):
            mask = (labels == j)
            if mask.sum() == 0:
                continue

            class_j_embs = embeddings[mask]
            prototype = class_prototypes[j]

            # 平方欧氏距离
            distances = torch.sum((class_j_embs - prototype.unsqueeze(0)) ** 2, dim=1)
            loss += distances.mean()
            valid_classes += 1

        if valid_classes == 0:
            return embeddings.new_tensor(0.0)

        return loss / valid_classes