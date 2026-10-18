import torch.nn as nn
import torch
import torch.nn.functional as F


def paired_contrastive_loss(logits):
    """CC3M: 第 i 张图像与第 i 条 caption 配对，两个方向同等权重。"""
    targets = torch.arange(logits.shape[0], device=logits.device)
    logits = logits.float()
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2

def contrastive_loss(logits, label):
    # labels = torch.arange(logits.shape[0], device=logits.device) # 仅图文一对一
    positive_mask = label[:, None] == label[None, :]   # 更灵活的同类图片匹配 bool
    
    targets = positive_mask.float()     # 1/0
    targets = targets / targets.sum(dim=1, keepdim=True)    # 归一化: 0.3,0.3,0.01...
    
    loss_cross = nn.CrossEntropyLoss()

    loss_img = loss_cross(logits, targets)
    loss_text = loss_cross(logits.T, targets)
    loss = (loss_img + loss_text) / 2
    
    return loss
