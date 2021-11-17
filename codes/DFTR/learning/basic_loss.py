import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    return bce.mean()


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


def dice_loss(pred, mask):
    iou = iou_loss(pred,mask)
    return 2*iou/(iou+1)


def logMSE_loss(dpred, depth):
    mse = nn.MSELoss()
    dpred = torch.sigmoid(dpred)
    dpred = 1.0 + dpred * 255.0
    depth = 1.0 + depth * 255.0
    dpred = 257.0 - dpred
    depth = 257.0 - depth
    return mse(torch.log(dpred), torch.log(depth))


def dec_loss(pred, mask, dpred, depth):
    dpred = torch.sigmoid(dpred)
    # deeper 255 -> deeper 1
    dpred = 256.0 - dpred * 255.0
    depth = 256.0 - depth * 255.0
    # Control the error window size by kernel_size
    # logDiff = torch.abs(torch.log(dpred) - torch.log(depth))
    logDiff = torch.abs(F.avg_pool2d(torch.log(dpred) - torch.log(depth), kernel_size=7, stride=1, padding=3))
    weit = logDiff / torch.max(logDiff)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()
