from .basic_loss import *


def SODLoss(mask, depth, pred, out2h, out3h, out4h, dpred, out2d, out3d, out4d, weights=(1,0.8,0.6,0.4), deep_supervision=False, bce_weight=1, use_iou=True, mask_deep_supervision=True):
    # sod loss
    region_loss = iou_loss if use_iou else dice_loss
    loss1b = bce_loss(pred, mask)
    loss1u = region_loss(pred, mask)
    loss2s_b = bce_loss(out2h, mask)
    loss2s_u = region_loss(out2h, mask)
    loss3s_b = bce_loss(out3h, mask)
    loss3s_u = region_loss(out3h, mask)
    loss4s_b = bce_loss(out4h, mask)
    loss4s_u = region_loss(out4h, mask)

    if not deep_supervision:
        out2d = out3d = out4d = dpred

    # depth correction loss
    loss1h = dec_loss(pred, mask, dpred, depth)
    loss2h = dec_loss(out2h, mask, out2d, depth)
    loss3h = dec_loss(out3h, mask, out3d, depth)
    loss4h = dec_loss(out4h, mask, out4d, depth)

    # depth loss
    loss1d = logMSE_loss(dpred, depth)
    loss2d = logMSE_loss(out2d, depth)
    loss3d = logMSE_loss(out3d, depth)
    loss4d = logMSE_loss(out4d, depth)

    w1,w2,w3,w4=weights
    ds_factor = 1 if deep_supervision else 0

    loss = bce_weight*loss1b + loss1u + loss1h + loss1d
    if mask_deep_supervision:
        loss =   w1 * (bce_weight*loss1b   + loss1u   + loss1h + loss1d) \
               + w2 * (bce_weight*loss2s_b + loss2s_u + loss2h + ds_factor*loss2d) \
               + w3 * (bce_weight*loss3s_b + loss3s_u + loss3h + ds_factor*loss3d) \
               + w4 * (bce_weight*loss4s_b + loss4s_u + loss4h + ds_factor*loss4d) 
    return loss, loss1b, loss1u, loss2d, loss1h
