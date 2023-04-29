from skimage.morphology import dilation, disk
from MedISeg.unet2d.NetworkTrainer.utils.losses_imbalance import IOULoss,WCELoss
from MedISeg.unet2d.NetworkTrainer.utils.util import *
import torch
import torch.nn.functional as F


def get_gt_bnd(gt):
    # get ground truth boundary using dilation
    gt = (gt > 0).astype(np.uint8).copy()
    bnd = np.zeros_like(gt).astype(np.uint8)
    for i in range(gt.shape[0]):
        _mask = gt[i]
        for j in range(1, _mask.max()+1):
            _gt = (_mask == j).astype(np.uint8).copy()
            _gt_dil = dilation(_gt, disk(2))
            bnd[i][_gt_dil - _gt == 1] = 1
    return bnd


def poly_lr(optimizer, init_lr, curr_epoch, max_epoch, power=0.9):
    lr = init_lr * (1 - float(curr_epoch) / max_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def structure_loss(pred,mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask.float(), kernel_size=31, stride=1, padding=15) - mask)
    WIOU = IOULoss(smooth=1)
    WCE = WCELoss()
    iou_loss = WIOU(pred,mask,weit)
    ce_loss = WCE(pred,mask,weit)
    return (iou_loss+ce_loss)