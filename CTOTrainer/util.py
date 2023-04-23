import numpy as np
from skimage.morphology import dilation, disk
import torch

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