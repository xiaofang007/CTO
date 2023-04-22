import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from MedISeg.unet2d.NetworkTrainer.network_trainer import NetworkTrainer
from MedISeg.unet2d.NetworkTrainer.utils.losses_imbalance import IOUloss
from network.CTO_net import CTO

class CTOTrainer(NetworkTrainer):
    def __init__(self, opt):
        super().__init__(opt)
    
    def set_network(self):
        self.net = CTO(self.opt.train['num_class'])
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()
    
    def structure_loss(self,pred,mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask.float(), kernel_size=31, stride=1, padding=15) - mask)
        WIOU = IOUloss(smooth=1)
        iou_loss = WIOU(pred,mask,weit)