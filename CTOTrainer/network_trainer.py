import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from MedISeg.unet2d.NetworkTrainer.utils.losses_imbalance import DiceLoss
from MedISeg.unet2d.NetworkTrainer.network_trainer import NetworkTrainer
from MedISeg.unet2d.NetworkTrainer.utils.util import AverageMeter
from network.CTO_net import CTO
from tqdm import tqdm
from util import*

class CTOTrainer(NetworkTrainer):
    def __init__(self, opt):
        super().__init__(opt)
    
    def set_network(self):
        self.net = CTO(self.opt.train['num_class'])
        self.net = torch.nn.DataParallel(self.net,device_ids=self.opt.train['gpus'])
        self.net = self.net.cuda()
    
    def train(self,scaler,dice_loss):
        self.net.train()
        loss_record3, loss_record2, loss_record1, loss_recorde = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for i_batch, sampled_batch in enumerate(self.train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            edges = torch.from_numpy(get_gt_bnd(label_batch.numpy())).cuda()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            if len(label_batch.shape) == 3:
                label_batch = label_batch.unsqueeze(1)
            with autocast():
                lateral_map_3, lateral_map_2, lateral_map_1, edge_map = self.net(volume_batch)
                loss3 = structure_loss(lateral_map_3, label_batch)
                loss2 = structure_loss(lateral_map_2, label_batch)
                loss1 = structure_loss(lateral_map_1, label_batch)
                losse = dice_loss(edge_map, edges)
                loss =  loss3 + loss2 + loss1 + 3*losse
            
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),self.opt.train['clip'])
            scaler.step(self.optimizer)
            scaler.update()

    
    def run(self):
        num_epoch = self.opt.train['train_epochs']
        scaler = GradScaler()
        dice_loss = DiceLoss()
        self.logger.info("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        self.logger.info("=> Batch size: {:d}".format(self.opt.train['batch_size']))
        self.logger.info("=> Number of training iterations: {:d} * {:d}".format(num_epoch, int(len(self.train_loader))))
        self.logger.info("=> Training epochs: {:d}".format(self.opt.train['train_epochs']))

        dataprocess = tqdm(range(self.opt.train['start_epoch'], num_epoch))
        best_val_loss = 100.0    
        print('=> start training!')
        for epoch in dataprocess:
            poly_lr(self.optimizer, self.opt.train['lr'], epoch, num_epoch)
            state = {'epoch': epoch + 1, 'state_dict': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
            train_loss = self.train(scaler,dice_loss)
            val_loss = self.val()
            self.logger_results.info('{:d}\t{:.4f}\t{:.4f}'.format(epoch+1, train_loss, val_loss))

            if val_loss<best_val_loss:
                best_val_loss = val_loss
                save_bestcheckpoint(state, self.opt.train['save_dir'])

                print(f'save best checkpoint at epoch {epoch}')
            if (epoch > self.opt.train['train_epochs'] / 2.) and (epoch % self.opt.train['checkpoint_freq'] == 0):
                save_checkpoint(state, epoch, self.opt.train['save_dir'], True)

        logging.info("training finished")
