import os
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from MedISeg.unet2d.NetworkTrainer.network_inference import NetworkInference
from MedISeg.unet2d.NetworkTrainer.utils.accuracy import compute_metrics
from MedISeg.unet2d.NetworkTrainer.utils.post_process import *
from CTOTrainer.util import *
from network.CTO_net import CTO

class CTOInference(NetworkInference):
    def __init__(self, opt):
        super().__init__(opt)
    
    def set_network(self):
        self.net = CTO(self.opt.train['num_class'])
        self.net = torch.nn.DataParallel(self.net,device_ids=self.opt.train['gpus'])
        self.net = self.net.cuda()

        # ----- load trained model ----- #
        print(f"=> loading trained model in {self.opt.test['model_path']}")
        checkpoint = torch.load(self.opt.test['model_path'])
        state_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in state_dict}
        state_dict.update(pretrained_dict)
        self.net.load_state_dict(state_dict)

        # self.net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model at epoch {}".format(checkpoint['epoch']))
        self.net = self.net.module
        self.net.eval()
    
    def run(self):
        metric_names = ['p_recall', 'p_precision', 'p_F1', 'miou']
        all_result = AverageMeterArray(len(metric_names))
        for i, data in enumerate(tqdm(self.test_loader)):
            input, gt, name = data['image'].cuda(), data['label'], data['name']
            tta = TTA_2d(flip=self.opt.test['flip'], rotate=self.opt.test['rotate'])
            input_list = tta.img_list(input)
            y_list = []
            for x in input_list:
                x = torch.from_numpy(x.copy()).cuda()               
                y = self.net(x)[2]
                y = torch.nn.Softmax(dim=1)(y)[:, 1]
                y = y.cpu().detach().numpy()
                y_list.append(y)
            y_list = tta.img_list_inverse(y_list)
            output = np.mean(y_list, axis=0)
            pred = (output > 0.5).astype(np.uint8)

            for j in range(pred.shape[0]):
                pred[j] = self.post_process(pred[j])
                metrics = compute_metrics(pred[j], gt[j], metric_names)
                if metrics[metric_names[0]] == -1:
                    continue
                # print(f"{name[j]}: {metrics[2]}")
                all_result.update([metrics[metric_name] for metric_name in metric_names])
                if self.opt.test['save_flag']:
                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_pred.png'), (pred[j] * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_gt.png'), (gt[j].numpy() * 255).astype(np.uint8))
                    # np.save(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_prob.npy'), output[j])
                    img_save = input[j].cpu().numpy().transpose(1, 2, 0)
                    img_save = img_save * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
                    img_save = (img_save * 255).astype(np.uint8)

                    imageio.imwrite(os.path.join(self.opt.test['save_dir'], 'img', f'{name[j]}_img.png'), img_save)

        for i in range(len(metric_names)):
            print(f"{metric_names[i]}: {all_result.avg[i]:.4f}", end='\t')

        result_avg = [[all_result.avg[i]*100 for i in range(len(metric_names))]]
        result_avg = pd.DataFrame(result_avg, columns=metric_names)
        result_avg.to_csv(os.path.join(self.opt.test['save_dir'], 'test_results.csv'), index=False)
