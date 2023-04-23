import sys
sys.path.append('../')
from network_trainer import CTOTrainer
from options import CTO_Options

def main():
    opt = CTO_Options(isTrain=True)
    opt.parse()
    opt.save_options()

    # set_loss() is not used here because in this paper we use more complex loss functions
    trainer = CTOTrainer(opt)
    trainer.set_GPU_device()
    trainer.set_logging()
    trainer.set_randomseed()
    trainer.set_network()
    trainer.set_optimizer()
    trainer.set_dataloader()
    trainer.run()

if __name__ == "__main__":
    main()