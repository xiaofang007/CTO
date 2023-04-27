import sys
sys.path.append('../')
from CTOTrainer.network_inference import CTOInference
from CTOTrainer.options import CTO_Options


def main():
    opt = CTO_Options(isTrain=True)
    opt.parse()
    opt.save_options()

    inferencer = CTOInference(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.set_save_dir()
    inferencer.run()

if __name__ == "__main__":
    main()