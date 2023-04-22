from MedISeg.unet2d.NetworkTrainer.options import Options

class CTO_Options(Options):
    def __init__(self, isTrain):
        super().__init__(isTrain)
    
    def parse(self):
        parser = super().parse() # define shared options
        parser.add_argument('--num-class',type=int,default=2,help='total segmentation classes including background')
        parser.add_argument('--train-clip',type=float,default=0.5,help='gradient clipping margin')
        parser.add_argument('--train-fp32',required=False,default=False,action="store_true",help='disable mixed precision training and run old school fp32')
        args = parser.parse_args()
        self.train['clip'] =  args.train_clip
        self.train['fp16'] = not args.train_fp32
        self.train['num_class'] = args.num_class
        return parser


