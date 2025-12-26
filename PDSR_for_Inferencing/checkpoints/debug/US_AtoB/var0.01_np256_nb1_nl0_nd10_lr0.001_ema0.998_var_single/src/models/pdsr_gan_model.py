
from .base_model import BaseModel
from . import networks
import util.util as util
import os


from torch.cuda import amp

class PDSRGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_var', type=float, default=0.01, help='weight for variance loss')
        parser.add_argument('--lambda_idt', type=float, default=10.0, help='weight for identity loss')
        parser.add_argument('--var_layers', type=str, default='0,4,8,12,16', help='compute density loss on which layers')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flow_type', type=str, default='bnaf', help='flow type to estimate density')
        parser.add_argument('--maf_dim', type=int, default=1024, help='dimension of MAF')
        parser.add_argument('--maf_layers', type=int, default=2, help='number of layers in MAF')
        parser.add_argument('--maf_comps', type=int, default=10, help='number of components in MAF')
        parser.add_argument('--flow_blocks', type=int, default=1, help='number of blocks in flow model')
        parser.add_argument('--bnaf_layers', type=int, default=0, help='number of layers in BNAF')
        parser.add_argument('--bnaf_dim', type=int, default=10, help='dimension of BNAF')
        parser.add_argument('--flow_lr', type=float, default=1e-3, help='learning rate for flow')
        parser.add_argument('--flow_ema', type=float, default=0.998, help='exponential moving average rate for flow')
        parser.add_argument('--var_all', action='store_true', help='compute var on all images or single image')
        parser.add_argument('--tag', type=str, default='debug', help='tag to recognize the checkpoints')

        parser.add_argument('--nce_layers', type=str, default='3,7,13,18,24,28',
                            help='compute BYOL loss on paired features, e.g. (0,31), (3,28), (7,24), (13,18)')
        parser.add_argument('--lambda_NCE', type=float, default=0.1, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--stop_gradient', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--no_predictor', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--prj_norm', type=str, default='BN', help='train, val, test, etc')
        parser.add_argument('--oversample_ratio', type=int, default=4, help='number of patches per layer')
        parser.add_argument('--random_ratio', type=float, default=.5, help='number of patches per layer')
        parser.add_argument('--two_F', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--lr_F', type=float, default=5e-5, help='initial learning rate for adam')


        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        dataset = os.path.basename(opt.dataroot.strip('/'))
        model_id = '%s_%s/var%s_np%s_nb%s_nl%s_nd%s_lr%s_ema%s' % (dataset, opt.direction,
                    opt.lambda_var, opt.num_patches, opt.flow_blocks, opt.bnaf_layers,
                    opt.bnaf_dim, opt.flow_lr, opt.flow_ema)
        if opt.var_all:
            model_id += '_var_all'
        else:
            model_id += '_var_single'
        parser.set_defaults(name='%s/%s' % (opt.tag, model_id))

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B']
        self.var_layers = [int(i) for i in self.opt.var_layers.split(',')]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if self.isTrain:
            self.model_names = ['G', 'F_A', 'F_B', 'D']
            self.opt_names = ['G', 'D', 'F']
        else:  # during test time, only load G
            self.model_names = ['G']



        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        self.scaler = amp.GradScaler()




    def data_dependent_initialize(self, data):

        # print("data_dependent_initialize data_dependent_initialize data_dependent_initialize")
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)

    def optimize_parameters(self):
        # print("optimize_parameters optimize_parameters optimize_parameters")
        # forward
        with amp.autocast(device_type='cuda'):
            self.forward()



    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def forward(self):
        # print("forward forward forward forward")
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, feat_list = self.netG(self.real_A, layers=self.var_layers, nce_layers=self.nce_layers)
        self.idt_B, self.patches_real_B = self.netG(self.real_B, layers=self.var_layers)





