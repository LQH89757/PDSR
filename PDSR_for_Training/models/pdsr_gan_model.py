import os
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp

from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from models.bnaf import Adam
from models.instance_whitening import cross_whitening_loss, get_covariance_matrix
import util.util as util


class PDSRGANModel(BaseModel):
    """Implementation of the PDSR GAN model."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
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
        parser.add_argument('--random_ratio', type=float, default=0.5, help='number of patches per layer')
        parser.add_argument('--two_F', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--lr_F', type=float, default=5e-5, help='initial learning rate for adam')

        parser.set_defaults(pool_size=0)
        opt, _ = parser.parse_known_args()
        dataset = os.path.basename(opt.dataroot.strip('/'))
        model_id = '%s_%s/var%s_np%s_nb%s_nl%s_nd%s_lr%s_ema%s' % (
            dataset,
            opt.direction,
            opt.lambda_var,
            opt.num_patches,
            opt.flow_blocks,
            opt.bnaf_layers,
            opt.bnaf_dim,
            opt.flow_lr,
            opt.flow_ema,
        )
        if opt.var_all:
            model_id += '_var_all'
        else:
            model_id += '_var_single'
        parser.set_defaults(name='%s/%s' % (opt.tag, model_id))
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G', 'G_GAN', 'D_real', 'D_fake', 'idt', 'var', 'nll_A', 'nll_B', 'exp_A', 'exp_B', 'SSL', 'BYOL']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B']
        self.var_layers = [int(i) for i in self.opt.var_layers.split(',')]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.use_amp = self.device.type == 'cuda'
        self.amp_device_type = 'cuda' if self.use_amp else 'cpu'

        if self.isTrain:
            self.model_names = ['G', 'F_A', 'F_B', 'D']
            self.opt_names = ['G', 'D', 'F']
        else:
            self.model_names = ['G', 'F_A', 'F_B']
            self.opt_names = ['F']

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
        )
        self.netF_A = networks.PatchDensityEstimator(opt, self.gpu_ids)
        self.netF_B = networks.PatchDensityEstimator(opt, self.gpu_ids)
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                self.gpu_ids,
                opt,
            )
            self.criterionNCE = []
            for _ in self.nce_layers[:len(self.nce_layers) // 2]:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSSL = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        bs_per_gpu = data['A'].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        self.compute_F_loss().backward()

        self.optimizer_F = Adam(
            chain(self.netF_A.parameters(), self.netF_B.parameters()),
            lr=self.opt.flow_lr,
            amsgrad=True,
            polyak=self.opt.flow_ema,
        )
        self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        with amp.autocast(device_type=self.amp_device_type, enabled=self.use_amp):
            self.forward()

        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netF_A, True)
        self.set_requires_grad(self.netF_B, True)
        self.optimizer_F.zero_grad()

        self.loss_F = self.compute_F_loss()
        self.loss_F.backward()
        torch.nn.utils.clip_grad_norm_(self.netF_A.parameters(), max_norm=0.1)
        torch.nn.utils.clip_grad_norm_(self.netF_B.parameters(), max_norm=0.1)
        self.optimizer_F.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        with amp.autocast(device_type=self.amp_device_type, enabled=False):
            self.loss_D = self.compute_D_loss()
        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()

        self.optimizer_F.swap()
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netF_A, False)
        self.set_requires_grad(self.netF_B, False)
        self.optimizer_G.zero_grad()

        with amp.autocast(device_type=self.amp_device_type, enabled=self.use_amp):
            self.loss_G = self.compute_G_loss()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.optimizer_F.swap()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def shuffle_patches(self, real_A, patch_num=16):
        ori_size = real_A.size(2)
        size = ori_size // patch_num

        patches = real_A.unfold(2, size, size).unfold(3, size, size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(1, patch_num * patch_num, 3, size, size)

        random_indices = torch.randperm(patch_num * patch_num)
        shuffled_patches = patches[:, random_indices]
        shuffled_patches = shuffled_patches.view(1, patch_num, patch_num, 3, size, size)
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        shuffled_patches = shuffled_patches.view(1, 3, ori_size, ori_size)
        return shuffled_patches, random_indices

    def restore_patches(self, shuffled_A, patch_indices, patch_num=16):
        ori_size = shuffled_A.size(2)
        size = ori_size // patch_num

        patches = shuffled_A.unfold(2, size, size).unfold(3, size, size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(1, patch_num * patch_num, 3, size, size)

        inverse_indices = torch.argsort(patch_indices)
        restored_patches = patches[:, inverse_indices]
        restored_patches = restored_patches.view(1, patch_num, patch_num, 3, size, size)
        restored_patches = restored_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        restored_patches = restored_patches.view(1, 3, ori_size, ori_size)
        return restored_patches

    def forward(self):
        self.fake_B, feat_list = self.netG(self.real_A, layers=self.var_layers, nce_layers=self.nce_layers)
        self.idt_B, self.patches_real_B = self.netG(self.real_B, layers=self.var_layers)
        self.feats = feat_list[-len(self.nce_layers):] if self.nce_layers else []
        self.patches_real_A = feat_list[:len(self.var_layers)]

        if self.opt.isTrain:
            real_A_s, s_idx = self.shuffle_patches(self.real_A)
            fake_B_s, _ = self.netG(real_A_s, layers=self.var_layers)
            self.fake_B_rs = self.restore_patches(fake_B_s, s_idx)

    def compute_D_loss(self):
        fake = self.fake_B.detach().float()
        real = self.real_B.float()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        self.pred_real = self.netD(real)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.cam_fake = F.interpolate(pred_fake.float(), size=fake.shape[2:], mode='bilinear')
        self.cam_real = F.interpolate(self.pred_real.float(), size=fake.shape[2:], mode='bilinear')
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        self.loss_idt = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_idt
        self.loss_var = self.calculate_var_loss()

        if self.opt.lambda_NCE > 0.0:
            cam = pred_fake.detach()
            self.loss_BYOL = self.calculate_BYOL_loss(self.feats, cam=cam) * self.opt.lambda_NCE
        else:
            self.loss_BYOL = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_idt + self.loss_BYOL
        self.loss_SSL = self.criterionSSL(self.fake_B_rs, self.fake_B)

        if self.opt.lambda_var > 0:
            self.loss_G += self.opt.lambda_var * self.loss_var

        self.loss_G += self.loss_SSL
        return self.loss_G

    def compute_F_loss(self):
        patches_real_A = [patch.float() for patch in self.patches_real_A]
        patches_real_B = [patch.float() for patch in self.patches_real_B]

        with amp.autocast(device_type=self.amp_device_type, enabled=False):
            log_probs_A, _, _ = self.netF_A(patches_real_A, self.opt.num_patches, None, detach=True)
            log_probs_B, _, _ = self.netF_B(patches_real_B, self.opt.num_patches, None, detach=True)

            self.loss_nll_A = 0.0
            self.loss_nll_B = 0.0
            for log_prob_a, log_prob_b in zip(log_probs_A, log_probs_B):
                self.loss_nll_A += -log_prob_a.float().mean()
                self.loss_nll_B += -log_prob_b.float().mean()

        self.loss_F = (self.loss_nll_A + self.loss_nll_B) / len(self.var_layers)
        return self.loss_F

    def calculate_BYOL_loss(self, feats, cam):
        n_layers = len(self.nce_layers)
        feat_k, feat_q = feats[:n_layers // 2], feats[n_layers // 2:][::-1]

        if self.opt.stop_gradient:
            feat_k = [x.detach() for x in feat_k]

        cams = []
        for feat in feat_q:
            cams.append(torch.nn.functional.interpolate(cam, size=feat.shape[2:], mode='bilinear'))

        prj = not self.opt.no_predictor
        feat_q_pool, sample_ids = self.netF_A(feat_q, self.opt.num_patches, None, cams=cams, prj=prj)
        feat_k_pool, _ = self.netF_A(feat_k, self.opt.num_patches, sample_ids, EnCo=True)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            k_cor, _ = get_covariance_matrix(f_k)
            q_cor, _ = get_covariance_matrix(f_q)
            loss = self.mse_loss(k_cor, q_cor)
            crosscov_loss = cross_whitening_loss(f_k, f_q)
            total_nce_loss += loss.mean()
            total_nce_loss += crosscov_loss.mean()

        return total_nce_loss / n_layers * 2

    def calculate_var_loss(self):
        n_layers = len(self.var_layers)
        patches_fake_B = self.netG(self.fake_B.float(), self.var_layers, encode_only=True)

        with torch.no_grad():
            log_probs_A, feat_lens, sample_ids = self.netF_A(
                [patch.float() for patch in self.patches_real_A],
                self.opt.num_patches,
                None,
                detach=True,
            )

        log_probs_fake_B, _, _ = self.netF_B(
            [patch.float() for patch in patches_fake_B],
            self.opt.num_patches,
            sample_ids,
        )

        total_var_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        nll_A = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        nll_B = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for log_prob_a, log_prob_b, feat_len in zip(log_probs_A, log_probs_fake_B, feat_lens):
            log_prob_a = log_prob_a.float()
            log_prob_b = log_prob_b.float()
            feat_len = feat_len.float()

            nll_A += -log_prob_a.mean()
            nll_B += -log_prob_b.mean()

            density_changes = (log_prob_a.detach() - log_prob_b).squeeze().float()
            density_changes_per_dim = density_changes / (feat_len.mean().item() * np.log(2))

            if self.opt.var_all:
                loss_layer = torch.var(density_changes_per_dim.float()).mean()
            else:
                density_changes_per_dim = density_changes_per_dim.view(self.opt.batch_size, int(log_prob_a.size(0)))
                loss_layer = torch.var(density_changes_per_dim.float(), dim=-1).mean()

            total_var_loss += loss_layer

        self.loss_exp_A = nll_A
        self.loss_exp_B = nll_B
        return total_var_loss / n_layers

    def mse_loss(self, feat_q, feat_k):
        if self.opt.stop_gradient:
            feat_k = feat_k.detach()
        return 2 - 2 * F.cosine_similarity(feat_q, feat_k, dim=-1)

    @torch.no_grad()
    def sample(self, x_A, x_B):
        if self.opt.direction != 'AtoB':
            x_A, x_B = x_B, x_A

        input_A, fake_B, input_B, idt_B = [], [], [], []
        for x_a, x_b in zip(x_A, x_B):
            x_a, x_b = x_a.unsqueeze(0), x_b.unsqueeze(0)
            fake_b = self.netG(x_a)
            idt_b = self.netG(x_b)
            input_A.append(x_a)
            input_B.append(x_b)
            fake_B.append(fake_b)
            idt_B.append(idt_b)
        return input_A, fake_B, input_B, idt_B
