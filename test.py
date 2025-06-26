"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from skimage.metrics import  peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import numpy as np
from skimage.color import deltaE_ciede2000
#from torchmetrics import StructuralSimilarityIndexMeasure
import cv2
import time

def CIEDE2000(im1, im2):


    im1_lab = np.transpose(im1, (1, 2, 0))  # Transpose to (256, 256, 3)
    im2_lab = np.transpose(im2, (1, 2, 0))


    # Convert the images to CIELAB color space
    im1_lab = cv2.cvtColor(im1_lab, cv2.COLOR_RGB2LAB).astype(np.float32)
    im2_lab = cv2.cvtColor(im2_lab, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Calculate color distance
    color_dist = deltaE_ciede2000(im1_lab, im2_lab).mean()

    return float(color_dist)

def calculate_metrics(outputs, targets, originals):
    target = targets.squeeze(0).cpu().detach().numpy()
    output = outputs.squeeze(0).cpu().detach().numpy()




    # PSNR(A, G(A))
    psnr = PSNR(target, output, data_range=1)

    # SSIM(A, G(A))
    ssim = SSIM(target, output, channel_axis=0, data_range=1)   # original  or  target,

    # CIEDE(G(A), B)
    ciede = CIEDE2000(output, target)

    # MSE (Mean Squared Error)
    mse = np.mean((output - target) ** 2)

    # L1 (Mean Absolute Error)
    l1 = np.mean(np.abs(output - target))

    return psnr, ssim, ciede, mse, l1

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    psnr = AverageMeter()
    ssim = AverageMeter()
    ciede = AverageMeter()
    MSE = AverageMeter()
    L1 = AverageMeter()
    inference_time = AverageMeter()
    flag = 0
    # SSIM = StructuralSimilarityIndexMeasure(data_range=255).to('cuda')

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        start_time = time.time()
        model.test()           # run inference
        elapsed_time = time.time() - start_time
        inference_time.update(elapsed_time)
        
        visuals = model.get_current_visuals()  # get image results

        fake_B = visuals['fake_B']
        real_B = visuals['real_B']
        real_A = visuals['real_A']
        fake_B = model.fake_B
        real_B = model.real_B
        real_A = model.real_A
        # ss = SSIM(real_B, fake_B)
        fake_B = fake_B.clamp(0, 255)
        real_B = real_B.clamp(0, 255)

        ps, ss, ci, mse, l1 = calculate_metrics(fake_B, real_B, real_A)
        psnr.update(ps)
        ssim.update(ss)
        ciede.update(ci)
        MSE.update(mse)
        L1.update(l1)

        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML
    print('PSNR=', psnr.avg, '   SSIM=', ssim.avg, '   CIEDE=', ciede.avg, '   MSE=', MSE.avg, '   L1=', L1.avg,'   Avg Inference Time (s)', inference_time.avg)
