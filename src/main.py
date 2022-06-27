import torch
import argparse
import os
import random


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='pptbf_dataset3', help='dataset path')
parser.add_argument('--data', type=str, default='poisson', help='dataset path')
parser.add_argument('--test_data', type=str, default='../test_data/testset_wref', help='dataset path')  # testset_wref
parser.add_argument('--output_folder', type=str, default='results', help='dataset path')
parser.add_argument('--logs', type=str, default='_', help='training logs')
parser.add_argument('--scene_name', type=str, default='vshape', help='scene name for optimization logging folder')
parser.add_argument('--logs_opt', type=str, default='corr1w1_style03w008', help='folder for all test scene results')

parser.add_argument('--train', type=str2bool, default=False, help='train or optimize')
parser.add_argument('--use_emd', type=str2bool, default=False, help='use emd of chamfer')
parser.add_argument('--load_checkpoint', type=str, default='model_init', help='load trained model checkpoint name')
parser.add_argument('--loss_type', type=str, default='gabor', help='loss type for optimization')

parser.add_argument('--use_pix', type=str2bool, default=False, help='use pix loss')
parser.add_argument('--use_vgg', type=str2bool, default=True, help='use vgg19 for synthesis')
parser.add_argument('--analysis', type=str2bool, default=False, help='visualize features for different exemplars')
parser.add_argument('--resume_opt', type=str2bool, default=False, help='resume optimization, currently designed for soft optimization')

parser.add_argument('--upscaling_rate', type=int, default=2, help='upscaling ratio')
parser.add_argument('--num_points', type=int, default=256, help='number of points in the dataset')
parser.add_argument('--emb_dims', type=int, default=512, help='embedding dimensions of autoencoder')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
parser.add_argument('--task_id', type=int, default=95, help='slurm job array task id for seeding, 5, 95 seems okay')
parser.add_argument('--epochs', type=int, default=20, help='number of traininig epochs')
parser.add_argument('--iterations', type=int, default=500, help='number of iterations for optimization')
parser.add_argument('--log_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--val_num', type=int, default=4, help='validation number during training')

parser.add_argument('--grid_num', type=int, default=112, help='number of underlyding grid per axis')
parser.add_argument('--conv_kernel_size', type=int, default=7, help='kernel size for convolution')

parser.add_argument('--num_classes', type=int, default=1, help='number of maximum classes in the dataset')

parser.add_argument('--kernel_sigma1', type=float, default=1, help='largest kernel sigma for gabor filters')
parser.add_argument('--kernel_sigma2', type=float, default=4, help='smallest kernel sigma for gabor filters')

parser.add_argument('--edge_space', type=float, default=0.0, help='edge space for normalization_np')
parser.add_argument('--weight_corr', type=float, default=1, help='weight of deep correlation loss')
parser.add_argument('--weight_hist', type=float, default=0.0, help='weight of histogram loss')
parser.add_argument('--weight_slice', type=float, default=0.0, help='weight of sliced wasserstein loss')
parser.add_argument('--weight_fourier', type=float, default=0.0, help='weight of fourier spectrum loss')
parser.add_argument('--weight_tv', type=float, default=0.0, help='weight of tv loss')

parser.add_argument('--weight_style', type=float, default=0.08, help='weight of style loss')  # 0.1
parser.add_argument('--corr_layer', type=int, default=2, help='corr layer index')  # 0.1
parser.add_argument('--style_layer_start', type=int, default=0, help='start of style layer')  # 0.1
parser.add_argument('--style_layer_end', type=int, default=3, help='end of style layer')  # 0.1
parser.add_argument('--model_type', type=str, default='cnn', help='gabor cnn or standard cnn')  # 0.1
parser.add_argument('--blur_sigma', type=float, default=0.01, help='blur for features')  # 0.1
parser.add_argument('--window_size', type=float, default=0.25, help='window size for patch-based methods')  # 0.1
parser.add_argument('--optim_method', type=str, default='adam', help='choice of gradient-descent optimizer')  # 0.1


opt = parser.parse_args()

opt.ngpus = torch.cuda.device_count()
print('ngpus:', opt.ngpus)

opt.seed = opt.task_id

opt.output_folder = opt.output_folder + \
                    '/' + opt.logs + '_seed' + \
                    str(opt.seed) + '_conv{:}_style{:}_corr{:}'.format(opt.conv_kernel_size, opt.weight_style, opt.weight_corr)


if not os.path.exists(opt.output_folder):
    os.makedirs(opt.output_folder, exist_ok=True)


for sub_folder in ['models']:
    if not os.path.exists(opt.output_folder + '/' + sub_folder):
        os.makedirs(opt.output_folder + '/' + sub_folder, exist_ok=True)

if opt.train:
    opt_file = os.path.join(opt.output_folder, 'opt.txt')
    with open(opt_file, 'w') as f:
        for k, v in vars(opt).items():
            f.write('%s: %s\n' % (k, v))


def main():
    if opt.test_data == '../test_data/testset_disk':
        from optimizers.optimize_gabor_disk import Optimizer
        Optimizer(opt)
    # elif opt.test_data == '../test_data/testset_3d':
    #     from optimizers.optimize_gabor_3d import Optimizer
    #     Optimizer(opt)
    elif opt.test_data == '../test_data/testset_multiattributes':
        from optimizers.optimize_gabor_multiattributes import Optimizer
        Optimizer(opt)
    elif opt.test_data == '../test_data/testset_point':
        from optimizers.optimize_gabor_point import Optimizer
        Optimizer(opt)
    else:
        raise Exception('===> Wrong test_data ...')


if __name__ == '__main__':
    main()
