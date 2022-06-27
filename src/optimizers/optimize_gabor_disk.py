import os
import shutil
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import utils
from dataloaders.poisson_disk import generate_possion_dis

from loss_fn import Slicing_torch
from models.pos_rad import PositionOptimizer_pro

from tqdm import tqdm

from utility.plot import plot_mc_point_np, plot_mc_disk_np, plot_img, plot_grid, plot_feature_as_point_cloud
from utility.knn import get_knearest_neighbors

from models.gabor_torch_disk import ContinuousGaborFilterBanks

from models.normal_cnn_disk import CNN, init_conv_weights

import random
from time import time

from utils import normalization_torch


class Optimizer:
    def __init__(self, opt):

        seed = opt.seed
        print('seed:', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.opt = opt

        self.output_folder = opt.output_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.upscaling_rate = opt.upscaling_rate
        self.num_kernel_sigmas = 4
        self.loss_type = opt.loss_type

        self.opt_time = defaultdict(list)

        self.use_step = False
        self.use_our_gram = False
        self.folder_name = opt.logs_opt + '/optimize_our_'

        self.folder_name += opt.loss_type + '_x' + str(opt.upscaling_rate)
        if not os.path.exists(self.output_folder + '/' + self.folder_name):
            os.makedirs(self.output_folder + '/' + self.folder_name, exist_ok=True)
        opt_file = os.path.join(self.output_folder + '/' + self.folder_name, 'opt.txt')
        with open(opt_file, 'w') as f:
            for k, v in vars(opt).items():
                f.write('%s: %s\n' % (k, v))
        self.all_test_output = self.output_folder + '/' + self.folder_name + '/all_test_output'

        self.folder_name += '/' + opt.scene_name

        if not os.path.exists(self.output_folder + '/' + self.folder_name):
            os.makedirs(self.output_folder + '/' + self.folder_name, exist_ok=True)
        if not os.path.exists(self.all_test_output):
            os.makedirs(self.all_test_output, exist_ok=True)

        test_disk = False
        if opt.test_data == '../test_data/testset_disk':
            test_disk = True
        if not test_disk:
            from dataloaders.dataloader_test_point import TestLoader
            test_dset_A = TestLoader(opt=opt, path=opt.test_data, scene_name=opt.scene_name, train_or_test='test')
            self.test_loader_A = torch.utils.data.DataLoader(test_dset_A, batch_size=1, shuffle=False, num_workers=0)
        else:
            from dataloaders.dataloader_test_disk import TestLoader
            test_dset_A = TestLoader(opt=opt, path=opt.test_data, scene_name=opt.scene_name, train_or_test='test')
            self.test_loader_A = torch.utils.data.DataLoader(test_dset_A, batch_size=1, shuffle=False, num_workers=0)

        self.opt_seq(opt)

        opt.run_time = 0
        for k, v in self.opt_time.items():
            # print(k, len(v), 100*4)
            opt.run_time += sum(v)
        opt.run_time /= 60.0
        opt_file = os.path.join(opt.output_folder + '/' + self.folder_name, 'statistics.txt')
        with open(opt_file, 'w') as f:
            for k, v in vars(opt).items():
                f.write('%s: %s\n' % (k, v))

    def init_all(self, opt, pts):

        ## copy subdirectory example
        for fromDirectory in ["models", 'dataloaders', 'preprocess', 'scripts']:
            toDirectory = self.output_folder + '/' + self.folder_name + '/code/' + fromDirectory
            if not os.path.exists(toDirectory):
                os.makedirs(toDirectory, exist_ok=True)
            # print(fromDirectory, toDirectory)
            # copy_tree(fromDirectory, toDirectory)
            shutil.copyfile('main.py', self.output_folder + '/' + self.folder_name + '/main.py')

        k = 30
        k = min(k, pts.shape[0])

        self.gabor_fb_inp = ContinuousGaborFilterBanks(device=self.device, res=opt.grid_num, upscaling_rate=1, k=k).to(self.device)
        self.gabor_fb_ref = ContinuousGaborFilterBanks(device=self.device, res=opt.grid_num, upscaling_rate=2, k=k).to(self.device)
        self.gabor_fb_out = ContinuousGaborFilterBanks(device=self.device, res=opt.grid_num, upscaling_rate=self.upscaling_rate, k=k).to(self.device)

        if opt.model_type == 'cnn':
            self.learned_filter = CNN(opt).to(self.device)
            self.learned_filter.apply(init_conv_weights)
        else:
            print('===> Warning: wrong model type, ended ...')
            return

        model_parameters = filter(lambda p: p.requires_grad, self.learned_filter.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('params:', params)

        if opt.load_checkpoint:
            if opt.load_checkpoint != 'model_init':
                print('===> Loaded Gabor CNN Checkpoint ...')
                self.learned_filter.load_state_dict(torch.load(self.output_folder + '/models/' + opt.load_checkpoint + '.pth', map_location=self.device))
            else:
                print('===> Saved Gabor CNN Checkpoint ...')
                torch.save(self.learned_filter.state_dict(), self.output_folder + '/models/model.pth')

        self.learned_filter.eval()

    def opt_seq(self, opt):

        epochs = opt.iterations
        exemplar = defaultdict(dict)
        num_classes = opt.num_classes

        inner_freq = 100

        for i, (x, c, r, x_l, c_l, r_l, _, x_name) in enumerate((self.test_loader_A)):

            x = x.to(self.device)[0]
            c = c.to(self.device)[0]
            r = r.to(self.device)[0]
            x_l = x_l.to(self.device)[0]
            c_l = c_l.to(self.device)[0]
            r_l = r_l.to(self.device)[0]

            num_classes = len(set(list(c.detach().cpu().numpy())))
            opt.num_classes = num_classes

            distances5, _ = get_knearest_neighbors(x, 2, 5)  #
            distances1, _ = get_knearest_neighbors(x, 2, 1)
            distances = distances1.reshape(-1)
            sorted_distances, _ = distances.sort()
            # sorted_distances = sorted_distances[sorted_distances > 0.05]  # filter noise from too close points
            sorted_distances = sorted_distances[-2:-1]
            num_distances = distances.numel()

            mean_distance = distances1.mean()

            print('mean_distance:', mean_distance, distances5.mean(), distances1.mean())
            kernel_sigma1 = (mean_distance / opt.kernel_sigma1).tolist()
            kernel_sigma2 = (mean_distance / opt.kernel_sigma2).tolist()

            img_res = torch.round(9 / mean_distance).tolist()

            print('img_res_computed:', img_res, opt.scene_name)
            img_res = min(256, img_res)
            img_res = max(128, img_res)
            print('img_res_final:', img_res, opt.scene_name, kernel_sigma1, kernel_sigma2)

            # print(opt.edge_space, (kernel_sigma1 + kernel_sigma2) / opt.edge_space)
            opt.grid_num = int(img_res)
            opt.num_samples = x.shape[0]
            # self.init_all(opt, x)

            x_name = x_name[0]
            folder_name = self.folder_name  # + '/' + x_name
            # print(x_name, folder_name)
            for sub_folder_name in ['target', 'output']:
                if not os.path.exists(self.output_folder + '/' + folder_name + '/' + sub_folder_name):
                    os.makedirs(self.output_folder + '/' + folder_name + '/' + sub_folder_name, exist_ok=True)

            # plot_mc_point_np(x.detach().cpu().numpy(), c.detach().cpu().numpy(), 1, self.all_test_output + '/{:}_exemplar'.format(x_name))
            # plot_mc_point_np(x_l.detach().cpu().numpy(), c_l.detach().cpu().numpy(), 1, self.all_test_output + '/{:}_ref'.format(x_name))

            plot_mc_disk_np(x.detach().cpu().numpy(), r.detach().cpu().numpy(), c.detach().cpu().numpy(), 2, self.all_test_output + '/{:}_exemplar_n{:}'.format(x_name, x.shape[0]),
                            ext='png')
            plot_mc_disk_np(x.detach().cpu().numpy(), r.detach().cpu().numpy(), c.detach().cpu().numpy(), 2, self.all_test_output + '/{:}_exemplar_n{:}'.format(x_name, x.shape[0]),
                            ext='pdf')
            # plot_mc_disk_np(x_l.detach().cpu().numpy(), r_l.detach().cpu().numpy(), c_l.detach().cpu().numpy(), 1, self.all_test_output + '/{:}_ref_n{:}'.format(x_name))

            pts_tar = np.concatenate([c.unsqueeze(1).detach().cpu().numpy(), x.detach().cpu().numpy()], 1)
            np.savetxt(self.all_test_output + '/{:}_exemplar_n{:}.txt'.format(x_name, x.shape[0]), pts_tar)

            init_pts = defaultdict(list)
            init_cls = defaultdict(list)
            init_rad = defaultdict(list)
            target_pts_stats = defaultdict(list)

            init_strategy = 1
            all_init = []
            all_init_cls = []
            all_init_rad = []
            for k in range(num_classes):
                ind = c == k
                num = c[ind].shape[0]
                tar_pts = x[ind]
                tar_rad = r[ind]
                # print(k, tar_pts.shape, tar_rad.shape)
                min_x = tar_pts[:, 0].min()
                max_x = tar_pts[:, 0].max()
                min_y = tar_pts[:, 1].min()
                max_y = tar_pts[:, 1].max()
                min_r = tar_rad.min()
                max_r = tar_rad.max()

                margin_x = min(min_x, 1 - max_x) / self.upscaling_rate
                margin_y = min(min_y, 1 - max_y) / self.upscaling_rate
                target_pts_stats[k] = [margin_x, margin_y, min_r / self.upscaling_rate, max_r / self.upscaling_rate]

                margin = min([margin_x, margin_y]).item()

                # print('margin:', margin, min_y, max_y, min_r, max_r)

                if init_strategy == 0:
                    print('wrong init strategy')
                    return
                if init_strategy == 1:
                    init_data_filename = '../test_data/init/init_{:}_c{:}_x{:}.npz'.format(x_name, k, self.upscaling_rate)
                    if os.path.exists(init_data_filename):
                        print('loading init data ...')
                        p = np.load(init_data_filename)['x']
                        # p = np.random.rand(p.shape[0], 2)
                        # p = generate_possion_dis(num * (self.upscaling_rate ** 2), 0.01, 0.99)
                    else:
                        print('generate init data and save ...')
                        p = generate_possion_dis(num * (self.upscaling_rate ** 2), 0.01, 0.99)
                        np.savez('../test_data/init/init_{:}_c{:}_x{:}'.format(x_name, k, self.upscaling_rate), x=p)
                    init_pts[k] = p
                    init_cls[k] = torch.ones(num * (self.upscaling_rate ** 2), dtype=torch.int32).to(self.device) * k
                    # init_rad[k] = (max_r - min_r) * torch.rand(num * (self.upscaling_rate ** 2), dtype=torch.float32).to(self.device) + min_r
                    init_rad[k] = tar_rad.repeat(self.upscaling_rate ** 2) / self.upscaling_rate
                    # print(init_rad[k].shape, tar_rad.shape, tar_rad.repeat(self.upscaling_rate**2).shape)
                    # print(init_rad[k].shape, init_cls[k].shape, init_pts[k].shape)
                    all_init.append(p)
                    all_init_cls.append(init_cls[k].detach().cpu().numpy())
                    all_init_rad.append(init_rad[k].detach().cpu().numpy())

            all_init = np.concatenate(all_init, 0)
            all_init_cls = np.concatenate(all_init_cls, 0)
            all_init_rad = np.concatenate(all_init_rad, 0)

            plot_mc_disk_np(all_init, all_init_rad, all_init_cls, 1, self.all_test_output + '/{:}_init_n{:}'.format(x_name, all_init.shape[0]), ext='.pdf')
            plot_mc_disk_np(all_init, all_init_rad, all_init_cls, 1, self.all_test_output + '/{:}_init_n{:}'.format(x_name, all_init.shape[0]), ext='png')

            init_input_pts = init_pts.copy()
            # print('init_input_pts:', init_input_pts[0].shape)
            # init_guess = torch.from_numpy(init_input_pts[0].copy()).to(self.device).float()
            pts_txt_save = []
            for k in range(num_classes):
                cls_k = init_cls[k]
                pts_k = init_pts[k]
                rad_k = init_rad[k]
                # print(k, cls_k.shape, pts_k.shape, rad_k.shape)
                all = np.concatenate([cls_k.unsqueeze(1).detach().cpu().numpy(), pts_k, rad_k.unsqueeze(1).detach().cpu().numpy()], 1)
                # print(all.shape)
                # plot_mc_disk_np(pts_k, rad_k.unsqueeze(1).detach().cpu().numpy(), cls_k.detach().cpu().numpy(), 1,
                #                 self.output_folder + '/' + folder_name + '/output/pts_init_cls{:}'.format(k))

                pts_txt_save.append(all)
            pts_txt_save = np.concatenate(pts_txt_save, 0)
            # print(pts_txt_save.shape)
            np.savetxt(self.output_folder + '/' + folder_name + '/output/{:}_init_n{:}.txt'.format(opt.scene_name, pts_txt_save.shape[0]), pts_txt_save)

            losses = []
            losses_wasserstein_layers = []
            exemplar['data'] = x
            exemplar['cls'] = c

            for k in range(num_classes):
                # ratio = np.sqrt(8192 / x.shape[0])
                kernel_sigmas = np.linspace(kernel_sigma1, kernel_sigma2, self.num_kernel_sigmas)  # vshape, [0.045, 0.03], [0.1, 0.03], [0.06, 0.03]
                print('kernel_sigmas:', kernel_sigmas)

                exist_cls = [kk for kk in range(k)]
                cur_cls = [k]
                print('exist_cls:', exist_cls, cur_cls)

                pos_opt = PositionOptimizer_pro(init_input_pts, init_rad, exist_cls, cur_cls).to(self.device)
                # lr_list = [0.02, 0.01, 0.01, 0.01]
                # lr_list = [0.01, 0.005, 0.002, 0.002]

                x_inp = []
                r_inp = []
                x_ref = []
                r_ref = []
                for kk in range(k + 1):
                    ind = c == kk
                    x_inp.append(x[ind])
                    r_inp.append(r[ind])
                    ind = c_l == kk
                    x_ref.append(x_l[ind])
                    r_ref.append(r_l[ind])

                x_inp = torch.cat(x_inp, 0)
                r_inp = torch.cat(r_inp, 0)
                x_ref = torch.cat(x_ref, 0)
                r_ref = torch.cat(r_ref, 0)
                self.init_all(opt, x_inp)

                lr_list = [0.005, 0.002, 0.002, 0.002]  # TODO: current
                # lr_list = [0.005] * self.num_kernel_sigmas
                # lr_list = [0.02, 0.01, 0.01, 0.01]  # TODO: old

                for j in range(len(kernel_sigmas)):

                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pos_opt.parameters()), lr=lr_list[j])
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
                    kernel_sigma = kernel_sigmas[j]

                    gabor_im = self.gabor_fb_inp(x_inp, r_inp, kernel_sigma)  # [30, 128, 128]
                    gabor_ref = self.gabor_fb_ref(x_ref, r_ref, kernel_sigma)

                    gabor_im_features = self.learned_filter.encode(gabor_im.unsqueeze(0))
                    gabor_ref_features = self.learned_filter.encode(gabor_ref.unsqueeze(0))
                    gabor_im_features = [f.detach() for f in gabor_im_features]
                    gabor_ref_features = [f.detach() for f in gabor_ref_features]

                    slicing_exemplar = gabor_im_features.copy()[0:]

                    slicing_torch_inp = Slicing_torch(device=self.device, layers=slicing_exemplar, repeat_rate=self.upscaling_rate)
                    slicing_torch_ref = Slicing_torch(device=self.device, layers=gabor_ref_features, repeat_rate=2)

                    target_style = []
                    target_corr = []
                    for si in range(0, len(gabor_im_features)):
                        cur_feature = gabor_im_features[si]
                        cur_feature_style = utils.gram_matrix(cur_feature).detach()
                        target_style.append(cur_feature_style)

                        cur_feature_corr = utils.deep_corr_matrix(self.device, cur_feature, cur_feature.shape[-2], cur_feature.shape[-1]).detach()
                        target_corr.append(cur_feature_corr)

                    target_corr_idx = opt.corr_layer
                    gabor_im_features = gabor_im_features[target_corr_idx]
                    gabor_ref_features = gabor_ref_features[target_corr_idx]

                    plt.figure(1)
                    plt.subplot(121)
                    plt.imshow(gabor_im[0].detach().cpu().numpy(), cmap='gray')
                    plt.subplot(122)
                    plt.imshow(gabor_ref[0].detach().cpu().numpy(), cmap='gray')
                    plt.savefig(self.output_folder + '/' + folder_name + '/target/gabor_sigma{:}'.format(j))
                    plt.clf()

                    target_corr_gabor = utils.deep_corr_matrix(self.device, gabor_im_features, gabor_im_features.shape[-2], gabor_im_features.shape[-1]).detach()
                    target_corr_gabor_ref = utils.deep_corr_matrix(self.device, gabor_ref_features, gabor_im_features.shape[-2], gabor_im_features.shape[-1]).detach()

                    out_cls = []
                    for kk in exist_cls:
                        out_cls.append(init_cls[kk])
                    for kk in cur_cls:
                        out_cls.append(init_cls[kk])
                    out_cls = torch.cat(out_cls, 0)

                    pos_opt.train()
                    break_loop = False
                    inner_run = [0]
                    log_losses = []

                    for epoch in tqdm(range(epochs)):
                        if break_loop:
                            break

                        start_time = time()

                        def closure():
                            global break_loop
                            # init_guess.data.clamp_(0, 1)

                            optimizer.zero_grad()

                            out, out_rad = pos_opt(exist_cls, cur_cls, target_pts_stats)

                            loss_dcor = torch.zeros(1).to(self.device)
                            loss_style = torch.zeros(1).to(self.device)
                            loss_slice = torch.zeros(1).to(self.device)
                            # loss_hist = torch.zeros(1).to(self.device)
                            # loss_wavelet = torch.zeros(1).to(self.device)
                            # loss_feature = torch.zeros(1).to(self.device)
                            # loss_fourier = torch.zeros(1).to(self.device)
                            # loss_dwt = torch.zeros(1).to(self.device)
                            loss_gabor = torch.zeros(1).to(self.device)
                            # loss_tv = torch.zeros(1).to(self.device)

                            if self.loss_type == 'slice':
                                gabor_out = self.gabor_fb_out(out, kernel_sigma)
                                gabor_out_features = self.learned_filter.encode(gabor_out.unsqueeze(0))  # [12][1, 1, 64, 64]
                                loss_slice += slicing_torch_inp(gabor_out_features)
                                gabor_out_features = gabor_out_features[2]

                                loss = loss_slice

                            if self.loss_type == 'slice_corr':

                                gabor_out = self.gabor_fb_out(out, out_rad, kernel_sigma)
                                gabor_out_features = self.learned_filter.encode(gabor_out.unsqueeze(0))  # [12][1, 1, 64, 64]

                                loss_slice += slicing_torch_inp(gabor_out_features[0:])

                                gabor_out_features = gabor_out_features[2]

                                out_corr_gabor = utils.deep_corr_matrix(self.device, gabor_out_features, gabor_im_features.shape[-2], gabor_im_features.shape[-1])

                                loss_gabor += F.mse_loss(out_corr_gabor, target_corr_gabor)

                                loss = opt.weight_corr * loss_gabor + opt.weight_slice * loss_slice

                            if self.loss_type == 'gabor':

                                jitter = (torch.rand((opt.grid_num * opt.upscaling_rate) * (opt.grid_num * opt.upscaling_rate), 2).to(self.device) * 2 - 1) * (
                                        1 / (opt.grid_num * opt.upscaling_rate + 1)) * 1.0

                                gabor_out = self.gabor_fb_out(out, out_rad, kernel_sigma, jitter)

                                gabor_out_features = self.learned_filter.encode(gabor_out.unsqueeze(0))  # [12][1, 1, 64, 64]

                                for si in range(opt.style_layer_start, opt.style_layer_end):
                                    cur_feature = gabor_out_features[si]
                                    cur_feature_style = utils.gram_matrix(cur_feature)
                                    loss_style += opt.weight_style * (cur_feature_style - target_style[si]).pow(2).sum()

                                gabor_out_features = gabor_out_features[target_corr_idx]

                                out_corr_gabor = utils.deep_corr_matrix(self.device, gabor_out_features, gabor_im_features.shape[-2], gabor_im_features.shape[-1])
                                loss_dcor += opt.weight_corr * F.mse_loss(out_corr_gabor[0:], target_corr_gabor[0:])

                                loss = loss_dcor + loss_style


                            loss.backward()
                            inner_run[0] += 1
                            log_losses.append(loss)
                            return loss

                        optimizer.step(closure)
                        # scheduler.step()

                        end_time = time()
                        self.opt_time[k].append(end_time - start_time)

                        if epoch % opt.log_freq == 0 or epoch == epochs - 1:

                            # TODO: visualize out and out_rad
                            out, out_rad = pos_opt(exist_cls, cur_cls, target_pts_stats)

                            gabor_out = self.gabor_fb_out(out, out_rad, kernel_sigma)

                            gabor_out_features = self.learned_filter.encode(gabor_out.unsqueeze(0))  # [12][1, 1, 64, 64]

                            gabor_out_features = gabor_out_features[target_corr_idx]

                            out_corr_gabor = utils.deep_corr_matrix(self.device, gabor_out_features, gabor_im_features.shape[-2], gabor_im_features.shape[-1])

                            gabor_out_grid = torch.clamp(torchvision.utils.make_grid(gabor_out.unsqueeze(1) / gabor_out.unsqueeze(1).max(), nrow=4, padding=2), 0, 1)
                            gabor_im_grid = torch.clamp(torchvision.utils.make_grid(gabor_im.unsqueeze(1) / gabor_im.unsqueeze(1).max(), nrow=4, padding=2), 0, 1)
                            gabor_ref_grid = torch.clamp(torchvision.utils.make_grid(gabor_ref.unsqueeze(1) / gabor_ref.unsqueeze(1).max(), nrow=4, padding=2), 0, 1)

                            gabor_out_features_grid = torch.clamp(torchvision.utils.make_grid(gabor_out_features[0].unsqueeze(1), nrow=8, padding=2), 0, 1)
                            gabor_im_features_grid = torch.clamp(torchvision.utils.make_grid(gabor_im_features[0].unsqueeze(1), nrow=8, padding=2), 0, 1)
                            gabor_ref_features_grid = torch.clamp(torchvision.utils.make_grid(gabor_ref_features[0].unsqueeze(1), nrow=8, padding=2), 0, 1)

                            out_corr_gabor_grid = torch.clamp(torchvision.utils.make_grid(out_corr_gabor.unsqueeze(1), nrow=8, padding=2), 0, 1)
                            inp_corr_gabor_grid = torch.clamp(torchvision.utils.make_grid(target_corr_gabor.unsqueeze(1), nrow=8, padding=2), 0, 1)
                            ref_corr_gabor_grid = torch.clamp(torchvision.utils.make_grid(target_corr_gabor_ref.unsqueeze(1), nrow=8, padding=2), 0, 1)

                            plt.figure(1)
                            plt.subplot(131)
                            plt.title('Output Transform', fontsize=10)
                            plt.imshow(gabor_out_grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                            plt.subplot(132)
                            plt.title('Exemplar Transform', fontsize=10)
                            plt.imshow(gabor_im_grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                            plt.subplot(133)
                            plt.title('Ref Transform', fontsize=10)
                            plt.imshow(gabor_ref_grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                            plt.savefig(self.output_folder + '/' + folder_name + '/output/gabor_transform_sigma{:}'.format(j))
                            plt.clf()

                            plot_grid(gabor_out_features_grid, self.output_folder + '/' + folder_name + '/output/gabor_out_feature_{:}'.format(j))
                            plot_grid(gabor_im_features_grid, self.output_folder + '/' + folder_name + '/output/gabor_inp_feature_{:}'.format(j))
                            plot_grid(gabor_ref_features_grid, self.output_folder + '/' + folder_name + '/output/gabor_ref_feature_{:}'.format(j))

                            plot_grid(out_corr_gabor_grid, self.output_folder + '/' + folder_name + '/output/corr_out_feature_{:}'.format(j))
                            plot_grid(inp_corr_gabor_grid, self.output_folder + '/' + folder_name + '/output/corr_inp_feature_{:}'.format(j))
                            plot_grid(ref_corr_gabor_grid, self.output_folder + '/' + folder_name + '/output/corr_ref_feature_{:}'.format(j))

                            log_loss = log_losses[-1]

                            print(
                                'Class: {:}/{:}, Scale: {:}/{:}. Epoch: {:}/{:}, Loss: {:.6f}'.format(k + 1, num_classes, j + 1, len(kernel_sigmas), epoch + 1, epochs,
                                                                                                      log_loss.item()))
                            losses.append(log_loss.item())
                            plt.figure(1)
                            plt.plot(losses)
                            plt.title('final loss: {:.6f}'.format(losses[-1]))
                            plt.savefig(self.output_folder + '/' + folder_name + '/output/losses_opt')
                            plt.clf()
                            # print('losses:', len(losses))
                            np.savetxt(self.output_folder + '/' + folder_name + '/output/losses.txt', losses)

                            print('lr:', optimizer.param_groups[0]['lr'])

                            plot_mc_disk_np(out.detach().cpu().numpy(), out_rad.detach().cpu().numpy(), out_cls.detach().cpu().numpy(), 1,
                                            self.output_folder + '/' + folder_name + '/output/{:}_n{:}_scale{:}_epoch{:}'.format(opt.scene_name, out.shape[0], j, epoch))
                            plot_mc_disk_np(out.detach().cpu().numpy(), out_rad.detach().cpu().numpy(), out_cls.detach().cpu().numpy(), 1,
                                            self.output_folder + '/' + folder_name + '/output/{:}_n{:}'.format(opt.scene_name, out.shape[0]), 'png')

                            pts_txt_save = np.concatenate([
                                np.expand_dims(out_cls.detach().cpu().numpy(), 1),
                                out.detach().cpu().numpy(),
                                np.expand_dims(out_rad.detach().cpu().numpy(), 1),
                            ], 1)
                            np.savetxt(self.output_folder + '/' + folder_name + '/output/{:}_n{:}_scale{:}_epoch{:}.txt'.format(opt.scene_name, out.shape[0], j, epoch), pts_txt_save)

                            if inner_run[0] > inner_freq:
                                loss_init = log_losses[0]
                                loss_pre = log_losses[inner_run[0] - inner_freq]
                                loss_now = log_losses[inner_run[0] - 1]
                                # print(loss_init, loss_pre, loss_now)
                                decrease_perc = ((loss_pre - loss_now) / (loss_init - loss_now)).tolist()[0]
                                print('decrease_perc:', decrease_perc, 0.01)
                                if decrease_perc < 0.01:
                                    print('converged')
                                    break_loop = True

                            if epoch == epochs - 1 or break_loop:

                                plot_mc_disk_np(out.detach().cpu().numpy(), out_rad.detach().cpu().numpy(), out_cls.detach().cpu().numpy(), 1,
                                                self.all_test_output + '/{:}_output_n{:}'.format(x_name, out.shape[0]), 'pdf')
                                plot_mc_disk_np(out.detach().cpu().numpy(), out_rad.detach().cpu().numpy(), out_cls.detach().cpu().numpy(), 1,
                                                self.all_test_output + '/{:}_output_n{:}'.format(x_name, out.shape[0]), 'png')

                                pts_txt_save = np.concatenate([
                                    np.expand_dims(out_cls.detach().cpu().numpy(), 1),
                                    out.detach().cpu().numpy(),
                                    np.expand_dims(out_rad.detach().cpu().numpy(), 1),
                                ], 1)
                                np.savetxt(self.all_test_output + '/{:}_output_n{:}.txt'.format(x_name, out.shape[0]), pts_txt_save)

                        if epoch % opt.log_freq == 0:
                            if self.loss_type == 'slice' and epoch % opt.log_freq == 0:  # default: iterations=250, log_freq=10
                                # print('===> update slicing exemplar')
                                slicing_torch_inp.update_slices(slicing_exemplar)  # optional

                    pts_txt_save = np.concatenate([
                        np.expand_dims(out_cls.detach().cpu().numpy(), 1),
                        out.detach().cpu().numpy(),
                        np.expand_dims(out_rad.detach().cpu().numpy(), 1)
                    ], 1)

                    np.savetxt(self.output_folder + '/' + folder_name + '/output/{:}_n{:}_scale{:}.txt'.format(opt.scene_name, pts_txt_save.shape[0], j), pts_txt_save)

                # rescale the existing points to leave space for synthesizing latter points
                # cur_out = pos_opt.get_cur_cls(k)
                cur_out, _ = pos_opt(exist_cls, cur_cls, target_pts_stats)
                cur_out_before = cur_out.clone()
                if num_classes > 1:
                    inp_xmin = x_inp[:, 0].min()
                    inp_xmax = x_inp[:, 0].max()
                    inp_ymin = x_inp[:, 1].min()
                    inp_ymax = x_inp[:, 1].max()
                    margin_space_out = min(min(inp_xmin, 1 - inp_xmax), min(inp_ymin, 1 - inp_ymax)) / self.upscaling_rate
                    normalization_torch(cur_out, 0, edge_space=margin_space_out)

                start_idx = 0
                end_idx = init_input_pts[0].shape[0]
                for kk in range(k + 1):
                    init_input_pts[kk] = cur_out[start_idx:end_idx].detach().cpu().numpy()
                    start_idx = end_idx
                    if kk < k:
                        end_idx += init_input_pts[kk + 1].shape[0]
