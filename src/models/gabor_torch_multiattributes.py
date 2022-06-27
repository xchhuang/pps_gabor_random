import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import defaultdict
from time import time
import sys

from utility.knn import knn, get_neighbor_torch
from models.pointnet2_utils import index_points
import torch.nn as nn


class ContinuousGaborFilterBanks(torch.nn.Module):
    def __init__(self, device, res, upscaling_rate, use_toroidal=False, k=30):
        super(ContinuousGaborFilterBanks, self).__init__()
        domain_size = 1
        self.res = res * upscaling_rate
        self.device = device
        self.upscaling_rate = upscaling_rate
        bnd = 1 / (self.res + 1)
        u = torch.from_numpy(np.linspace(bnd, domain_size - bnd, self.res)).to(device)
        v = torch.from_numpy(np.linspace(bnd, domain_size - bnd, self.res)).to(device)
        uu, vv = torch.meshgrid(u, v)
        uu = uu.contiguous().view(-1, 1)
        vv = vv.contiguous().view(-1, 1)
        self.grid_pts = torch.cat([vv, uu], 1).float()
        self.k = k

        params = []

        thetas = [i / 6 * np.pi for i in range(6)]
        for theta in thetas:
            for lamda in [np.pi]:  # np.pi
                for gamma in [0.99, 1.0, 1.01]:
                    params.append([theta, lamda, 0, gamma])

        # print('num params:', len(params))
        self.params = params

        params_torch = torch.from_numpy(np.array(self.params)).float().to(device)

        self.thetas = params_torch[:, 0]
        self.lambdas = params_torch[:, 1]
        self.phis = params_torch[:, 2]
        self.gammas = params_torch[:, 3]

        self.tao = [0, 1]

        use_endpoint = True
        spectrumRes = 3
        if spectrumRes % 2 == 0:
            use_endpoint = False

        freqStep = 1
        xlow = -spectrumRes * freqStep * (upscaling_rate) * 0.5
        xhigh = spectrumRes * freqStep * (upscaling_rate) * 0.5

        ylow = xlow
        yhigh = xhigh
        u_freq = torch.from_numpy(np.linspace(xlow, xhigh, spectrumRes, endpoint=use_endpoint)).float().to(device)
        v_freq = torch.from_numpy(np.linspace(ylow, yhigh, spectrumRes, endpoint=use_endpoint)).float().to(device)
        uu_freq, vv_freq = torch.meshgrid(u_freq, v_freq)
        self.grid_freq = torch.cat((vv_freq.unsqueeze(0).to(device), uu_freq.unsqueeze(0).to(device)), 0)
        self.spectrumRes = spectrumRes

    def forward(self, pts, rad, kernel_sigma, jitter=None):
        """
        pts: torch.tensor with size [N, 2]
        rad: torch.tensor with size [N, NUM_ATTRIBUTES]
        """

        rad_max, _ = rad.max(0)

        rad = rad / rad_max.unsqueeze(0)

        res = self.res
        grid_pts = self.grid_pts.unsqueeze(0)
        if jitter is not None:
            grid_pts = grid_pts + jitter.unsqueeze(0)

        # start_time = time()
        k = self.k
        if k > pts.shape[0]:
            k = pts.shape[0]

        idx = knn(pts.unsqueeze(0).permute(0, 2, 1), grid_pts.permute(0, 2, 1), k=k)
        group = index_points(pts.unsqueeze(0), idx)[0]

        group_rad = index_points(rad.unsqueeze(0), idx)[0]

        diff = group.permute(1, 0, 2) - grid_pts

        x = diff[:, :, 0].unsqueeze(0)
        y = diff[:, :, 1].unsqueeze(0)

        thetas = self.thetas.unsqueeze(-1).unsqueeze(-1)
        gammas = self.gammas.unsqueeze(-1).unsqueeze(-1)
        lambdas = self.lambdas.unsqueeze(-1).unsqueeze(-1)
        phis = self.phis.unsqueeze(-1).unsqueeze(-1)

        sigma = torch.from_numpy(np.array(kernel_sigma / self.upscaling_rate)).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # .repeat(len(self.params), 1, 1)
        spatial_term = torch.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma ** 2))

        real_val = []

        NUM_ATTRIBUTES = rad.shape[1]

        for tao in self.tao[1:]:

            spectral_term = torch.tensordot(diff, self.grid_freq, dims=[[2], [0]]).view(k, res * res, -1)
            angle = 2.0 * np.pi * spectral_term
            realCoeff = torch.pow(torch.cos(angle), tao)

            # imagCoeff = torch.sin(angle)
            # power = realCoeff ** 2 + imagCoeff ** 2
            scale = 1
            spatial_term_scale = torch.exp(-0.5 * (x ** 2 / (sigma * scale) ** 2 + y ** 2 / (sigma * scale) ** 2))
            power = realCoeff.permute(2, 0, 1) * spatial_term_scale
            # imagCoeff = torch.sin(angle)
            # power = (realCoeff ** 2 + imagCoeff ** 2) / k

            rv = power  # gabor feature function
            real_val.append(rv)

        real_val = torch.cat(real_val, 0)

        real_val = real_val.repeat(NUM_ATTRIBUTES, 1, 1) * group_rad.permute(2, 1, 0).repeat(self.spectrumRes * self.spectrumRes, 1, 1)
        real_val = torch.cat([spatial_term, real_val], 0)

        real_val = torch.sum(real_val, 1)
        out = real_val / k
        out = out.view(-1, res, res).float()

        return out

