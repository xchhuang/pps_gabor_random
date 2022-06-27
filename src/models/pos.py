import numpy as np
import torch
import matplotlib.pyplot as plt


class PositionOptimizer(torch.nn.Module):
    def __init__(self, init_pos, cur_cls, k):
        super(PositionOptimizer, self).__init__()

        self.w_opt_list = torch.nn.ParameterList()
        self.w_fix_list = torch.nn.ParameterList()

        # self.w_opt_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[k])))
        for kk in cur_cls:
            # if kk != k:
            #     self.w_fix_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk]), requires_grad=False))
            self.w_opt_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk])))

    def toroidalWrapAround_tr2(self, points, domain_size=1):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1), points - torch.floor(points), points)
        return torch.where(torch.lt(points, 0), points + torch.ceil(torch.abs(points)), points)

    def toroidalWrapAround_tr(self, points, margin=0):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1 - margin), points + margin - torch.floor(points + margin) + margin, points)
        return torch.where(torch.lt(points, margin), points - margin + torch.ceil(torch.abs(points - margin)) - margin, points)

    def get_cur_cls(self, k):
        p = self.w_opt_list[0]
        return p

    def forward(self, cur_cls, k, target_pts_stats):
        # print('target_pts_stats:', target_pts_stats)
        margin = min(target_pts_stats[k][0], target_pts_stats[k][1])
        # print('margin:', margin)
        y = []
        for c in cur_cls:
            y.append(self.w_opt_list[c])
        y = torch.cat(y, 0)
        y = self.toroidalWrapAround_tr2(y)
        return y


class PositionOptimizer_pro(torch.nn.Module):
    def __init__(self, init_pos, exist_cls, cur_cls):
        super(PositionOptimizer_pro, self).__init__()

        self.w_opt_list = torch.nn.ParameterList()
        self.w_fix_list = torch.nn.ParameterList()
        for kk in exist_cls:
            self.w_fix_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk]), requires_grad=False))
        for kk in cur_cls:
            self.w_opt_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk])))

    def toroidalWrapAround_tr2(self, points, domain_size=1):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1), points - torch.floor(points), points)
        return torch.where(torch.lt(points, 0), points + torch.ceil(torch.abs(points)), points)

    def toroidalWrapAround_tr(self, points, margin=0):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1 - margin), points + margin - torch.floor(points + margin) + margin, points)
        return torch.where(torch.lt(points, margin), points - margin + torch.ceil(torch.abs(points - margin)) - margin, points)

    def get_cur_cls(self, k):
        p = self.w_opt_list[0]
        return p

    def forward(self, exist_cls, cur_cls, target_pts_stats, use_jitter=False):
        # print('exist_cls, cur_cls:', exist_cls, cur_cls)
        k = cur_cls[0]
        margin = min(target_pts_stats[k][0], target_pts_stats[k][1])
        # print('margin:', margin)
        y = []
        for c in exist_cls:
            y.append(self.w_fix_list[c])
        for c in cur_cls:
            y.append(self.w_opt_list[0])
        y = torch.cat(y, 0)

        if use_jitter:
            jitter = (torch.rand(y.shape).to(y.device) * 2 - 1) * 0.2   # manually jitter the samples for better convergence
            y = y + jitter

        # y = self.toroidalWrapAround_tr2(y)
        y = self.toroidalWrapAround_tr(y)
        return y


class PositionOptimizer_pro_whole(torch.nn.Module):
    def __init__(self, init_pos, exist_cls, cur_cls):
        super(PositionOptimizer_pro_whole, self).__init__()

        self.w_opt_list = torch.nn.ParameterList()
        self.w_fix_list = torch.nn.ParameterList()
        for kk in exist_cls:
            self.w_fix_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk]), requires_grad=False))
        for kk in cur_cls:
            self.w_opt_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk])))

    def toroidalWrapAround_tr2(self, points, domain_size=1):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1), points - torch.floor(points), points)
        return torch.where(torch.lt(points, 0), points + torch.ceil(torch.abs(points)), points)

    def toroidalWrapAround_tr(self, points, margin=0):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1 - margin), points + margin - torch.floor(points + margin) + margin, points)
        return torch.where(torch.lt(points, margin), points - margin + torch.ceil(torch.abs(points - margin)) - margin, points)

    def get_cur_cls(self, k):
        p = self.w_opt_list[0]
        return p

    def forward(self, exist_cls, cur_cls, target_pts_stats):
        # print('target_pts_stats:', target_pts_stats)
        k = cur_cls[0]
        margin = min(target_pts_stats[k][0], target_pts_stats[k][1])
        # print('margin:', margin)
        y = [self.w_opt_list[0]]
        y = torch.cat(y, 0)
        y = self.toroidalWrapAround_tr2(y)
        # y = torch.clamp(y, margin, 1-margin)
        return y


class PositionOptimizer_whole(torch.nn.Module):
    def __init__(self, init_pos, cur_cls):
        super(PositionOptimizer_whole, self).__init__()

        self.w_opt_list = torch.nn.ParameterList()
        self.w_opt_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos)))

    def toroidalWrapAround_tr(self, points, margin=0):  # for arbitrary domain size
        points = torch.where(torch.gt(points, 1 - margin), points + margin - torch.floor(points + margin) + margin, points)
        return torch.where(torch.lt(points, margin), points - margin + torch.ceil(torch.abs(points - margin)) - margin, points)

    def forward(self, cur_cls, target_pts_stats):
        # print(target_pts_stats)
        margin = 0.04
        y = self.w_opt_list[0]
        # print('PositionOptimizer_whole forward:', y.shape)
        y = self.toroidalWrapAround_tr(y, margin)
        return y
