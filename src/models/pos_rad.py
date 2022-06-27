import numpy as np
import torch


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
    def __init__(self, init_pos, init_rad, exist_cls, cur_cls):
        super(PositionOptimizer_pro, self).__init__()

        self.w_opt_list = torch.nn.ParameterList()
        self.w_fix_list = torch.nn.ParameterList()
        self.r_opt_list = torch.nn.ParameterList()
        self.r_fix_list = torch.nn.ParameterList()

        for kk in exist_cls:
            self.w_fix_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk]), requires_grad=False))
            self.r_fix_list.append(torch.nn.Parameter(init_rad[kk], requires_grad=False))

        for kk in cur_cls:
            self.w_opt_list.append(torch.nn.Parameter(torch.FloatTensor(init_pos[kk])))
            self.r_opt_list.append(torch.nn.Parameter(init_rad[kk], requires_grad=False))

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
        min_r, max_r = target_pts_stats[k][2], target_pts_stats[k][3]
        # print('margin:', margin, min_r, max_r)
        y = []
        r = []

        num_exist = {}
        for c in exist_cls:
            num_exist[c] = self.w_fix_list[c].shape[0]
        num_cur = self.w_opt_list[0].shape[0]
        # print('num_exist:', num_exist)
        for c in exist_cls:
            # print('exist_cls:', c)
            y.append(self.w_fix_list[c])
            # print(num_cur / num_exist[c])
            r.append(self.r_fix_list[c])  # *(np.sqrt(num_cur / num_exist[c]))
        for c in cur_cls:  # always have only one cur_cls
            # print('cur_cls:', c)
            y.append(self.w_opt_list[0])
            r.append(self.r_opt_list[0])
        y = torch.cat(y, 0)
        r = torch.cat(r, 0)

        y = self.toroidalWrapAround_tr2(y)

        return y, r


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
        y = self.toroidalWrapAround_tr(y, margin)

        return y
