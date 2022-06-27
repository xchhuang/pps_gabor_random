import torch
import numpy as np
import platform
import glob
from torch.utils.data import Dataset
import sys
import os

import matplotlib.pyplot as plt
from utils import normalization_np
import itertools
import random


class TestLoader(Dataset):
    def __init__(self, opt, path, res=128, scene_name='Army', train_or_test='train'):
        # print('Loader for progressive')
        self.splitter = '/'
        if platform.system() == 'Windows':
            self.splitter = '\\'

        self.upscaling_rate = opt.upscaling_rate
        self.num_classes = opt.num_classes
        self.train_or_test = train_or_test
        self.res = res
        self.edge_space = opt.edge_space
        self.scene_name = scene_name

        # np = path.split('/')[-1]
        path_prefix = '/'.join(path.split('/')[0:2])
        path_scene = path.split('/')[2]
        print('path_scene:', path_scene)
        # print(path_prefix, path_scene,)

        id_map = {}
        class_index = [1000 * (i + 1) for i in range(self.num_classes)]
        for i in range(self.num_classes):
            id_map[class_index[i]] = i

        data_s = []
        cls_s = []

        data_l = []
        cls_l = []

        exist_ref = False
        self.data_names = []
        folder = path

        if os.path.exists(folder + '/ref'):
            exist_ref = True

        filenames = glob.glob(folder + '/exemplar/*.txt')
        filenames.sort(key=self.cmp)

        for filename in filenames:

            scene_name = filename.split(self.splitter)[-1].split('.')[0]
            if scene_name not in [self.scene_name]:  # Part
                continue
            # print(filename, scene_name)
            self.data_names.append(scene_name)
            with open(filename) as f:
                lines = f.readlines()
                d = []
                c = []
                for line in lines:
                    line = line.strip().split(' ')
                    if folder != '../test_data/testset_stress':
                        line = [int(x) for x in line]
                        line_c = id_map[line[0]]
                        line_d = [float(x) / 10000 for x in line[1:]]
                    else:
                        line_c = float(line[0])
                        line_d = [float(x) for x in line[1:]]
                    d.append(line_d)
                    c.append(line_c)
                    # print(line_d, line_c)
                data_s.append(d)
                cls_s.append(c)

        if exist_ref:
            # print('exist_ref')
            filenames = glob.glob(folder + '/ref/*.txt')
            filenames.sort(key=self.cmp)

            # print(filenames)
            for filename in filenames:
                scene_name = filename.split(self.splitter)[-1].split('.')[0]

                if scene_name not in [self.scene_name]:  # Part
                    continue
                with open(filename) as f:
                    lines = f.readlines()
                    d = []
                    c = []
                    for line in lines:
                        line = line.strip().split(' ')
                        line = [int(x) for x in line]
                        # print(line)
                        line_c = id_map[line[0]]
                        line_d = [float(x) / 10000 for x in line[1:]]
                        d.append(line_d)
                        c.append(line_c)
                        # print(line_c)
                    data_l.append(d)
                    cls_l.append(c)

        else:
            data_l = data_s.copy()
            cls_l = cls_s.copy()

        self.data_s = data_s
        self.cls_s = cls_s
        self.data_l = data_l
        self.cls_l = cls_l
        print('data size small:', len(data_s), len(cls_s))
        print('data size large:', len(data_l), len(cls_l))

        arr = list(range(self.num_classes))
        cls_choices = []
        for nc in range(self.num_classes):
            cls_choices += list(itertools.combinations(arr, nc + 1))
        self.cls_choices = [list(range(self.num_classes))]

    def cmp(self, x):
        x = x.split(self.splitter)[-1].split('.')[0]
        # print(x)
        return x

    def __getitem__(self, index):
        x_s = self.data_s[index]
        c_s = self.cls_s[index]
        x_l = self.data_l[index]
        c_l = self.cls_l[index]

        x_name = self.data_names[index]
        x_s = np.array(x_s)
        c_s = np.array(c_s)
        r_s = np.ones_like(c_s)
        x_l = np.array(x_l)
        c_l = np.array(c_l)
        r_l = np.ones_like(c_l)

        ep = self.edge_space

        if ep > 0:  # optional
            x_s = normalization_np(x_s, edge_space=ep)
            x_l = normalization_np(x_l, edge_space=ep / 2)

        x_s = x_s.astype(np.float32)
        c_s = c_s.astype(np.int32)
        x_l = x_l.astype(np.float32)
        c_l = c_l.astype(np.int32)

        _kernel_sigma = 0.005

        cur_cls = np.random.choice(list(range(self.num_classes)))
        exist_cls = [i for i in range(cur_cls)]
        if self.train_or_test == 'train' and cur_cls == 0 and np.random.rand() > 1 / self.num_classes:
            c_s = np.zeros_like(c_s)
            c_s = c_s.astype(np.int32)

        return x_s, c_s, r_s, x_l, c_l, r_l, _kernel_sigma, x_name

    def __len__(self):
        return len(self.data_s)
