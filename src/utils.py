import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from time import time


class TVLoss(torch.nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def gram_matrix(input):
    if input.ndim == 4:
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        div_num = a * b * c * d

    if input.ndim == 5:
        a, b, c, d, e = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d * e)  # resise F_XL into \hat F_XL
        div_num = a * b * c * d * e

    G = torch.mm(features, features.t())  # compute the gram product
    # print('gram_matrix:', input.shape, features.shape, G.shape)
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(div_num)


def gram_matrix_our(input):
    # print('gram_matrix_our', input.shape)
    batch = input.shape[0]
    input = input[:, 0, :, :].view(batch, -1).unsqueeze(-1).unsqueeze(-1)
    # print('gram_matrix_our:', input.shape)
    if input.ndim == 4:
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        div_num = a * b * c * d
    else:
        print('wrong number of dimensions')

    G = torch.mm(features, features.t())  # compute the gram product
    # print('gram_matrix:', input.shape, features.shape, G.shape)
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    G = G.div(div_num)
    # G = torch.sigmoid(G)
    return G


def deep_corr_matrix(device, input, q, m):
    # start = time()

    a, b, c, d = input.size()  # a=batch size(=1)
    assert (c == d)
    assert (q == m)
    R = torch.zeros(b, 2 * c - 1, 2 * d - 1).to(device)

    x_cord = torch.arange(1, c + 1).float().to(device)
    _x_cord = torch.arange(c - 1, 0, -1).float().to(device)
    x_cord = torch.cat((x_cord, _x_cord), 0)

    x_grid = x_cord.repeat(2 * c - 1).view(2 * c - 1, 2 * c - 1)
    y_grid = x_grid.t()

    xconv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=c, padding=c - 1, groups=1, bias=False).to(device)

    center = (2 * c - 1) // 2.0
    # print('deep_corr_matrix:', c)

    for i in range(b):
        xconv2.weight.data = input[:, i, :, :].unsqueeze(1)
        xconv2.weight.requires_grad = False
        # print(xconv2.shape, input[:, i, :, :].unsqueeze(1).shape)
        corr_matrix = xconv2(input[:, i, :, :].unsqueeze(1))
        corr_matrix /= (x_grid * y_grid)
        R[i, :, :] = corr_matrix.squeeze(0).squeeze(0)

        if q % 2 == 1:
            out = R[:, int(center - c // 2 + math.ceil((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - (c - q) // 2.0 + 1),
                  int(center - c // 2 + math.ceil((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - (c - q) // 2.0 + 1)]
        else:
            out = R[:, int(center - c // 2 + math.floor((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - math.ceil((c - q) / 2.0) + 1),
                  int(center - c // 2 + math.floor((c - q) / 2.0)):int(
                      center + math.ceil(c / 2) - 1 - math.ceil((c - q) / 2.0) + 1)]

    # end = time()
    # out = out / torch.max(out, 0)[0]
    # print(input.shape, R.shape, out.shape)
    return out


def deep_corr_matrix_batch(device, input, q, m):
    # start = time()

    a, b, c, d = input.size()  # a=batch size(=1)
    assert (c == d)
    assert (q == m)
    R = torch.zeros(a, b, 2 * c - 1, 2 * d - 1).to(device)

    x_cord = torch.arange(1, c + 1).float().to(device)
    _x_cord = torch.arange(c - 1, 0, -1).float().to(device)
    x_cord = torch.cat((x_cord, _x_cord), 0)

    x_grid = x_cord.repeat(2 * c - 1).view(2 * c - 1, 2 * c - 1)
    y_grid = x_grid.t()

    xconv2 = torch.nn.Conv2d(in_channels=1, out_channels=1,
                             kernel_size=c, padding=c - 1, groups=1, bias=False).to(device)

    center = (2 * c - 1) // 2.0

    for i in range(b):
        xconv2.weight.data = input[:, i, :, :].unsqueeze(1)
        xconv2.weight.requires_grad = False
        corr_matrix = xconv2(input[:, i, :, :].unsqueeze(1))
        # print(corr_matrix.shape)
        corr_matrix /= (x_grid * y_grid)
        R[:, i, :, :] = corr_matrix.squeeze(0).squeeze(0)
        if q % 2 == 1:
            out = R[:, :, int(center - c // 2 + math.ceil((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - (c - q) // 2.0 + 1),
                  int(center - c // 2 + math.ceil((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - (c - q) // 2.0 + 1)]
        else:
            out = R[:, :, int(center - c // 2 + math.floor((c - q) / 2.0)):int(center + math.ceil(c / 2) - 1 - math.ceil((c - q) / 2.0) + 1),
                  int(center - c // 2 + math.floor((c - q) / 2.0)):int(
                      center + math.ceil(c / 2) - 1 - math.ceil((c - q) / 2.0) + 1)]

    # end = time()
    return out


def deep_corr_matrix_fast(device, input, q, m):
    out = []
    for b in range(input.shape[1]):
        x = input[:, b:b + 1, ...]
        # o = corr_cuda(x, x)
        o = corr_my(x, x)
        # print(x.shape, o.shape)
        # o = torch.sum(o, 1)
        # for j in range(o.shape[1]):
        #     plt.figure(1)
        #     plt.imshow(o[0, j].detach().cpu().numpy())
        #     plt.show()
        # print(o.shape)
        out.append(o)
    out = torch.cat(out, 0)
    return out


def normalization_np(points, d_space=2, edge_space=0.02):
    if edge_space == 0:
        return points
    if not isinstance(points, np.ndarray):
        points = points.numpy()
    else:
        points = points.copy()
    # print (points.shape)
    # print (points.shape)
    min = points.min(0)
    max = points.max(0)
    # print(min, max)
    # print (min.shape)
    # min = points.min(0)[0]
    # max = points.max(0)[0]
    for id in range(d_space):
        r = max[id] - min[id]
        points[:, id] = (((points[:, id] - min[id]) / r) * (1 - 2 * edge_space) + edge_space)
    return points


def normalization_torch(points, dd_feature, d_space=2, edge_space=0.1, egde_feature=0.2, norm=True):
    min = points.min(0)[0]
    max = points.max(0)[0]
    for id in range(d_space):
        r = max[id] - min[id]
        points[:, id] = (((points[:, id] - min[id]) / r) * (1 - 2 * edge_space) + edge_space)

    if norm:
        if points.size()[1] > 2:
            for id in range(d_space + dd_feature, points.size()[1]):
                r = max[id] - min[id]

                if r == 0:
                    continue

                points[:, id] = ((points[:, id] - min[id]) / r) * (1 - egde_feature) + egde_feature

    return points


def plot_mc_point_np(pts, cls, size, title):
    color_dict = {
        0: 'r',
        1: 'g',
        2: 'b',
        3: 'c',
        4: 'm',
        5: 'y',
        6: 'k',
        7: 'orange',
        8: 'gray',
        9: 'pink',
        10: 'brown',
        11: 'purple',
    }
    N = pts.shape[0]
    plt.figure(1)
    for i in range(N):
        plt.scatter(pts[i, 0], pts[i, 1], s=size, c=color_dict[cls[i]])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.show()
    plt.savefig(title + '.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def plot_img(im, title, sub_title=None):
    im_save = np.clip(im / im.max(), 0, 1)
    plt.figure(1)
    plt.imshow(im_save)
    # if sub_title:
    #     plt.title(sub_title)
    plt.axis('off')
    # plt.savefig(title, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.imsave(title + '.png', im_save)
    plt.close('all')
