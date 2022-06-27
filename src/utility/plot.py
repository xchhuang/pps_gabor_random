import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# color_dict = {
#     0: 'r',
#     1: 'g',
#     2: 'b',
#     3: 'c',
#     4: 'm',
#     5: 'y',
#     6: 'k',
#     7: 'orange',
#     8: 'gray',
#     9: 'pink',
#     10: 'brown',
#     11: 'purple',
# }

""" more classes """
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
    12: 'royalblue',
    13: 'blueviolet',
    14: 'navy',
    15: 'tomato',
    16: 'chocolate',
    17: 'lawngreen'
}


feature_color_dict = {
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
    12: 'royalblue',
    13: 'blueviolet',
    14: 'navy',
    15: 'tomato',
    16: 'chocolate',
    17: 'lawngreen'
}


def plot_mc_point_np(pts, cls, size, title, ext='pdf'):
    # size = 10  # TODO: hard-coded for now
    N = pts.shape[0]
    plt.figure(1)
    # for i in range(N):
    #     plt.scatter(pts[i, 0], pts[i, 1], s=size, c=color_dict[cls[i]])
    plt.scatter(pts[:, 0], pts[:, 1], s=size, c=color_dict[0])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.show()
    plt.savefig(title + '.' + ext, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def plot_mc_disk_np(pts, rad, cls, size, title, ext='pdf', plot_point=True):
    # print('plot_mc_disk_np')
    # size = 10  # TODO: hard-coded for now
    plot_point = plot_point
    # print('rad:', rad.shape)
    if rad.ndim == 1:
        num_diff_rad = len(set(list(rad)))
        # print('num_diff_rad:', num_diff_rad)
    else:   # TODO: highdim features OR points
        # if rad.shape[1] > 1:
        #     num_diff_rad = 1
        # else:
        num_diff_rad = rad.shape[1]
        # num_diff_rad = 1

    if num_diff_rad == 1:  # TODO: manual setting
        plot_point = True
    else:
        plot_point = False
    if plot_point:
        scale = 10
    else:
        scale = 1  # tmp chnaged
    s = size * scale

    # print('plot_point:', plot_point)
    plt.figure(1)
    # plt.title(title)
    fig, ax = plt.subplots()
    # plt.scatter(pts[:, 0], pts[:, 1], s=size, c=color_dict[0])
    if plot_point:
        num_class = len(set(list(cls)))
        for i in range(num_class):
            idx = cls == i
            # print(i, cls[idx], num_class)
            # print(cls[idx][0], num_class)
            plt.scatter(pts[idx, 0], pts[idx, 1], s=s, c=color_dict[cls[idx][0]])
    else:

        for i in range(pts.shape[0]):
            if not plot_point:
                # print('rad:', pts.shape, rad.shape)
                if rad.ndim == 1:
                    circle = plt.Circle((pts[i, 0], pts[i, 1]), rad[i], color=color_dict[cls[i]], lw=s, fill=False)
                    ax.add_artist(circle)
                else:   # > 1, assume that in multi-attribute cases, we don't have multiple class IDs
                    plt.scatter(pts[i, 0], pts[i, 1], s=s, c=color_dict[cls[i]])
                    # for j in range(rad.shape[1]):
                    #     circle = plt.Circle((pts[i, 0], pts[i, 1]), rad[i, j], color=color_dict[j], lw=s, fill=False)
                    #     # circle = plt.Circle((pts[i, 0], pts[i, 1]), 0.02, color=color_dict[j], lw=s, fill=True)
                    #     ax.add_artist(circle)
            else:
                plt.scatter(pts[i, 0], pts[i, 1], s=s, c=color_dict[cls[i]])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.show()
    plt.savefig(title + '.' + ext, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def plot_mc_feature_np(pts, rad, cls, size, title, ext='pdf'):
    # size = 10  # TODO: hard-coded for now
    plot_point = False
    # print('rad:', rad.shape)
    if rad.ndim == 1:
        num_diff_rad = len(set(list(rad)))
        # print('num_diff_rad:', num_diff_rad)
    else:
        num_diff_rad = rad.shape[1]
    if num_diff_rad == 1:  # TODO: manual setting
        plot_point = True
    if plot_point:
        scale = 10
    else:
        scale = 1
    s = size * scale
    plt.figure(1)
    # plt.title(title)
    fig, ax = plt.subplots()
    # plt.scatter(pts[:, 0], pts[:, 1], s=size, c=color_dict[0])
    if plot_point:
        num_class = len(set(list(cls)))
        for i in range(num_class):
            idx = cls == i
            # print(i, cls[idx], num_class)
            # print(cls[idx][0], num_class)
            plt.scatter(pts[idx, 0], pts[idx, 1], s=s, c=color_dict[cls[idx][0]])
    else:

        for i in range(pts.shape[0]):
            # if not plot_point:
            # print('rad:', pts.shape, rad.shape)
            if rad.ndim == 1:
                circle = plt.Circle((pts[i, 0], pts[i, 1]), rad[i], color=color_dict[cls[i]], lw=s, fill=False)
                ax.add_artist(circle)
            else:   # > 1, assume that in multi-attribute cases, we don't have multiple class IDs
                for j in range(rad.shape[1]):
                    circle = plt.Circle((pts[i, 0], pts[i, 1]), rad[i, j], color=color_dict[len(color_dict)-j-1], lw=s/rad.shape[1], fill=False)
                    # circle = plt.Circle((pts[i, 0], pts[i, 1]), 0.02, color=color_dict[j], lw=s, fill=True)
                    ax.add_artist(circle)
                    plt.scatter(pts[i, 0], pts[i, 1], s=s, c=color_dict[cls[i]])

            # else:
            #     plt.scatter(pts[i, 0], pts[i, 1], s=s, c=color_dict[cls[i]])

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.show()
    plt.savefig(title + '.' + ext, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def plot_img(im, title):
    # print(im.shape, im.min(), im.max())
    im_save = np.clip(im / im.max(), 0, 1)
    # plt.figure(1)
    # plt.imshow(im_save, cmap='viridis')  # RdBu
    # plt.axis('off')
    # plt.savefig(title, bbox_inches='tight', pad_inches=0)
    # plt.clf()
    plt.imsave(title + '.png', im_save)
    plt.clf()


def plot_grid(im, title):
    # print(im.min(), im.max())
    im_save = im.detach().cpu().numpy().transpose(1, 2, 0)
    im_save = np.clip(im_save / im_save.max(), 0, 1)
    # plt.figure(1)
    # plt.imshow(im_save, cmap='viridis')  # RdBu
    # plt.axis('off')
    # plt.savefig(title, bbox_inches='tight', pad_inches=0)
    # plt.clf()
    plt.imsave(title + '.png', im_save)
    plt.clf()


def plot_feature_as_point_cloud(feature_map, title):
    # feature_map = feature_map / torch.sum(feature_map, 0)
    # feature_map = feature_map / feature_map.max()
    # feature_map = feature_map.view(-1)
    # feature_map = feature_map.detach().cpu().numpy()
    x_coord = feature_map[0].view(-1).detach().cpu().numpy()
    y_coord = feature_map[1].view(-1).detach().cpu().numpy()
    x_coord = x_coord / x_coord.max()
    y_coord = y_coord / y_coord.max()

    val = torch.mean(feature_map, 0).view(-1).detach().cpu().numpy()
    # print(feature_map.shape, feature_map.min(), feature_map.max())
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(feature_map[0].detach().cpu().numpy(), cmap='gray')
    # plt.subplot(122)
    # plt.imshow(feature_map[1].detach().cpu().numpy(), cmap='gray')
    # plt.show()
    # x_coord = feature_map
    # y_coord = 1 - feature_map
    plt.figure(1)
    plt.scatter(x_coord, y_coord, s=5, c=val)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.savefig(title, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.clf()


def plot_mc_point_3d(pts, cls, size, title, ext='png'):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r', s=size)
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # plt.show()
    plt.savefig(title + '.' + ext, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.clf()


# def plot_mc_point_highdim(pts, cls, size, title, ext='png'):
#     plt.figure(1)
#     for i in range(pts.shape[0]):
#         for f in range(num_feature_dim):
#             circle = plt.Circle((pts[i, 0], pts[i, 1]), rad[i, f], color=feature_color_dict[f], lw=1, fill=False)
#             plt.gca().add_artist(circle)
#         # plt.scatter(pts[:, 0], pts[:, 1], s=size, c=color_dict[0])
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.gca().set_aspect('equal', adjustable='box')
#     # plt.axis('off')
#     plt.show()
#     
#     plt.savefig(title + '.' + ext, bbox_inches='tight', pad_inches=0, dpi=200)
#     plt.clf()
