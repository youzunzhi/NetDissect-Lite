import os
from PIL import Image
import numpy as np
import cv2
import torch


def make_depth_bin(relative):
    # # testing set
    # depth_dir = '/work/u2263506/nyu2_data/data/bts_splits/test' if torch.cuda.is_available() else '/Users/youzunzhi/pro/datasets/nyuv2_depth_data/bts_splits/test'
    # if relative:
    #     save_dir = '../dataset/nyuv2/images/test_rel_annot/'
    # else:
    #     save_dir = '../dataset/nyuv2/images/test_abs_annot/'
    # os.makedirs(save_dir, exist_ok=True)
    # dataset_file = '/home/u2263506/MDE_Dissect/data/nyudv2_test.txt' if torch.cuda.is_available() else '/Users/youzunzhi/pro/EVA/code/MDE_Dissect/data/nyudv2_test.txt'
    # with open(dataset_file, 'r') as f:
    #     for l in f.readlines():
    #         depth_path = os.path.join(depth_dir, l.split()[1])
    #         depth = np.array(Image.open(depth_path)).astype(np.float)
    #         depth /= 1000.
    #         labels = get_labels_sid(depth, relative=relative, ordinal_c=10)
    #         save_fname = os.path.join(save_dir, depth_path.split('/')[-1])
    #         # Image.fromarray(labels).save(save_fname)
    #         cv2.imwrite(save_fname, labels)

    # training set
    depth_dir = '/work/u2263506/nyu2_data/data/bts_splits/train' if torch.cuda.is_available() else '/Users/youzunzhi/pro/datasets/nyuv2_depth_data/bts_splits/train'
    if relative:
        save_dir = '../dataset/nyuv2/images/train_rel_annot/'
    else:
        save_dir = '../dataset/nyuv2/images/train_abs_annot/'
    os.makedirs(save_dir, exist_ok=True)
    dataset_file = '' if torch.cuda.is_available() else '/Users/youzunzhi/pro/EVA/source_code/bts/train_test_inputs/nyudepthv2_train_files_with_gt.txt'
    with open(dataset_file, 'r') as f:
        for l in f.readlines():
            depth_path = os.path.join(depth_dir, l.split()[1][1:])
            depth = np.array(Image.open(depth_path)).astype(np.float)
            depth /= 1000.
            labels = get_labels_sid(depth, relative=relative, ordinal_c=10)
            bin_name =  (depth_path.split('/')[-1]).replace('sync_depth_', 'depth_bin_abs10_')
            save_fname = os.path.join(save_dir, bin_name)
            cv2.imwrite(save_fname, labels)


def get_labels_sid(depth, relative=True, ordinal_c=10.0, dataset='nyu'):
    # alpha = 0.001
    # beta = 80.0

    # set as consistant with paper to add min value to 1 and set min as 0.01 (cannot converge on both nets)

    # if dataset == 'kitti':
    #     alpha = 1.0
    #     beta = 80.999#new alpha is 0.01 which is consistant with other network
    # elif dataset == 'nyu':
    #     alpha = 1.0
    #     beta  = 10.999

    K = float(ordinal_c)
    if relative:
        alpha = depth[depth != 0].min() + 0.999
        beta = depth.max() + 1.
    else:
        alpha = 1.0
        beta = 11.0

    # labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    labels = K * np.log((depth + 0.999) / alpha) / np.log(beta / alpha)
    labels = labels.astype(np.int) + 1
    labels[depth == 0] = 0
    return labels


def visualize_discretization(depth_min=0, depth_max=10, K=10.):
    depth = np.tile(np.linspace(depth_min, depth_max, 100), (20, 1))
    labels = get_labels_sid(depth, ordinal_c=K)
    import matplotlib.pyplot as plt
    plt.imshow(labels)
    plt.show()
    plt.close()



if __name__ == '__main__':
    make_depth_bin(relative=False)
