import os
from PIL import Image
import numpy as np
import cv2
import torch

def make_depth_bin(relative):
    depth_dir = '/Users/youzunzhi/pro/datasets/nyuv2_depth_data/bts_splits/test'
    if relative:
        save_dir = '../dataset/nyuv2/images/test_rel_annot/'
    else:
        save_dir = '../dataset/nyuv2/images/test_abs_annot/'
    os.makedirs(save_dir, exist_ok=True)
    dataset_file = '/home/u2263506/MDE_Dissect/data/nyudv2_test.txt' if torch.cuda.is_available() '/Users/youzunzhi/pro/EVA/code/MDE_Dissect/data/nyudv2_test.txt'
    with open(dataset_file, 'r') as f:
        for l in f.readlines():
            depth_path = os.path.join(depth_dir, l.split()[1])
            depth = np.array(Image.open(depth_path)).astype(np.float)
            depth /= 1000.
            labels = get_labels_sid(depth, relative=relative, ordinal_c=10)
            save_fname = os.path.join(save_dir, depth_path.split('/')[-1])
            # Image.fromarray(labels).save(save_fname)
            cv2.imwrite(save_fname, labels)


def get_labels_sid(depth, relative=True, ordinal_c=10.0, alpha = 1.0, beta = 11., dataset='nyu'):
    #alpha = 0.001
    #beta = 80.0

    # set as consistant with paper to add min value to 1 and set min as 0.01 (cannot converge on both nets)

    # if dataset == 'kitti':
    #     alpha = 1.0
    #     beta = 80.999#new alpha is 0.01 which is consistant with other network
    # elif dataset == 'nyu':
    #     alpha = 1.0
    #     beta  = 10.999

    K = float(ordinal_c)
    if relative:
        alpha = depth.min()+0.999
        beta = depth.max()+1.

    # labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    labels = K * np.log((depth+0.999) / alpha) / np.log(beta / alpha)
    labels = labels.astype(np.int)
    return labels


def visualize_discretization(depth_min=0, depth_max=10, K=10., alpha = 1.0, beta = 11):
    depth = np.tile(np.linspace(depth_min, depth_max, 100), (20, 1))
    labels = get_labels_sid(depth, ordinal_c=K, alpha=alpha, beta=beta)
    import matplotlib.pyplot as plt
    plt.imshow(labels)
    plt.show()
    plt.close()


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


if __name__ == '__main__':
    make_depth_bin(relative=True)