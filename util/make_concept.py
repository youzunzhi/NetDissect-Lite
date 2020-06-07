import os
from PIL import Image
import numpy as np
import cv2
import torch


def make_index_csv_trash():
    # import csv
    fn = 'index.csv'
    image_names_file_test = '/Users/youzunzhi/pro/EVA/code/MDE_Dissect/data/nyudv2_test.txt'
    image_names_file_train = '/Users/youzunzhi/pro/EVA/source_code/bts/train_test_inputs/nyudepthv2_train_files_with_gt.txt'
    with open(fn, 'w') as f:
        # f.write('image,split,ih,iw,sh,sw,sem\n')
        f.write('image,split,ih,iw,sh,sw,abs\n')
        # f.write('image,split,ih,iw,sh,sw,rel\n')
        with open(image_names_file_test, 'r') as imf:
            for l in imf.readlines():
                im_name = l.split(' ')[0]
                im_index = int(im_name.split('_')[-1].split('.')[0])
                # for sem
                # write_line = f'test/{im_name},test,228,304,228,304,test_sem_annot/new_nyu_class13_{im_index + 1:04d}.png\n'
                # for abs
                write_line = f'test/{im_name},test,228,304,228,304,test_abs_annot/depth_bin_abs10_{im_index:05d}.png\n'
                # for rel
                # write_line = f'test/{im_name},test,228,304,228,304,test_rel_annot/depth_bin_rel10_{im_index:05d}.png\n'
                f.write(write_line)
        # training set
        with open(image_names_file_train, 'r') as imf:
            for l in imf.readlines():
                im_name, depth_name = l.split(' ')[:2]
                # im_index = int(im_name.split('_')[-1].split('.')[0])
                bin_name = depth_name.replace('sync_depth_', 'depth_bin_abs10_')
                # for abs
                write_line = f'train{im_name},train,228,304,228,304,train_abs_annot{bin_name}\n'
                # for rel
                # write_line = f'train{im_name},train,228,304,228,304,train_rel_annot/depth_bin_rel10_{im_index:05d}.png\n'
                f.write(write_line)


def make_sem_index_csv():
    fn = 'index.csv'
    with open(fn, 'w') as f:
        f.write('image,split,ih,iw,sh,sw,sem\n')
        for dataset in ['test', 'train']:
            if dataset == 'test':
                dir_name = 'dataset/nyuv2/images/test/'
            elif dataset == 'train':
                dir_name = 'dataset/nyuv2/images/train/'
            for rgb_name in recursive_glob(dir_name, '.jpg'):
                rgb_index = int(rgb_name.split('/')[-1].split('.')[0].split('_')[-1])
                write_line = f"{rgb_name[rgb_name.find(dataset):]},{dataset},228,304,228,304,{dataset}_sem_annot/new_nyu_class13_{rgb_index + 1:04d}.png\n"
                f.write(write_line)


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


def print_label_etc():
    # label.csv
    for i in range(10):
        print(f"{i + 14},abs-{i},abs(0),0,0,")
    for i in range(10):
        print(f"{i + 24},rel-{i},rel(0),0,0,")
    # c_abs/c_rel
    for i in range(10):
        print(f"{i},{i + 14},abs-{i},0,0")
    for i in range(10):
        print(f"{i},{i + 24},rel-{i},0,0")


def make_depth_bin_concept(dataset, abs_or_rel):
    """
    :param dataset: dense|bts_train|official
    :param abs_or_rel: abs|rel
    :return:
    """
    assert dataset in ['dense', 'bts_train', 'official']
    assert abs_or_rel in ['abs', 'rel']
    if dataset == 'dense':
        depth_dir = '/work/u2263506/nyu2_data/nyu2_dense' if torch.cuda.is_available() else '/Users/youzunzhi/pro/datasets/nyuv2_depth_data/nyu2_dense'
    else:
        raise NotImplementedError
    save_dir = f'../dataset/nyuv2/images/{dataset}_{abs_or_rel}_annot/'
    os.makedirs(save_dir, exist_ok=True)
    for depth_fname in recursive_glob(depth_dir, 'png'):
        if depth_fname.find('colors') != -1:  # in dense/test dir, rgb also end with png
            continue
        depth = get_meters_depth(depth_fname, dataset)
        labels = get_labels_sid(depth, abs_or_rel)
        save_fname = os.path.join(save_dir, depth_fname[len(depth_dir)+1:])
        dir_to_make = save_fname[:save_fname.find(save_fname.split('/')[-1])]
        os.makedirs(dir_to_make, exist_ok=True)
        if not cv2.imwrite(save_fname, labels):
            print('fail to save', save_fname)


def get_meters_depth(depth_fname, dataset):
    if dataset == 'dense':
        if depth_fname.find('train')!=-1:
            from torchvision.transforms import ToTensor
            depth = ToTensor()(Image.open(depth_fname)).numpy()[0]
            depth *= 10
        else:
            depth = np.array(Image.open(depth_fname)).astype(np.float)
            depth /= 1000.
    else:
        raise NotImplementedError

    return depth


def make_depth_bin_trash(relative):
    # testing set
    depth_dir = '/work/u2263506/nyu2_data/data/bts_splits/test' if torch.cuda.is_available() else '/Users/youzunzhi/pro/datasets/nyuv2_depth_data/bts_splits/test'
    if relative:
        save_dir = '../dataset/nyuv2/images/test_rel_annot/'
    else:
        save_dir = '../dataset/nyuv2/images/test_abs_annot/'
    os.makedirs(save_dir, exist_ok=True)
    dataset_file = '/home/u2263506/MDE_Dissect/data/nyudv2_test.txt' if torch.cuda.is_available() else '/Users/youzunzhi/pro/EVA/code/MDE_Dissect/data/nyudv2_test.txt'
    with open(dataset_file, 'r') as f:
        for l in f.readlines():
            depth_path = os.path.join(depth_dir, l.split()[1])
            depth = np.array(Image.open(depth_path)).astype(np.float)
            depth /= 1000.
            labels = get_labels_sid(depth, relative=relative, ordinal_c=10)
            save_fname = os.path.join(save_dir, depth_path.split('/')[-1])
            # Image.fromarray(labels).save(save_fname)
            cv2.imwrite(save_fname, labels)

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
            bin_name = (depth_path.split('/')[-1]).replace('sync_depth_', 'depth_bin_abs10_')
            save_fname = os.path.join(save_dir, bin_name)
            cv2.imwrite(save_fname, labels)


def get_labels_sid(depth, abs_or_rel, K=10.0):

    if abs_or_rel == 'abs':
        alpha = 1.0
        beta = 11.0
    elif abs_or_rel == 'rel':
        raise NotImplementedError
        # alpha = depth[depth != 0].min() + 0.999
        # beta = depth.max() + 1.
    else:
        raise NotImplementedError


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
    make_depth_bin_concept('dense', 'abs')