from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
from PIL import Image
# import scipy.misc
from skimage.measure import compare_psnr, compare_ssim
# import h5py
from six.moves import xrange

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def read_images_from_folder(data_path, img_width=224, img_height=224, low=-1, high=1):
    img_list = sorted([f for f in os.listdir(data_path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)])
    images = np.zeros(shape=(len(img_list), img_width, img_height, 3))
    print('Reading images from: {}'.format(data_path))
    for i in xrange(len(img_list)):
        image = Image.open(os.path.join(data_path, img_list[i])).convert('RGB')
        image = image.resize((img_width, img_height))
        image = np.asarray(image, dtype=float)
        cmax = image.max()
        cmin = image.min()
        image = (image - cmin) / (cmax - cmin) * (high - low) + low
        images[i] = image
    print('Images loaded, shape: {}'.format(images.shape))
    return images


def mkdir(path, exist_ok=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not exist_ok:
        os.removedirs(path)
        os.makedirs(path)

def cell2img(cell_image, image_size=100, margin_syn=2):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    num_imgs = 0
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            if temp.max() == 0:
                break
            images[ir*num_cols+ic] = temp
            num_imgs = num_imgs + 1
    return images[:num_imgs]

def saveCellImages(filename, image_size=64):
    save_dir = filename[:-4]
    img_name = filename.rsplit('/')[-1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cellImg = np.asarray(Image.open(filename).convert('RGB'), dtype=np.float32)
    print('Read image {}, size: {}, range: [{:.1f}, {:.1f}]'.format(filename, cellImg.shape, cellImg.min(), cellImg.max()))

    images = cell2img(cellImg, image_size=image_size)

    for i in range(len(images)):
        scipy.misc.imsave(save_dir + '/' + img_name[:-4] + '_%03d.png' % i, images[i])

def cut_interpolates(filename, image_size=64, margin_syn=2):
    save_dir = filename[:-4]
    img_name = filename.rsplit('/')[-1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cell_image = np.asarray(Image.open(filename).convert('RGB'), dtype=np.float32)
    # print('Read image {}, size: {}, range: [{:.1f}, {:.1f}]'.format(filename, cellImg.shape, cellImg.min(), cellImg.max()))
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_rows, image_size, cell_image.shape[1], 3))
    for ir in range(num_rows):
        temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),:,:]
        images[ir] = temp
    for i in range(len(images)):
        scipy.misc.imsave(save_dir + '/' + img_name[:-4] + '_%03d.png' % i, images[i])

    return images

def eval_psnr(true_imgs, test_imgs, a_min=0.0, a_max=255.0):
    #true_imgs = get_cell_images(im_true, image_size=image_size)
    # test_imgs = get_cell_images(im_test, image_size=image_size)
    # print(test_imgs.shape)
    psnrs = []
    drange = a_max - a_min
    for i in range(len(true_imgs)):
        t_img = np.clip(true_imgs[i], a_min, a_max)
        f_img = np.clip(test_imgs[i], a_min, a_max)
        psnrs.append(compare_psnr(t_img, f_img, data_range=drange))
    return psnrs

def eval_mse(true_imgs, test_imgs, a_min=-1, a_max=1):
    #true_imgs = get_cell_images(im_true, image_size=image_size)
    # test_imgs = get_cell_images(im_test, image_size=image_size)
    # print(test_imgs.shape)
    mses = []
    drange = a_max - a_min
    for i in range(len(true_imgs)):
        t_img = np.clip(true_imgs[i], a_min, a_max)
        f_img = np.clip(test_imgs[i], a_min, a_max)
        mses.append(np.square(t_img - f_img).mean())
    return mses


def eval_ssim(true_imgs, test_imgs, a_min=0.0, a_max=255.0):
    ssims = []
    drange = a_max - a_min
    for i in range(len(true_imgs)):
        t_img = np.clip(true_imgs[i], a_min, a_max)
        f_img = np.clip(test_imgs[i], a_min, a_max)
        ssims.append(compare_ssim(t_img, f_img, data_range=drange, multichannel=True))
    return ssims


def get_mean_ssim(im_true, im_test, image_size=256):
    true_imgs = get_cell_images(im_true, image_size=image_size)
    test_imgs = get_cell_images(im_test, image_size=image_size)
    print(test_imgs.shape)
    mssim = np.zeros(shape=len(true_imgs), dtype=np.float32)
    drange = test_imgs.max() - test_imgs.min()
    for i in range(len(true_imgs)):
        mssim[i] = compare_ssim(true_imgs[i], test_imgs[i], data_range=drange, multichannel=True)
    return mssim.mean()


def clip_by_value(input_, low=0, high=1):
    return np.minimum(high, np.maximum(low, input_))

def numpy_to_image(np_arr):
    img = (np.squeeze(np_arr) + 1.) * 127.5
    img = np.clip(img, a_min=0., a_max=255.).astype(np.uint8)
    return img

def img2cell(images, row_num=10, col_num=10, low=-1, high=1):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cells = []
    for i_cell in range(num_cells):
        row_imgs = []
        for ir in range(row_num):
            col_imgs = []
            for ic in range(col_num):
                idx = i_cell * row_num * col_num + ir * col_num + ic
                col_imgs.append(numpy_to_image(images[idx]))
            row_imgs.append(np.concatenate(col_imgs, axis=1))
        cells.append(np.concatenate(row_imgs, axis=0))
    return cells

# def img2cell(images, row_num=10, col_num=10, low=-1, high=1, margin_syn=2):
#     [num_images, image_size] = images.shape[0:2]
#     num_cells = int(math.ceil(num_images / (col_num * row_num)))
#     cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
#                            col_num * image_size + (col_num-1)*margin_syn, 3))
#     for i in range(num_images):
#         cell_id = int(math.floor(i / (col_num * row_num)))
#         idx = i % (col_num * row_num)
#         ir = int(math.floor(idx / col_num))
#         ic = idx % col_num
#         temp = clip_by_value(np.squeeze(images[i]), low, high)
#         cmin = temp.min()
#         cmax = temp.max()
#         cscale = cmax - cmin
#         if cscale == 0:
#             cscale = 1
#         temp = (temp - cmin) / cscale
#         cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
#                     (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic,:] = temp
#     return cell_image

def saveSampleImages(sample_results, filename, row_num=10, col_num=10, margin_syn=2, save_all=False):
    if save_all:
        cell_image = img2cell(sample_results, row_num=row_num, col_num=col_num)
        for ci in range(len(cell_image)):
            Image.fromarray(cell_image[ci].astype(np.uint8)).save(filename[:-4] + '_%03d.png' % ci)
    else:
        cell_image = img2cell(sample_results[:(row_num * col_num)], row_num=row_num, col_num=col_num)
        Image.fromarray(cell_image[0].astype(np.uint8)).save(filename)

def get_cell_images(filename, image_size=256):
    def get_cell_img(cell_im):
        print('Read image: {}'.format(cell_im))
        cell = np.asarray(Image.open(cell_im).convert('RGB'), dtype=np.float32)
        imgs = cell2img(cell, image_size=image_size)
        return imgs
    if type(filename) == list:
        return np.concatenate([get_cell_img(im) for im in filename], axis=0)
    else:
        return get_cell_img(filename)


def saveFinalResults(filename, input_, gt, synthesis, row_num=10, margin_syn=2):
    [num_syn, num_images, image_size] = synthesis.shape[0:3]
    final_results = np.zeros(shape=((num_syn+2) * num_images, image_size, image_size, 3), dtype=np.float32)
    for i in range(len(final_results)):
        img_id = int(i / (num_syn + 2))
        if i % (num_syn+2) == 0:
            final_results[i] = input_[img_id]
        elif i % (num_syn+2) == 1:
            final_results[i] = gt[img_id]
        else:
            syn_id = int(i % (num_syn+2) - 2)
            final_results[i] = synthesis[syn_id, img_id, ...]
    cell_image = img2cell(final_results, row_num=row_num, col_num=(num_syn+2), margin_syn=margin_syn)
    for ci in range(len(cell_image)):
        scipy.misc.imsave(filename[:-4] + '_%03d.png' % ci, cell_image[ci])

def generate_pix2pix_dataset(src_img_path, tgt_img_path, output_dir, img_width=256, img_height=256):
    src_images = read_images_from_folder(src_img_path, img_width, img_height, 0, 1)
    tgt_images = read_images_from_folder(tgt_img_path, img_width, img_height, 0, 1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(src_images)):
        pair = np.concatenate([[tgt_images[i]], [src_images[i]]], axis=0)
        cell = img2cell(pair, row_num=1, col_num=2, low=0, high=1, margin_syn=0)
        scipy.misc.imsave( '%s/%d.jpg' % (output_dir, i+1), np.squeeze(cell))

