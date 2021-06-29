import os
import math
import numpy as np
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def read_images_from_folder(data_path, img_width=224, img_height=224, low=-1, high=1):
    img_list = sorted([f for f in os.listdir(data_path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)])
    images = np.zeros(shape=(len(img_list), img_width, img_height, 3))
    print('Reading images from: {}'.format(data_path))
    for i in range(len(img_list)):
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

def clip_by_value(input_, low=0, high=1):
    return np.minimum(high, np.maximum(low, input_))

def normalize_image(img, mean=0.5, stddev=0.5):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img - mean ) / stddev
    return img

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

def saveSampleImages(sample_results, filename, row_num=10, col_num=10, margin_syn=2, save_all=False):
    file_dir = os.path.dirname(filename)
    os.makedirs(file_dir, exist_ok=True)
    if save_all:
        cell_image = img2cell(sample_results, row_num=row_num, col_num=col_num)
        for ci in range(len(cell_image)):
            Image.fromarray(cell_image[ci].astype(np.uint8)).save(filename[:-4] + '_%03d.png' % ci)
    else:
        cell_image = img2cell(sample_results[:(row_num * col_num)], row_num=row_num, col_num=col_num)
        Image.fromarray(cell_image[0].astype(np.uint8)).save(filename)
