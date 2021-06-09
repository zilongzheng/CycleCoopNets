from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import numpy as np
from PIL import Image
import scipy.misc
# from skimage.measure import compare_psnr, compare_ssim
import h5py
from six.moves import xrange

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class UnalignedDataLoader(object):
    def __init__(self, dir_A, dir_B, cache=None, load_size=286, crop_size=256, max_dataset_size='inf', shuffle=False, serial_batches=False, no_flip=False):

        self.serial_batches = serial_batches
        self.dir_A = dir_A
        self.dir_B = dir_B

        self.A_paths = make_dataset(self.dir_A, max_dataset_size)
        self.B_paths = make_dataset(self.dir_B, max_dataset_size)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.load_size = load_size
        self.crop_size = crop_size
        self.no_flip = no_flip

        # if cache != None and os.path.exists(cache):
        #     with h5py.File(cache, 'r') as h5:
        #         self.images_A = np.asarray(h5['images_A'], dtype=np.float32)
        #         self.images_B = np.asarray(h5['images_B'], dtype=np.float32)
        # else:
        #     print('Preloading images...')
        #     self.images_A = preload_images(self.A_paths, load_size, load_size)
        #     self.images_B = preload_images(self.B_paths, load_size, load_size)
        #     if cache != None:
        #         print('Saving image cache to %s' % cache)
        #         with h5py.File(cache, 'w') as h5:
        #             h5.create_dataset('images_A', data=self.images_A)
        #             h5.create_dataset('images_B', data=self.images_B)
        # self.images_A = self.images_A[:self.A_size]
        # self.images_B = self.images_B[:self.B_size]
        # print('Loaded images, images A: {}, images B: {}'.format(self.images_A.shape, self.images_B.shape))
        # print(self.A_size, self.B_size)
        self.num_images = max(self.A_size, self.B_size)

        if shuffle:
            self.indices = np.random.permutation(self.num_images)
        else:
            self.indices = np.arange(self.num_images)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def _process_images(self, image_list):
        # print(type(image_list))
        if isinstance(image_list, np.str):
            image_list = [image_list]
        images = []
        for img_path in image_list:
            images.append(preprocess_image(Image.open(img_path).convert(
                'RGB'), self.load_size, self.crop_size, do_flip=not self.no_flip))
        return np.stack(images)

    def sample_image_pair(self):
        img_A = Image.open(self.A_paths[np.random.randint(0, self.A_size)]).convert('RGB')

        img_B = Image.open(self.B_paths[np.random.randint(0, self.B_size)]).convert('RGB')
        return img_A, img_B

    def __getitem__(self, index):
        """
        Args:
            index: np.array of batch indices
        Returns:
            img_A: batch of images in domain A
            img_B: batch of images in domain B
        """
        img_A = self.A_paths[self.indices[index] % self.A_size]
        if self.serial_batches:
            B_inds = self.indices[index] % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs
            B_inds = np.random.randint(0, self.B_size, size=len(img_A))
        img_B = self.B_paths[B_inds]
        return self._process_images(img_A), self._process_images(img_B)

    def __len__(self):
        return self.num_images


def preprocess_image(input_image, load_size, crop_size, do_flip=True):
    img = input_image.resize((load_size, load_size), Image.BICUBIC)
    if load_size > crop_size:
        crop_p = np.random.randint(0, load_size - crop_size, size=(2,))
        img = img.crop((crop_p[0], crop_p[1], crop_p[0] +
                        crop_size, crop_p[1] + crop_size))
    img = np.asarray(img, dtype=np.float32)
    # normalize to [-1, 1]
    img = (img / 127.5) - 1.0
    if do_flip and random.random() > 0.5:
        img = np.fliplr(img)
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images.sort()
    return np.array(images[:int(min(float(max_dataset_size), len(images)))])


def preload_images(paths, img_width=256, img_height=256, img_channel=3, mean=0.5, stddev=0.5):

    images = np.zeros((len(paths), img_width, img_height,
                       img_channel), dtype=np.float32)
    for i, img_path in enumerate(paths):
        img = Image.open(img_path).convert('RGB').resize(
            (img_width, img_height), Image.BICUBIC)
        img = np.array(img).astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img - mean) / stddev

        images[i] = img
    return images
