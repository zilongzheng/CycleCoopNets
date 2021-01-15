import os
from PIL import Image

datapath = './edges2shoes'
ori_train = os.path.join(datapath, 'train')
ori_test = os.path.join(datapath, 'val')

trainA = os.path.join(datapath, 'trainA')
trainB = os.path.join(datapath, 'trainB')
testA = os.path.join(datapath, 'testA')
testB = os.path.join(datapath, 'testB')
greyA = True


if not os.path.exists(trainA):
    os.makedirs(trainA)

if not os.path.exists(trainB):
    os.makedirs(trainB)

if not os.path.exists(testA):
    os.makedirs(testA)

if not os.path.exists(testB):
    os.makedirs(testB)

train_imgs = os.listdir(ori_train)
test_imgs = os.listdir(ori_test)


def img_split(ori_path, pathA, pathB, greyA=False):
    im = Image.open(ori_path)
    im1 = im.crop((0, 0, 255, 255))
    if greyA:
        im1 = im1.convert('LA')
    im2 = im.crop((256, 0, 511, 255))
    im1.save(pathA)
    im2.save(pathB)

def RGB2Grey(path):
    im = Image.open(path).convert('LA')
    im.save(path)

import time
start_time = time.time()
for i in range(len(train_imgs)):
    if i % 1000 == 0:
        print(i, time.time() - start_time)
    cur_ori_path = os.path.join(ori_train, train_imgs[i])
    cur_pathA = os.path.join(trainA, '{}.png'.format(i))
    cur_pathB = os.path.join(trainB, '{}.png'.format(i))
    img_split(cur_ori_path, cur_pathA, cur_pathB, greyA)

for i in range(len(test_imgs)):
    cur_ori_path = os.path.join(ori_test, test_imgs[i])
    cur_pathA = os.path.join(testA, '{}.png'.format(i))
    cur_pathB = os.path.join(testB, '{}.png'.format(i))
    img_split(cur_ori_path, cur_pathA, cur_pathB, greyA)

'''
imgs = os.listdir(trainA)
for path in imgs:
    RGB2Grey(os.path.join(trainA, path))

imgs = os.listdir(testA)
for path in imgs:
    RGB2Grey(os.path.join(testA, path))
'''