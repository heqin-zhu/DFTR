#!/usr/bin/python3
# coding=utf-8

import os
import random
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, depth):
        image = (image - self.mean) / self.std
        mask /= 255
        depth /= 255
        return image, mask, depth


class RandomCrop(object):
    def __call__(self, image, mask, depth):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], depth[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask, depth):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1], depth[:, ::-1]
        else:
            return image, mask, depth


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, depth):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, depth


class ToTensor(object):
    def __call__(self, image, mask, depth):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        depth = torch.from_numpy(depth)
        return image, mask, depth


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # for rgb DUT
        # self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        # self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        #
        # for rgbd sod
        # self.mean = np.array([[[112.03, 108.56, 100.15]]])
        # self.std = np.array([[[55.86, 53.71, 53.99]]])
        # for rgbd sod 0810
        self.mean = np.array([[[112.77, 109.66, 102.11]]])
        self.std = np.array([[[55.45, 53.60, 53.86]]])
        # for imagenet
        #self.mean = np.array([[[123.67, 116.28, 103.53]]])
        #self.std = np.array([[[58.39, 57.12, 57.37]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class SODData(Dataset):
    def __init__(self, datapath, data_names, mode, img_size=256, val_test=False):
        self.mode = mode
        self.img_size = img_size
        self.datapath = datapath
        assert isinstance(data_names, list)
        self.data_names = data_names

        mean = np.array([[[112.77, 109.66, 102.11]]])
        std = np.array([[[55.45, 53.60, 53.86]]])
        self.normalize = Normalize(mean=mean, std=std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(img_size, img_size)
        self.totensor = ToTensor()

        self.samples= self.read_dir(self.datapath)
        self.samples.sort()
        random.shuffle(self.samples)

        train_num = round(len(self.samples)*0.9)
        
        if mode=='train':
            if val_test:
                pass
            else:
                self.samples = self.samples[:train_num]
        elif mode=='validate':
            if val_test:
                self.data_names = [i.replace('train', 'test') for i in data_names]
                self.samples = self.read_dir(self.datapath)
            else:
                self.samples = self.samples[train_num:]

        self.rgb_suf = '.jpg'
        self.mask_suf = '.png'


    def read_dir(self, path):
        samples = []
        for idx, name in enumerate(self.data_names):
            index_dir = os.path.join(path, name, 'depth')
            if not os.path.exists(index_dir):
                index_dir = os.path.join(path, name, 'GT')

            samples +=[(i[:-4], idx) for i in os.listdir(index_dir)]
        return samples


    def __getitem__(self, i):
        name, idx = self.samples[i]
        dataset_name = self.data_names[idx]
        cur_path = os.path.join(self.datapath, dataset_name)
        image = cv2.imread(cur_path + '/RGB/' + name + self.rgb_suf)[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(cur_path + '/GT/' + name + self.mask_suf, 0).astype(np.float32)
        shape = mask.shape

        if self.mode == 'train' or self.mode=='validate':
            depth = cv2.imread(cur_path + '/depth/' + name + '.png', 0).astype(np.float32)
            image, mask, depth = self.normalize(image, mask, depth)
            image, mask, depth = self.randomcrop(image, mask, depth)
            image, mask, depth = self.randomflip(image, mask, depth)
            image = np.ascontiguousarray(image) # solve error caused by [:::-1]
            mask = np.ascontiguousarray(mask) # solve error caused by [:::-1]
            depth = np.ascontiguousarray(depth) # solve error caused by [:::-1]
            return image, mask, depth
        else:
            depth=mask.copy() # useless
            image, mask, depth = self.normalize(image, mask, depth)
            image, mask, depth = self.resize(image, mask, depth)
            image, mask, depth = self.totensor(image, mask, depth)
            image = np.ascontiguousarray(image) # solve error caused by [:::-1]
            mask = np.ascontiguousarray(mask) # solve error caused by [:::-1]
            depth = np.ascontiguousarray(depth) # solve error caused by [:::-1]
            return image, mask, shape, name
    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, depth = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i] = cv2.resize(depth[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        depth = torch.from_numpy(np.stack(depth, axis=0)).unsqueeze(1)
        return image, mask, depth

    def __len__(self):
        return len(self.samples)
