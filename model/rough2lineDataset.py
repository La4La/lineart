#!/usr/bin/env python

import numpy as np
import chainer
'''
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
'''
import six
import os

from chainer import cuda, optimizers, serializers, Variable
import cv2


class Rough2LineDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./terget', dtype=np.float32, leak=(0, 0), root_ref = None, train=False):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._img_dict = {}
        self._train = train

    def set_img_dict(self, img_dict):
        self._img_dict = img_dict

    def get_vec(self, name):
        tag_size = 1539
        v = np.zeros(tag_size).astype(np.int32)
        if name in self._img_dict.keys():
            for i in self._img_dict[name][3]:
                v[i] = 1
        return v

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9

        readed = False
        if np.random.rand() < bin_r:
            if np.random.rand() < 0.3:
                path1 = os.path.join(self._root1 + "_rough/", self._paths[i])
            else:
                path1 = os.path.join(self._root1 + "_line/", self._paths[i])
            path2 = os.path.join(self._root2 + "_rough/", self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            if image1 is not None and image2 is not None:
                if image1.shape[0] > 0 and image1.shape[1] and image2.shape[0] > 0 and image2.shape[1]:
                    readed = True
        if not readed:
            path1 = os.path.join(self._root1, self._paths[i])
            path2 = os.path.join(self._root2, self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # input image size: 384*384

        # randomly down sampling
        if self._train:
            scale = np.random.choice(range(6,15)) / 6.0
            if image1.shape[0] // scale >= 384 and image1.shape[1] // scale >= 384:
                image1 = cv2.resize(image1, (image1.shape[0] // scale, image1.shape[1] // scale))
                image2 = cv2.resize(image2, (image2.shape[0] // scale, image2.shape[1] // scale))
            elif image1.shape[0] // scale < 384:
                image1 = cv2.resize(image1, (384, int(image1.shape[1] / image1.shape[0] * 384)))
                image2 = cv2.resize(image2, (384, int(image2.shape[1] / image2.shape[0] * 384)))
            elif image1.shape[1] // scale < 384:
                image1 = cv2.resize(image1, (int(image1.shape[0] / image1.shape[1] * 384), 384))
                image2 = cv2.resize(image2, (int(image2.shape[0] / image2.shape[1] * 384), 384))

        # randomly crop
        if self._train:
            x = np.random.randint(0, image1.shape[1] - 383)
            y = np.random.randint(0, image1.shape[0] - 383)
            image1 = image1[y:y+384, x:x+384]
            image2 = image2[y:y+384, x:x+384]

        # add flip
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.9:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2, self._dtype)

        # replace rough sketches with line arts
        if self._train:
            if np.random.rand() > 0.9:
                image1 = image2

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))

        return image1, image2

