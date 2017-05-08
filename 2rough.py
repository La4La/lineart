# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
np.set_printoptions(threshold=np.inf)

def aff(img):
    rows, cols = img.shape
    img_out = img / 255.0
    a = random.randint(1, 3)
    for i in range(a):
        img_cp = 1.0 - img / 255.0
        # affine transformation
        rad = np.pi * random.uniform(-0.03, 0.03)
        if random.random() > 0.7:
            move_x = random.uniform(-1.5,1.5)
        else:
            move_x = 0
        if random.random() > 0.7:
            move_y = random.uniform(-1.5,1.5)
        else:
            move_y = 0
        afn_M = np.float32([[np.cos(rad), -1 * np.sin(rad), move_x],
                            [np.sin(rad), np.cos(rad), move_y]])
        img_cp = cv2.warpAffine(img_cp, afn_M, (cols, rows), flags=cv2.INTER_LINEAR)
        img_out = img_out - img_cp
    img_out = np.where(img_out < 0, -img_out, img_out)
    return img_out


def addPointNoise(img):
    rows, cols = img.shape
    row = random.randint(0, rows - 1)
    col = random.randint(0, cols - 1)
    for j in range(random.randint(4, 25)):
        img[row, col] = 0
        patern = random.randint(1, 8)
        if patern == 1:
            row = min(row + 1, rows - 1)
        elif patern == 2:
            col = min(col + 1, cols - 1)
        elif patern == 3:
            col = min(col + 1, cols - 1)
            row = min(row + 1, rows - 1)
        elif patern == 4:
            row = max(row - 1, 0)
        elif patern == 5:
            col = max(col - 1, 0)
        elif patern == 6:
            row = max(row - 1, 0)
            col = max(col - 1, 0)
        elif patern == 7:
            row = min(row + 1, rows - 1)
            col = max(col - 1, 0)
        elif patern == 8:
            row = max(row - 1, 0)
            col = min(col + 1, cols - 1)


def addGaussianNoise(img):
    rows, cols = img.shape
    mean = 0
    sigma = random.randint(2,8)
    gauss = np.random.normal(mean, sigma, (rows, cols))
    gauss = gauss.reshape(rows, cols)
    gauss = np.where(gauss > 0, gauss, -gauss).astype(np.uint8)
    noisy = np.where(img < 10, img + gauss, img - gauss)
    return noisy




img = cv2.imread('line2.jpg', 0)
rows, cols = img.shape
zero = np.zeros(img.shape)

# apply affine transformation
itv = random.randint(25,35) #interval
i_n = rows//itv
j_n = cols//itv
for i in range(i_n-1):
    for j in range(j_n-1):
        zero[i*itv:(i+1)*itv, j*itv:(j+1)*itv] = aff(img[i*itv:(i+1)*itv, j*itv:(j+1)*itv])
    zero[i*itv:(i+1)*itv, (j_n-1)*itv:cols] = aff(img[i*itv:(i+1)*itv, (j_n-1)*itv:cols])
for j in range(j_n):
    zero[(i_n-1)*itv:rows, j*itv:(j+1)*itv] = aff(img[(i_n-1)*itv:rows, j*itv:(j+1)*itv])
zero[(i_n-1)*itv:rows, (j_n-1)*itv:cols] = aff(img[(i_n-1)*itv:rows, (j_n-1)*itv:cols])

# add point noise
for i in range(random.randint(30,50)):
    addPointNoise(zero)

# reduce contrast
min_table = 125
max_table = 255
diff_table = max_table - min_table
look_up_table = np.arange(256, dtype='uint8')
for i in range(0, 255):
    look_up_table[i] = min_table + i * (diff_table) / 255
img = cv2.LUT((zero * 255.0).astype(np.uint8), look_up_table)

# add Gaussian Noise
img = addGaussianNoise(img)

# blur
k_size = 3#random.randint(0, 1) * 2 + 1
out = cv2.GaussianBlur(img, (k_size, k_size), 0)

cv2.imwrite('img_out.jpg', out)