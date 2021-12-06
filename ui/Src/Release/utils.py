"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import cv2

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

# def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
#     return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(src_images, syn_images, target_images, size, image_path):
    return imsave(inverse_transform(src_images), inverse_transform(syn_images), inverse_transform(target_images), size,
                  image_path)

def imread(path):
    return cv2.imread(path, -1).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(src_images, syn_images, target_images, size):
    h, w = src_images.shape[1], src_images.shape[2]
    img = np.zeros((h * size[0], w * size[1]*3, 3))
    for idx, image in enumerate(src_images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
        img[j * h: j * h + h, i * w + w: i * w + 2 * w, :] = syn_images[idx]
        img[j * h: j * h + h, i * w + 2 * w: i * w + 3 * w, :] = target_images[idx]

    return img

def imsave(src_images, syn_images, target_images, size, path):
    return cv2.imwrite(path, merge(src_images, syn_images, target_images, size))

# def transform(image, npx=64, is_crop=True, resize_w=64):
#     # npx : # of pixels width/height of image
#     if is_crop:
#         cropped_image = center_crop(image, npx, resize_w=resize_w)
#     else:
#         cropped_image = image
#     return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.*255

def load_batch_data(data_file, batch_size, img_dim):
    batch_data = []
    for i in range(0, batch_size):
        img1 = cv2.imread(data_file[i][0], -1)
        assert img1.shape[0]
        img1 = img1 / 127.5 - 1.

        img2 = cv2.imread(data_file[i][1], -1)
        assert img2.shape[0]
        img2 = img2 / 127.5 - 1.

        img3 = cv2.imread(data_file[i][2], -1)
        assert img3.shape[0]
        img3 = img3 / 127.5 - 1.

        img = np.concatenate((img1, img2, img3), -1)
        batch_data.append(img)

    return np.array(batch_data, dtype=np.float32).reshape((batch_size, 256, 256, img_dim*3))

def read_ply(fname, with_normal=True):
	point_set = []
	with open(fname) as f:
		i = 0
		for line in f.readlines():
			i += 1
			if i > 12:
				line = line.strip('\n')
				x, y, z, nx, ny, nz = line.split(" ")
				x = np.float32(x)
				y = np.float32(y)
				z = np.float32(z)
				nx = np.float32(nx)
				ny = np.float32(ny)
				nz = np.float32(nz)
				point_set.append([x, y, z, nx, ny, nz])
		f.close()
	assert (len(point_set)>0)

	point_set = np.array(point_set).reshape(len(point_set), 6)
	if not with_normal:
		point_set = point_set[:, 0:3]

	return point_set