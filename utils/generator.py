# Data generator to train and test the model

from __future__ import print_function, division

import numpy as np
from os import listdir
from os.path import splitext
from random import shuffle
from re import split
import h5py
from PIL import Image


# Find unique data regardless of the file prefix
def calc_generator_info(data_path, batch_size):
    
    files = listdir(data_path)
    unique_filename = {}

    for file in files:
        file,_ = splitext(file)
        if not file in unique_filename:
            unique_filename[file] = file

    files = list(unique_filename.keys())
    num_files = len(files)
    batches_per_epoch = num_files // batch_size

    return (files, batches_per_epoch)


# Generate images, masks to be inputted into the model
def img_generator_oai(data_path, mask_path, batch_size, img_size, tissue, testing=False, shuffle_epoch=True, real_data=False):

    files, batches_per_epoch = calc_generator_info(data_path, batch_size)

    if not real_data:
        x = np.zeros((batch_size,)+img_size)
        y = np.zeros((batch_size,)+img_size)

        while True:

            if shuffle_epoch:
                shuffle(files)

            for batch_cnt in range(batches_per_epoch):
                for file_cnt in range(batch_size):

                    # Prepare image files
                    file_ind = batch_cnt*batch_size+file_cnt
                    im_path = '%s/%s.jpg'%(data_path, files[file_ind])
                    with Image.open(im_path, 'r') as f:
                        im = np.asarray(f)

                    # Prepare mask files
                    seg_path = '%s/%s.npy'%(mask_path, files[file_ind])
                    msk = np.load(seg_path)
                    seg = msk.copy()
                    seg[np.where(seg == 0)] = 255
                    seg[np.where(seg < 12)] = 0
                    seg = np.reshape(seg, (img_size[0], img_size[1], 1))

                    for n in tissue:
                        if n == 0:
                            continue
                        tmp = msk.copy()
                        tmp[np.where(tmp != n)] = 0
                        tmp[np.where(tmp == n)] = 255
                        tmp = np.reshape(tmp, (img_size[0], img_size[1], 1))

                        seg = np.concatenate((seg, tmp), axis=2)

                    seg = np.asarray(seg).astype('float32')

                    x[file_cnt, ..., 0] = im
                    y[file_cnt, ...] = seg
                    fname = files[file_ind]

                if testing is False:
                    yield (x, y)
                else:
                    fname = files[file_ind]
                    yield (x, y, fname)

    if real_data:
        x = np.zeros((batch_size,)+img_size)

        while True:

            if shuffle_epoch:
                shuffle(files)

            for batch_cnt in range(batches_per_epoch):
                for file_cnt in range(batch_size):

                    # Prepare image files
                    file_ind = batch_cnt * batch_size + file_cnt
                    im_path = '%s/%s.jpg' % (data_path, files[file_ind])
                    with Image.open(im_path, 'r') as f:
                        im = np.asarray(f)

                    x[file_cnt, ..., 0] = im
                    fname = files[file_ind]

                    yield (x, fname)


