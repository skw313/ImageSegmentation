import os
import re
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

dir_code = os.getcwd()
dir_parent = os.path.dirname(dir_code)

dir_data_mat = dir_parent + '/DataAugmentation/data_mat'
dir_label_mat = dir_parent + '/DataAugmentation/data_aug'
archive_data = os.listdir(dir_data_mat)
archive_label = os.listdir(dir_label_mat)

# Prepare array that saves indices of random images extracted for dataset
random.seed(1)  # comment out for "true" randomness
n = 1           # TODO change number of images extracted per .mat file
idx = np.zeros((len(archive_data), n), dtype=int)

# Extract data from the files from data augmentation part
for f in range(0, len(archive_data)):
    file_img = io.loadmat(dir_data_mat + '/' + archive_data[f])
    file_lbl = io.loadmat(dir_label_mat + '/' + archive_label[f])

    # Files are a dictionaries - extract the actual data from it
    data_img = file_img['im_intensity']
    data_lbl = file_lbl['atlas_aug']

    # Extract n random images and labels from MATLAB file
    for i in range(0, n):
        idx[f, i] = random.randint(0, data_img.shape[0] - 1)
    for i in idx[f]:
        image = np.reshape(data_img[i, :, :], (256, 256))
        image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
        label = np.reshape(data_lbl[i, :, :], (256, 256))
        label = (255.0 / label.max() * (label - label.min())).astype(np.uint8)
        # Uncomment to show images
        #plt.imshow(image)
        #plt.imshow(label)
        #plt.show()

        # Save image and label in tmp folders
        image_name = dir_code + '/dataset/tmp_images/image' + str(len(os.listdir(dir_code + '/dataset/tmp_images/')) + 1) + '.jpg'
        im = Image.fromarray(image, mode='L')
        im.save(image_name)

        mask_name = dir_code + '/dataset/tmp_masks/image' + str(len(os.listdir(dir_code + '/dataset/tmp_masks/')) + 1) + '.jpg'
        msk = Image.fromarray(label, mode='L')
        msk.save(mask_name)


# Shuffle data
dir_images = dir_code + '/dataset/tmp_images'
dir_masks = dir_code + '/dataset/tmp_masks'
all_images = os.listdir(dir_images)
all_masks = os.listdir(dir_masks)

all_images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
random.seed(230)
random.shuffle(all_images)


# Split data into train, validation, test set (ratio: 70, 20, 10)
train_split = int(0.7 * len(all_images))
val_split = int(0.9 * len(all_images))

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

# Generate corresponding mask lists for masks
train_masks = [f for f in all_masks if f in train_images]
val_masks = [f for f in all_masks if f in val_images]
test_masks = [f for f in all_masks if f in test_images]

image_folders = [(train_images, 'train_images'), (val_images, 'val_images'), (test_images, 'test_images')]
mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), (test_masks, 'test_masks')]


### Functions to add images and masks ###
def add_frames(dir_name, img):
    im = Image.open(dir_images + '/' + img)
    im_name = dir_code + '/dataset' + '/{}'.format(dir_name) + '/' + img
    im.save(im_name)


def add_masks(dir_name, msk):
    im = Image.open(dir_masks + '/' + msk)
    im_name = dir_code + '/dataset' + '/{}'.format(dir_name) + '/' + msk
    im.save(im_name)


# Add images and labels into corresponding folders
for folder in image_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames, name, array))

# Add masks
for folder in mask_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_masks, name, array))


### Delete all images -- comment out for final dataset###
def delete():
    for folder in os.listdir(dir_code + '/dataset'):
        for file in os.listdir(dir_code + '/dataset/' + folder):
            os.remove(dir_code + '/dataset/' + folder + '/' + file)
