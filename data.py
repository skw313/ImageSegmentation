import numpy as np
import os
from PIL import Image


def get_data(s):
    '''
    Function to prepare the images and labels for training, testing, validation
    Train, test, val data must be saved in separate folders
    INPUT: s (string) - specify which dataset to prepare (train, test, or val)
    OUTPUT: [x,y] (list) - list of image and label as numpy arrays
    '''
    dir_code = os.getcwd()
    dir_im = dir_code + '/dataset/' + s + '_images'
    dir_lbl = dir_code + '/dataset/' + s + '_masks'
    archive_im = os.listdir(dir_im)
    archive_lbl = os.listdir(dir_lbl)

    x = np.zeros((len(archive_im), 256, 256))  # TODO change image size
    y = np.zeros((len(archive_lbl), 256, 256))
    for f in range(0, len(archive_im)):
        # Open image and convert to numpy array
        image = Image.open(dir_im + '/' + archive_im[f])
        mask = Image.open(dir_lbl + '/' + archive_lbl[f])
        # Save image (array) into x/y
        x[f, :, :] = np.asarray(image)
        y[f, :, :] = np.asarray(mask)

    return [x, y]


# TODO write function that visualises a predicted mask (and the corresponding image and true mask)




