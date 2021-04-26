import numpy as np
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOUR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

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


def save_prediction(prediction, data, label, num_classes):
    '''
    Function to save image, label, and prediction as a subplot
    INPUT: prediction (array) - the predicted mask
           data (array) - the corresponding input images
           label (array) - the true masks
           num_classes (int) - number of classes we segmented
    '''
    if num_classes == 1:
        for i in range(0, 20):
            input_data = data[i]
            truth_data = label[i]
            segmentation = np.reshape(prediction[i, :, :], (256, 256))

            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(input_data, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(truth_data, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(segmentation, cmap='gray')
            plt.axis('off')
            plt.savefig("results/predictions/predict_%d.jpg" % i, bbox_inches='tight')
            plt.close()

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.01)
