# Test the performance of the model
# Test using synthetic or real MPRAGE data

from __future__ import print_function, division

import numpy as np
import h5py
import time
import os
import tensorflow as tf
from keras import backend as K

from utils.generator import calc_generator_info, img_generator_oai
from utils.model import unet_2d_model
from utils.losses import dice_loss_test
import utils.utils_performance as segutils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Specify directories
dir_code = os.getcwd()
dir_results = dir_code + '/results/predictions'
dir_test_data = dir_code + '/dataset/real_test_images'
dir_test_masks = dir_code + '/dataset/test_masks'

# Define colour space
Background = [0, 0, 0]          # black
CSF = [0, 0, 128]               # blue
GreyMatter = [128, 0, 0]        # red
WhiteMatter = [128, 128, 0]     # yellow
Fat = [0, 128, 0]               # green
Muscle = [128, 0, 128]          # pink

COLOUR_DICT = np.array([Background, CSF, GreyMatter, WhiteMatter, Fat, Muscle])

# Parameters for model testing
# List of channels / classes (tissue): 0. Background, 1. CSF, 2. Grey Matter, 3. White Matter, 4. Fat, 5. Muscle
tissue = np.array([0, 1, 2, 3])         # TODO
test_batch_size = 1
real_data = True                        # TODO
save_file = True                        # TODO
max_im_save = 20

# Test with pre-trained weights
model_weights = dir_code + '/results/archive_27_210524/weights/weights.020--1.9888.h5'      # TODO
img_size = (256, 256, len(tissue))


# Add legend to plot
def colour_artist():
    handle = []
    classes = ['Background', 'CSF', 'Grey Matter', 'White Matter', 'Fat', 'Muscle']
    for i in range(1, len(tissue)):
        patch = mpatches.Patch(color=COLOUR_DICT[i]/128, label=classes[i])
        handle.append(patch)

    return handle


# Function to save predictions as images
def save_predictions(seg, x, y=np.array([0]), save_path='', real_data=False, show=False):
    if real_data:
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(x, cmap='gray')
        ax1.set_title('Real Image', fontweight="bold")
        plt.axis('off')

        ax2 = plt.subplot(1, 2, 2)
        plt.imshow((seg * 255).astype(np.uint8))
        ax2.set_title('Predicted Mask', fontweight="bold")
        plt.axis('off')

        plt.legend(handles=colour_artist(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    else:
        plt.figure()
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(x, cmap='gray')
        ax1.set_title('Synthetic Image', fontweight="bold")
        plt.axis('off')

        ax2 = plt.subplot(1, 3, 2)
        plt.imshow((y * 255).astype(np.uint8))
        ax2.set_title('True Mask', fontweight="bold")
        plt.axis('off')

        ax3 = plt.subplot(1, 3, 3)
        plt.imshow((seg * 255).astype(np.uint8))
        ax3.set_title('Predicted Mask', fontweight="bold")
        plt.axis('off')

        plt.legend(handles=colour_artist(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


# Function to segment predictions
def segment_predictions(prediction, image, mask=np.array([0]), num_classes=1, label=True):
    if label:
        if num_classes == 1:
            out_image = np.reshape(image, (256, 256))
            y = np.reshape(mask, (256, 256))
            segmentation = np.reshape(prediction, (256, 256))
            seg = np.around(segmentation)
        else:
            x = np.reshape(image, img_size)
            out_image = x[..., 0]

            # Create real masks
            y_test = np.reshape(mask, img_size)
            y = np.zeros((img_size[0], img_size[1]))
            for n in range(0, len(tissue)):
                y[np.where(y_test[..., n] > 0)] = tissue[n]

            # Create predicted masks - comparing the multiple masks outputted by the model
            # and taking the class with the highest probability for each pixel
            segmentation = np.reshape(prediction, img_size)
            seg = np.zeros((img_size[0], img_size[1]))
            for i in range(0, img_size[0]):
                for j in range(0, img_size[1]):
                    seg[i, j] = tissue[np.argmax(segmentation[i, j, :])]

        out_mask = np.zeros((img_size[0], img_size[1], 3))
        out_seg = np.zeros((img_size[0], img_size[1], 3))
        for n in range(0, len(tissue)):
            out_mask[np.where(y == n)] = COLOUR_DICT[n]
            out_seg[np.where(seg == n)] = COLOUR_DICT[n]

        return (out_image, out_mask, out_seg)

    else:
        if num_classes == 1:
            out_image = np.reshape(image, (256, 256))
            segmentation = np.reshape(prediction, (256, 256))
            seg = np.around(segmentation)
        else:
            x = np.reshape(image, img_size)
            out_image = x[..., 0]

            segmentation = np.reshape(prediction, img_size)
            seg = np.zeros((img_size[0], img_size[1]))
            for i in range(0, img_size[0]):
                for j in range(0, img_size[1]):
                    seg[i, j] = tissue[np.argmax(segmentation[i, j, :])]

        out_seg = np.zeros((img_size[0], img_size[1], 3))
        for n in range(0, len(tissue)):
            out_seg[np.where(seg == n)] = COLOUR_DICT[n]

        return (out_image, out_seg)


# Function to test the model with synthetic data
def synth_test(test_result_path, test_data_path, test_mask_path, tissue, img_size, test_batch_size, save_file, model_weights):
    img_cnt = 0

    # Set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')

    # Create the unet model
    model = unet_2d_model(img_size, num_classes=len(tissue))
    model.load_weights(model_weights)

    # All of the testing currently assumes that there is a reference to test against.
    # Comment out these lines if testing on reference-les data
    dice_losses = np.array([])
    cv_values = np.array([])
    voe_values = np.array([])
    vd_values = np.array([])

    start = time.time()

    # Read the files that will be segmented 
    test_files, num_test = calc_generator_info(test_data_path, test_batch_size)
    print('INFO: Test size: %d, Number of batches: %d' % (len(test_files), num_test))

    # Iterate through the files to be segmented
    for x_test, y_test, file_name in img_generator_oai(test_data_path, test_mask_path, test_batch_size, img_size, tissue,
                                                       testing=True, shuffle_epoch=True, real_data=False):

        # Perform the actual segmentation using pre-loaded model
        prediction = model.predict(x_test, batch_size=test_batch_size)

        # Calculate real time metrics
        dl = np.mean(segutils.calc_dice(y_test, prediction))
        dice_losses = np.append(dice_losses,dl)

        cv = np.mean(segutils.calc_cv(prediction,y_test))
        cv_values = np.append(cv_values,cv)

        voe = np.mean(segutils.calc_voe(y_test, prediction))
        voe_values = np.append(voe_values,voe)

        vd = np.mean(segutils.calc_vd(y_test, prediction))
        vd_values = np.append(vd_values,vd)

        # print('Image #%0.2d (%s). Dice = %0.3f CV = %2.1f VOE = %2.1f VD = %2.1f' % ( img_cnt, fname[0:11], dl, cv, voe, vd) )

        # Segment prediction
        num_classes = len(tissue)
        image, mask, segmentation = segment_predictions(prediction, x_test, y_test, num_classes)

        # Save real image, real mask, and predicted mask
        if save_file is True and img_cnt < max_im_save:
            save_name = '%s/%s.jpg' %(test_result_path, file_name)
            save_predictions(segmentation, image, y=mask, save_path=save_name)

        img_cnt += 1
        if img_cnt == num_test:
            break

    end = time.time()

    # Print some summary statistics
    print('--' * 20)
    print('Overall Summary:')
    print('Dice Mean= %0.4f, Std = %0.3f' % (np.mean(dice_losses), np.std(dice_losses)))
    print('CV Mean= %0.4f, Std = %0.3f' % (np.mean(cv_values), np.std(cv_values)))
    print('VOE Mean= %0.4f, Std = %0.3f' % (np.mean(voe_values), np.std(voe_values)))
    print('VD Mean= %0.4f, Std = %0.3f' % (np.mean(vd_values), np.std(vd_values)))
    print('Time required = %0.1f seconds.' % (end - start))
    print('--' * 20)


# Function to test model with real data
def real_test(test_result_path, test_data_path, tissue, img_size, test_batch_size, save_file, model_weights):
    img_cnt = 0

    # Set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')

    # Create the unet model
    model = unet_2d_model(img_size, num_classes=len(tissue))
    model.load_weights(model_weights)


    # Read the files that will be segmented
    test_files, num_test = calc_generator_info(test_data_path, test_batch_size)
    print('INFO: Test size: %d, Number of batches: %d' % (len(test_files), num_test))

    # Iterate through the files to be segmented
    for x_test, file_name in img_generator_oai(test_data_path, '', test_batch_size, img_size, tissue,
                                               testing=True, shuffle_epoch=True, real_data=True):

        # Perform the actual segmentation using pre-loaded model
        prediction = model.predict(x_test, batch_size=test_batch_size)

        # Segment prediction
        image, segmentation = segment_predictions(prediction, x_test, num_classes=len(tissue), label=False)

        # Save real image and predicted mask
        if save_file is True and img_cnt < max_im_save:
            save_name = '%s/%s.jpg' % (test_result_path, file_name)
            save_predictions(segmentation, image, save_path=save_name, real_data=True)

        img_cnt += 1
        if img_cnt == num_test:
            break


if __name__ == '__main__':

    if real_data:
        real_test(dir_results, dir_test_data, tissue, img_size, test_batch_size, save_file, model_weights)
    else:
        synth_test(dir_results, dir_test_data, dir_test_masks, tissue, img_size, test_batch_size, save_file,
                   model_weights)
