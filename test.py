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

# Model parameters
# List of channels / classes (tissue): 0. Background, 1. CSF, 2. Grey Matter, 3. White Matter, 4. Fat, 5. Muscle
tissue = np.array([0, 1, 2, 3, 4, 5])
img_size = (256, 256, len(tissue))

# Parameters for the model testing
test_batch_size = 1
save_file = True
real_data = True

# Test with pre-trained weights
model_weights = dir_code + '/results/archive_30_210525/weights/weights.036--1.9759.h5'


# Test model with synthetic data
def synth_test_seg(test_result_path, test_data_path, test_mask_path, tissue, img_size, test_batch_size, save_file, model_weights):
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
    test_files, ntest = calc_generator_info(test_data_path, test_batch_size)
    print('INFO: Test size: %d, Number of batches: %d' % (len(test_files), ntest))

    # Iterate through the files to be segmented
    for x_test, y_test, fname in img_generator_oai(test_data_path, test_mask_path, test_batch_size, img_size, tissue,
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
        if len(tissue) == 1:
            x_test = np.reshape(x_test, (256, 256))
            y_test = np.reshape(y_test, (256, 256))
            segmentation = np.reshape(prediction, (256, 256))
            segmentation = np.around(segmentation)

            # Save real image, real mask, and predicted mask
            if save_file is True:
                save_name = '%s/%s.jpg' %(test_result_path,fname)
                if img_cnt < 20:
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(x_test, cmap='gray')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(y_test, cmap='gray')
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
                    plt.imshow(segmentation, cmap='gray')
                    plt.axis('off')
                    plt.savefig(save_name, bbox_inches='tight')
                    plt.close()
        else:
            x_test = np.reshape(x_test, img_size)
            out_image = x_test[..., 0]

            # Create real masks
            y_test = np.reshape(y_test, img_size)
            out_mask = np.zeros((img_size[0], img_size[1]))
            for n in range(0, len(tissue)):
                out_mask[np.where(y_test[..., n] > 0)] = tissue[n]

            # Create predicted masks - comparing the multiple masks outputted by the model
            # and taking the class with the highest probability for each pixel
            segmentation = np.reshape(prediction, img_size)
            out_seg = np.zeros((img_size[0], img_size[1]))
            for i in range(0, img_size[0]):
                for j in range(0, img_size[1]):
                    out_seg[i,j] = tissue[np.argmax(segmentation[i,j,:])]

            # Save real image, real mask, and predicted mask
            if save_file is True:
                save_name = '%s/%s.jpg' %(test_result_path,fname)
                if img_cnt < 20:
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(out_image, cmap='gray')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(out_mask, cmap='gray')
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
                    plt.imshow(out_seg, cmap='gray')
                    plt.axis('off')
                    plt.savefig(save_name, bbox_inches='tight')
                    plt.close()


            #with h5py.File(save_name,'w') as h5f:
            #    h5f.create_dataset('prediction',data=prediction)

        img_cnt += 1
        if img_cnt == ntest:
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


# Test model with real data
def real_test_seg(test_result_path, test_data_path, tissue, img_size, test_batch_size, save_file, model_weights):
    img_cnt = 0

    # Set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')

    # Create the unet model
    model = unet_2d_model(img_size, num_classes=len(tissue))
    model.load_weights(model_weights)

    start = time.time()

    # Read the files that will be segmented
    test_files, ntest = calc_generator_info(test_data_path, test_batch_size)
    print('INFO: Test size: %d, Number of batches: %d' % (len(test_files), ntest))

    for x_test, fname in img_generator_oai(test_data_path, '', test_batch_size, img_size, tissue,
                                           testing=True, shuffle_epoch=True, real_data=True):

        # Perform the actual segmentation using pre-loaded model
        prediction = model.predict(x_test, batch_size=test_batch_size)

        if len(tissue) == 1:
            x_test = np.reshape(x_test, (256, 256))
            segmentation = np.reshape(prediction, (256, 256))
            segmentation = np.around(segmentation)
            # Save real image and predicted mask
            if save_file is True:
                save_name = '%s/%s.jpg' % (test_result_path, fname)
                if img_cnt < 20:
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(x_test, cmap='gray')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(segmentation, cmap='gray')
                    plt.axis('off')
                    plt.savefig(save_name, bbox_inches='tight')
                    plt.close()


        else:
            x_test = np.reshape(x_test, img_size)
            out_image = x_test[..., 0]

            segmentation = np.reshape(prediction, img_size)
            out_seg = np.zeros((img_size[0], img_size[1]))
            for i in range(0, img_size[0]):
                for j in range(0, img_size[1]):
                    out_seg[i, j] = tissue[np.argmax(segmentation[i, j, :])]

            # Save real image and predicted mask
            if save_file is True:
                save_name = '%s/%s.jpg' % (test_result_path, fname)
                if img_cnt < 20:
                    plt.figure()
                    plt.subplot(1, 3, 1)
                    plt.imshow(out_image, cmap='gray')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(out_seg, cmap='gray')
                    plt.axis('off')
                    plt.savefig(save_name, bbox_inches='tight')
                    plt.close()

        img_cnt += 1
        if img_cnt == ntest:
            break


if __name__ == '__main__':

    if real_data:
        real_test_seg(dir_results, dir_test_data, tissue, img_size, test_batch_size, save_file, model_weights)
    else:
        synth_test_seg(dir_results, dir_test_data, dir_test_masks, tissue, img_size, test_batch_size, save_file, model_weights)
