# Train the model on (synthetic) dataset

from __future__ import print_function, division

import numpy as np
import pickle
import math
import os
import tensorflow as tf

from keras.optimizers import Adam
from keras import backend as K
import keras.callbacks as kc  

from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import LambdaCallback as lcb
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import TensorBoard as tfb

from utils.generator import calc_generator_info, img_generator_oai
from utils.model import unet_2d_model
from utils.losses import dice_loss

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

# Training and validation data locations
dir_code = os.getcwd()
train_path = dir_code + '/dataset/train_images'
train_mask_path = dir_code + '/dataset/train_masks'
valid_path = dir_code + '/dataset/val_images'
valid_mask_path = dir_code + '/dataset/val_masks'
test_path = dir_code + '/dataset/test_images'
test_mask_path = dir_code + '/dataset/test_masks'

# Locations and names for saving training checkpoints
cp_save_path = dir_code + '/results/weights'
logs_path = dir_code + '/results/logs'
pik_save_path = dir_code + '/results/checkpoint/loss_history.dat'

# Model parameters
# List of channels / classes (tissue): 0. Background, 1. CSF, 2. Grey Matter, 3. White Matter, 4. Fat, 5. Muscle
tissue = np.array([0, 1, 2, 3, 4, 5]) # TODO
img_size = (256, 256, len(tissue))

train_batch_size = 10    # TODO
valid_batch_size = 10    # TODO
num_epochs = 40          # TODO

# Load pre-trained model
model_weights = None #'/bmrNAS/people/akshay/dl/oai_data/unet_2d/weights/unet_2d_men_weights.009--0.7682.h5'

# learning rate schedule
# Implementing a step decay for now
def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def train_seg(img_size, train_path, valid_path, train_batch_size, valid_batch_size, 
                cp_save_path, num_epochs, pik_save_path, tissue):

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    train_files, train_nbatches = calc_generator_info(train_path, train_batch_size)
    valid_files, valid_nbatches = calc_generator_info(valid_path, valid_batch_size)

    # Print some useful debugging information
    print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))    
    print('INFO: Image size: %s' % (img_size,))
    print('INFO: Number of tissues being segmented: %d' % len(tissue))


    # create the unet model
    model = unet_2d_model(img_size, num_classes=len(tissue))

    # Set up the optimizer 
    model.compile(optimizer=Adam(lr=1e-9, beta_1=0.99, beta_2=0.995, epsilon=1e-08, decay=0.0), 
                  loss=dice_loss)

    # model callbacks per epoch
    cp_cb = ModelCheckpoint(cp_save_path + '/weights.{epoch:03d}-{val_loss:.4f}.h5', save_best_only=True)
    tfb_cb = tfb(logs_path, histogram_freq=1)
    lr_cb   = lrs(step_decay)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb, lr_cb]

    # Start the training    
    model.fit_generator(
            img_generator_oai(train_path, train_mask_path, train_batch_size, img_size, tissue),
            train_nbatches,
            epochs=num_epochs,
            validation_data=img_generator_oai(valid_path, valid_mask_path, valid_batch_size, img_size, tissue),
            validation_steps=valid_nbatches,
            callbacks=callbacks_list)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.lr, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    return hist_cb


# Print and save the training history
class LossHistory(kc.Callback):
    def on_train_begin(self, logs={}):
       self.val_losses = []
       self.losses = []
       self.lr = []
       self.epoch = []
 
    def on_epoch_end(self, batch, logs={}):
       self.val_losses.append(logs.get('val_loss'))
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))
       self.epoch.append(len(self.losses))

if __name__ == '__main__':

    model = unet_2d_model(img_size, num_classes=len(tissue))
    # print(model.summary())
    train_seg(img_size, train_path, valid_path, train_batch_size, valid_batch_size,
                cp_save_path, num_epochs, pik_save_path, tissue)

