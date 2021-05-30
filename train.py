import tensorflow as tf
from data import *
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, History
from utils.models import unet_2d_model
from utils.losses import dice_loss
from utils.generator_msk_seg import img_generator_oai
import time

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

# Paths
dir_code = os.getcwd()
train_path = dir_code + '/dataset/train_images'
train_mask_path = dir_code + '/dataset/train_masks'
valid_path = dir_code + '/dataset/val_images'
valid_mask_path = dir_code + '/dataset/val_masks'
test_path = dir_code + '/dataset/test_images'
test_mask_path = dir_code + '/dataset/test_masks'
img_size = (256,256,1)
# Tissues are in the following order
# 0. Femoral 1. Lat Tib 2. Med Tib. 3. Pat 4. Lat Men 5. Med Men # TODO edit
tissue = np.arange(0,1)

#-- Hyperparamters --#
learning_rate = 1e-3    # TODO
batch_size = 5          # TODO
num_epochs = 15         # TODO

# Get the data
'''
[x_train, y_train] = get_data("train", batch_size)
[x_val, y_val] = get_data("val", batch_size)
[x_test, y_test] = get_data("test", batch_size)
'''

K.set_image_data_format('channels_last')

# Initialise the model
model = unet_2d_model(img_size)
model.summary()
model.compile(optimizer=Adam(lr=learning_rate),
              loss=dice_loss)
checkpoint = ModelCheckpoint('results/weights/{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5',
                             save_weights_only=True,
                             #monitor='val_categorical_accuracy',
                             #mode='max',
                             save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='results/logs', histogram_freq=1)
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
callbacks_list = [checkpoint, tensorboard]

# Train model
start = time.time()
history = model.fit_generator(img_generator_oai(train_path, train_mask_path, batch_size, img_size, tissue, 'oai'),
                              steps_per_epoch=batch_size,
                              epochs=num_epochs,
                              validation_data=img_generator_oai(valid_path, train_mask_path, batch_size, img_size, tissue, 'oai'),
                              validation_steps=batch_size,
                              callbacks=callbacks_list)
end = time.time()
print("Training Complete in " + "{0:.2f}".format(end - start) + " secs")

'''
# Test model
evaluation = model.evaluate_generator(img_generator_oai(train_path, train_mask_path, batch_size, img_size, tissue, 'oai'), 
                                                        steps=1, verbose=1)

# Save predictions of the model
print("Saving the masks predicted by the model")
prediction = model.predict(x_test, batch_size, verbose=1)
save_prediction(prediction, x_test, y_test, num_classes=1)
'''
