import tensorflow as tf
from model import *
from data import *
from loss_functions import dice_coeff_loss
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

#-- Hyperparamters --#
learning_rate = 1e-3    # TODO
batch_size = 2          # TODO
num_epochs = 15         # TODO

# Get the data
[x_train, y_train] = get_data("train")
[x_val, y_val] = get_data("val")
[x_test, y_test] = get_data("test")

# Initialise the model
model = u_net()
model.summary()
model.compile(optimizer=Adam(lr=learning_rate),
              loss=dice_coeff_loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
checkpoint = ModelCheckpoint('results/weights/{epoch:02d}_{val_categorical_accuracy:.2f}.hdf5',
                             save_weights_only=True,
                             monitor='val_categorical_accuracy',
                             mode='max',
                             save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='results/logs', update_freq='epoch')
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Train model
start = time.time()
history = model.fit(x_train, y_train, batch_size, num_epochs, validation_data=(x_val, y_val), callbacks=[checkpoint, tensorboard])
end = time.time()
print("Training Complete in " + "{0:.2f}".format(end - start) + " secs")

# Test model
evaluation = model.evaluate(x_test, y_test, batch_size, verbose=1)

# Save predictions of the model
print("Saving the masks predicted by the model")
prediction = model.predict(x_test, batch_size, verbose=1)
save_prediction(prediction, x_test, y_test, num_classes=1)

