# Display the learning history of the model
# Plot of training and validation loss over epochs

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

dir_code = os.getcwd()
data = dir_code + '/results/checkpoint/loss_history.dat'

with open(data, "rb") as f:
    dict = pickle.load(f)

epoch = dict[0]
lr = dict[1]
loss = dict[2]
val_loss = dict[3]

plt.plot(epoch, loss, label='Loss')
plt.plot(epoch, val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# TODO save plot