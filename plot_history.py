from utils.models import *
#from model import *
from data import *
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

plt.plot(epoch, loss)
plt.plot(epoch, val_loss)
plt.xlabel('Epoch')
plt.show()



'''
[x_test, y_test] = get_data("test", 1)

model = unet_2d_model((256,256,1))
model.load_weights('unet_2d/weights/unet_2d_men_weights.012--1.9482.h5')
model.summary()
prediction = model.predict(x_test, batch_size=1, verbose=1)

for i in range(0, 20):
    input_data = x_test[i]
    truth_data = y_test[i]

    segmentation = np.greater(prediction[i,:,:,1], prediction[i,:,:,0]) * 1.0
    segmentation = np.reshape(segmentation, (256, 256))

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(input_data, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(truth_data[:,:,1], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(segmentation, cmap='gray')
    plt.axis('off')
    plt.savefig("results/predictions/predict_%d.jpg" % i, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(truth_data[:,:,0])
    plt.imshow(truth_data[:,:,1])
    plt.show()
'''