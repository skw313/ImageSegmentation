from model import *
from data import *
import matplotlib.pyplot as plt

[x_test, y_test] = get_data("test")

model = u_net()
model.load_weights('results/archive_9_210414_good/weights/01_0.39.hdf5')
model.summary()
prediction = model.predict(x_test, batch_size=4, verbose=1)

for i in range(0, 20):
    input_data = x_test[i]
    truth_data = y_test[i]
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
    plt.savefig("results/archive_9_210414_good/predictions_test/predict_%d.jpg" % i, bbox_inches='tight')
    plt.close()
