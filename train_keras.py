import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time

from PIL import Image

import load_data as data
import model

# Parameter
training_epochs = 200
batch_size = 8
TRAIN_DATA_SIZE = 120
TEST_DATA_SIZE = 30
IMG_SIZE = 256
OUTPUT_SIZE = IMG_SIZE*IMG_SIZE
CATEGORY = 1

def main(data, model):
    print("Reading images...")
    x_train = data.read_images('./data/trainImage256.txt', TRAIN_DATA_SIZE)
    x_test = data.read_images('./data/testImage256.txt', TEST_DATA_SIZE)
    y_train = data.read_labels('./data/trainLABEL256.txt', TRAIN_DATA_SIZE)
    y_test = data.read_labels('./data/testLABEL256.txt', TEST_DATA_SIZE)
    
    print("Creating model...")
    model.create_model(multi_gpu=False)

    print("Now training...")
    history = model.training(x_train, y_train, x_test, y_test)
    accuracy = history.history["accuracy"]
    loss = history.history["loss"]
    eval = model.evaluate(x_test, y_test)
    
    print("accuracy = " + str(eval))
    model.save('./model.h5')

    if not os.path.exists('./result_keras'):
        os.mkdir('./result_keras')
    for i in range(TEST_DATA_SIZE):
        ret = model.predict(x_test[i, :, :, 0].reshape([1, IMG_SIZE, IMG_SIZE, 1]), 1)
        np.savetxt('./result_keras/' + str(i) + '.txt', ret[0, :, :, 0])
    
    with open("training_log.txt", "w") as f:
        for i in range(training_epochs):
            f.write(str(loss[i]) + "," + str(accuracy[i]) + "\n")
    ax1 = plt.subplot()
    ax1.plot(loss, color="blue")
    ax2 = ax1.twinx()
    ax2.plot(accuracy, color="orange")
    plt.show()

if __name__=='__main__':
    data = data.MyLoadData(IMG_SIZE, OUTPUT_SIZE)
    model = model.MyModel((IMG_SIZE, IMG_SIZE, 1), batch_size, training_epochs)
    main(data, model)