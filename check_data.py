import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image

import common as c

def readImages(filename):
    images = np.zeros((c.TRAIN_DATA_SIZE*c.CATEGORY, c.IMG_SIZE, c.IMG_SIZE, c.CATEGORY))
    fileImg = open(filename)
    for k in range(c.TRAIN_DATA_SIZE*c.CATEGORY):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(c.IMG_SIZE):
            for j in range(c.IMG_SIZE):
                for n in range(c.CATEGORY):
                    images[k, i, j, n] = float(val[c.CATEGORY*(c.IMG_SIZE*i + j) + n + 1])
    return images

if __name__=='__main__':
    train_image = readImages('./data/trainImage256_%d.txt'%c.CATEGORY)
    train_label = readImages('./data/trainLabel256_%d.txt'%c.CATEGORY)

    for i in range(c.TRAIN_DATA_SIZE*c.CATEGORY):
        plt.figure(figsize=[10, 4])
        for j in range(c.CATEGORY):
            plt.subplot(2, 6, j + 1)
            fig = plt.imshow(train_image[i, :, : , j].reshape([c.IMG_SIZE, c.IMG_SIZE]))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)    
            
        for j in range(c.CATEGORY):
            plt.subplot(2, 6, c.CATEGORY + j + 1)
            fig = plt.imshow(train_label[i, :, :, j].reshape([c.IMG_SIZE, c.IMG_SIZE]))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

        plt.show()