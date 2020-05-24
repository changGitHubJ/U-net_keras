import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image

NUM = 30
IMG_SIZE = 256
OUTPUT_SIZE = 256*256
CATEGORY = 1

def readImages(filename):
    images = np.zeros((NUM*CATEGORY, IMG_SIZE*IMG_SIZE))
    fileImg = open(filename)
    for k in range(NUM*CATEGORY):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(IMG_SIZE*IMG_SIZE):
            images[k, i] = float(val[i + 1])
    return images

def readLabels(filename):
    labels = np.zeros((NUM*CATEGORY, OUTPUT_SIZE*CATEGORY))
    fileImg = open(filename)
    for k in range(NUM*CATEGORY):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(OUTPUT_SIZE*CATEGORY):
            labels[k, i] = float(val[i + 1])
    return labels

if __name__=='__main__':
    tst_image = readImages('./data/testImage256.txt')
    tst_label = readLabels('./data/testLABEL256.txt')
    label = tst_label.reshape([-1, IMG_SIZE, IMG_SIZE, CATEGORY])

    for i in range(NUM*CATEGORY):
        plt.figure(figsize=[15, 4])
        plt.subplot(1, 4, 1)
        fig = plt.imshow(tst_image[i, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='gray', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)    
        
        plt.subplot(1, 4, 2)
        fig = plt.imshow(label[i, :, :, 0], cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        rslt = np.loadtxt('./result/' + str(i) + '_0.txt')
        max_val = np.max(rslt)
        min_val = np.min(rslt)
        rslt = rslt - min_val
        rslt = rslt / (max_val - min_val)
        plt.subplot(1, 4, 3)
        fig = plt.imshow(rslt, vmin=0.8, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        rslt = np.loadtxt('./result_keras/' + str(i) + '.txt')
        plt.subplot(1, 4, 4)
        fig = plt.imshow(rslt, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.show()
