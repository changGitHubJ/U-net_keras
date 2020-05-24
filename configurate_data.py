import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from PIL import Image

TRAIN_DATA_SIZE = 120
TEST_DATA_SIZE = 30
IMG_SIZE = 256
OUTPUT_SIZE = 256*256
CATEGORY = 1

FILENAMES = ['../DAGM/Class1_def/',
            '../DAGM/Class2_def/',
            '../DAGM/Class3_def/',
            '../DAGM/Class4_def/',
            '../DAGM/Class5_def/',
            '../DAGM/Class6_def/']

def savetxt(filename, data):
    for i in range(data.shape[0]):
        print(filename + ', line ' + str(i))
        file = open(filename, 'a')     
        file.write('{:.9f}'.format(data[i, 0]))
        for j in range(1, data.shape[1]):
            file.write(',' + '{:.9f}'.format(data[i, j]))
        file.write('\n')
        file.close()

if __name__ == "__main__":

    init = tf.global_variables_initializer()
    sess = tf.Session()
    with sess.as_default():

        if not os.path.exists('./data'):
            os.mkdir('./data')

        # remove old file
        if(os.path.exists('./data/trainImage256.txt')):
           os.remove('./data/trainImage256.txt')
        if(os.path.exists('./data/trainLABEL' + str(IMG_SIZE) + '.txt')):
            os.remove('./data/trainLABEL' + str(IMG_SIZE) + '.txt')
        if(os.path.exists('./data/testImage256.txt')):
           os.remove('./data/testImage256.txt')
        if(os.path.exists('./data/testLABEL' + str(IMG_SIZE) + '.txt')):
            os.remove('./data/testLABEL' + str(IMG_SIZE) + '.txt')

        cntTrain = 0
        cntTest = 0
        for n in range(CATEGORY):
            for k in range(TRAIN_DATA_SIZE + TEST_DATA_SIZE):
                filename = FILENAMES[n] + str(k + 1) + '.png'
                print(filename)
                imgtf = tf.read_file(filename)
                img = tf.image.decode_png(imgtf, channels=1)
                resized = tf.image.resize_images(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
                array = resized.eval()
                
                if(k < TRAIN_DATA_SIZE):
                    line = str(cntTrain)
                    for i in range(IMG_SIZE):
                        for j in range(IMG_SIZE):
                            line = line + ',' + str(array[i, j, 0])
                    line = line + '\n'
                    file = open('./data/trainImage256.txt', 'a')
                    file.write(line)
                    file.close()
                    cntTrain += 1
                else:
                    line = str(cntTest)
                    for i in range(IMG_SIZE):
                        for j in range(IMG_SIZE):
                            line = line + ',' + str(array[i, j, 0])
                    line = line + '\n'
                    file = open('./data/testImage256.txt', 'a')
                    file.write(line)
                    file.close()
                    cntTest += 1

        # label #
        trnLABEL = []
        tstLABEL = []
        for n in range(CATEGORY):
            x = np.linspace(1.0, 511, IMG_SIZE)
            y = np.linspace(1.0, 511, IMG_SIZE)
            filename = FILENAMES[n] + 'labels.txt'
            label1 = open(filename, 'r')
            print('reading ' + filename)
            for k in range(TRAIN_DATA_SIZE + TEST_DATA_SIZE):
                line = label1.readline()
                val = line.split('\t')
                num = int(val[0]) - 1
                mjr = float(val[1])
                mnr = float(val[2])
                rot = float(val[3])
                cnx = float(val[4])
                cny = float(val[5]) 

                # inverse rotate pixels
                label = np.zeros([OUTPUT_SIZE*CATEGORY + 1])
                label[0] = num # index
                for i in range(IMG_SIZE):
                    for j in range(IMG_SIZE):
                        dist = math.sqrt((x[i] - cnx)**2 + (y[j] - cny)**2)
                        xTmp = (x[i] - cnx) * math.cos(-rot) - (y[j] - cny) * math.sin(-rot)
                        yTmp = (x[i] - cnx) * math.sin(-rot) + (y[j] - cny) * math.cos(-rot)
                        ang = math.atan2(yTmp, xTmp)
                        distToEllipse = math.sqrt((mjr * math.cos(ang))**2 + (mnr * math.sin(ang))**2)
                        if(dist < distToEllipse):
                            label[(j*IMG_SIZE + i)*CATEGORY + n + 1] = 1.0 # defection
                        else:
                            label[(j*IMG_SIZE + i)*CATEGORY + n + 1] = 0.0
            
                # plot test
                #if(k == 0):
                    #plt.figure(figsize=(5, 5))
                    #z = label[1:OUTPUT_IMG_SIZE*OUTPUT_IMG_SIZE + 1].reshape([OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE])
                    #plt.imshow(z)
                    #plt.show()
            
                if(k < TRAIN_DATA_SIZE):
                    trnLABEL.append(label)
                else:
                    tstLABEL.append(label)
        
        # normalize
        w_array = np.array(trnLABEL)
        # for k in range(TRAIN_DATA_SIZE*CATEGORY):
        #     s = sum(w_array[k, 1:OUTPUT_SIZE*CATEGORY + 1])
        #     w_array[k, 1:OUTPUT_SIZE*CATEGORY + 1] = w_array[k, 1:OUTPUT_SIZE*CATEGORY + 1]/s
        # trnLABEL = w_array.tolist()
        
        w_tst_array = np.array(tstLABEL)
        # for k in range(TEST_DATA_SIZE*CATEGORY):
        #     s = sum(w_tst_array[k, 1:OUTPUT_SIZE*CATEGORY + 1])
        #     w_tst_array[k, 1:OUTPUT_SIZE*CATEGORY + 1] = w_tst_array[k, 1:OUTPUT_SIZE*CATEGORY + 1]/s
        # tstLABEL = w_tst_array.tolist()
        
        #np.savetxt('./data/trainLABEL' + str(IMG_SIZE) + '.txt', trnLABEL, fmt='%.10f', delimiter=',')
        #np.savetxt('./data/testLABEL' + str(IMG_SIZE) + '.txt', tstLABEL, fmt='%.10f', delimiter=',')
        savetxt('./data/trainLABEL' + str(IMG_SIZE) + '.txt', w_array)
        savetxt('./data/testLABEL' + str(IMG_SIZE) + '.txt', w_tst_array)

        sess.close()