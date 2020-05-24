import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time

from PIL import Image

# Parameter
training_epochs = 301
batch_size = 20
display_step = 1
TRAIN_DATA_SIZE = 120
TEST_DATA_SIZE = 30
IMG_SIZE = 256
CATEGORY = 1

# Batch components
trainingImages = np.zeros((TRAIN_DATA_SIZE*CATEGORY, IMG_SIZE*IMG_SIZE + 1))
trainingLabels = np.zeros((TRAIN_DATA_SIZE*CATEGORY, IMG_SIZE*IMG_SIZE*CATEGORY + 1))
testImages = np.zeros((TEST_DATA_SIZE*CATEGORY, IMG_SIZE*IMG_SIZE))
testLabels = np.zeros((TEST_DATA_SIZE*CATEGORY, IMG_SIZE*IMG_SIZE*CATEGORY))

def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def conv2d_no_activation(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)

def conv2dtranspose(input, weight_shape, bias_shape, output_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME'), b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def inference(x, input_size):
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])

    # encoding ##############
    with tf.variable_scope("conv1_1"):
        conv1_1 = conv2d(x, [3, 3, 1, 64], [64])
    with tf.variable_scope("conv1_2"):
        conv1_2 = conv2d(conv1_1, [3, 3, 64, 64], [64])
        pool1 = max_pool(conv1_2)

    with tf.variable_scope("conv2_1"):
        conv2_1 = conv2d(pool1, [3, 3, 64, 128], [128])
    with tf.variable_scope("conv2_2"):
        conv2_2 = conv2d(conv2_1, [3, 3, 128, 128], [128])
        pool2 = max_pool(conv2_2)

    with tf.variable_scope("conv3_1"):
        conv3_1 = conv2d(pool2, [3, 3, 128, 256], [256])
    with tf.variable_scope("conv3_2"):
        conv3_2 = conv2d(conv3_1, [3, 3, 256, 256], [256])
        pool3 = max_pool(conv3_2)

    with tf.variable_scope("conv4_1"):
        conv4_1 = conv2d(pool3, [3, 3, 256, 512], [512])
    with tf.variable_scope("conv4_2"):
        conv4_2 = conv2d(conv4_1, [3, 3, 512, 512], [512])
        pool4 = max_pool(conv4_2)

    with tf.variable_scope("conv5_1"):
        conv5_1 = conv2d(pool4, [3, 3, 512, 1024], [1024])
    with tf.variable_scope("conv5_2"):
        conv5_2 = conv2d(conv5_1, [3, 3, 1024, 1024], [1024])
    with tf.variable_scope("conv5_3"):
        conv5_3 = conv2dtranspose(conv5_2, [3, 3, 512, 1024], [512], [input_size, 32, 32, 512])
        concated1 = tf.concat([conv5_3, conv4_2], axis=3)

    # decoding ##############
    with tf.variable_scope("conv_up6_1"):
        conv_up1_1 = conv2d(concated1, [3, 3, 1024, 512], [512])
    with tf.variable_scope("conv_up6_2"):
        conv_up1_2 = conv2d(conv_up1_1, [3, 3, 512, 512], [512])
    with tf.variable_scope("conv_up6_3"):
        conv6_3 = conv2dtranspose(conv_up1_2, [3, 3, 256, 512], [256], [input_size, 64, 64, 256])
        concated2 = tf.concat([conv6_3, conv3_2], axis=3)

    with tf.variable_scope("conv_up7_1"):
        conv_up7_1 = conv2d(concated2, [3, 3, 512, 256], [256])
    with tf.variable_scope("conv_up7_2"):    
        conv_up7_2 = conv2d(conv_up7_1, [3, 3, 256, 256], [256])
    with tf.variable_scope("conv_up7_3"):
        conv_up7_3 = conv2dtranspose(conv_up7_2, [3, 3, 128, 256], [128], [input_size, 128, 128, 128])
        concated3 = tf.concat([conv_up7_3, conv2_2], axis=3)

    with tf.variable_scope("conv_up8_1"):
        conv_up8_1 = conv2d(concated3, [3, 3, 256, 128], [128])
    with tf.variable_scope("conv_up8_2"):
        conv_up8_2 = conv2d(conv_up8_1, [3, 3, 128, 128], [128])
    with tf.variable_scope("conv_up8_3"):
        conv_up8_3 = conv2dtranspose(conv_up8_2, [3, 3, 64, 128], [64], [input_size, 256, 256, 64])
        concated4 = tf.concat([conv_up8_3, conv1_2], axis=3)

    with tf.variable_scope("conv9_1"):
        conv9_1 = conv2d(concated4, [3, 3, 128, 64], [64])
    with tf.variable_scope("conv9_2"):
        conv9_2 = conv2d(conv9_1, [3, 3, 64, 64], [64])
    with tf.variable_scope("output"):
        outputs = conv2d_no_activation(conv9_2, [1, 1, 64, CATEGORY], [CATEGORY])

    # return inputs, outputs, teacher, is_training
    return tf.reshape(outputs, [-1, IMG_SIZE * IMG_SIZE * CATEGORY])

def loss(output, y):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def evaluate(output, y):
    s = tf.sign(output)
    z = tf.constant(0, shape=[batch_size, IMG_SIZE*IMG_SIZE*CATEGORY], dtype=tf.float32)
    zero_cut = tf.maximum(s, z)
    correct_prediction = tf.multiply(zero_cut, y)
    accuracy = tf.reduce_sum(correct_prediction)
    return accuracy

def training(cost):
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    return train_op

def openfile(filename):
    file = open(filename)
    VAL = []
    while True:
        line = file.readline()
        if(not line):
            break
        val = line.split(' ')
        VAL.append(val)
    return VAL

def read_training_data():
    fileImg = open('./data/trainImage256.txt', 'r')
    for i in range(TRAIN_DATA_SIZE*CATEGORY):
        line = fileImg.readline()
        val = line.split(',')
        trainingImages[i, :] = val
    for i in range(TRAIN_DATA_SIZE*CATEGORY):
        for j in range(1,IMG_SIZE*IMG_SIZE + 1):
            trainingImages[i, j] /= 255.0

    filelbl = open('./data/trainLABEL256.txt', 'r')
    for i in range(TRAIN_DATA_SIZE*CATEGORY):
        line = filelbl.readline()
        val = line.split(',')
        trainingLabels[i, :] = val

def defineBatchComtents():
    num = np.linspace(0, TRAIN_DATA_SIZE*CATEGORY - 1, TRAIN_DATA_SIZE*CATEGORY)
    num = num.tolist()
    component = random.sample(num, batch_size)
    return component

def next_batch(batch_component):
    num = sorted(batch_component)
    lineNum = 0
    cnt = 0
    batch_x = []
    batch_y = []
    while True:
        if(cnt == batch_size):
            break
        else:
            if(int(num[cnt]) == int(trainingImages[lineNum, 0])):
                image = trainingImages[lineNum, 1:IMG_SIZE*IMG_SIZE + 1]
                label = trainingLabels[lineNum, 1:IMG_SIZE*IMG_SIZE*CATEGORY + 1]
                batch_x.append(image)
                batch_y.append(label)
                cnt += 1
        lineNum += 1

    return np.array(batch_x), np.array(batch_y)

def defineTestComtents():
    num = np.linspace(0, TEST_DATA_SIZE*CATEGORY - 1, TEST_DATA_SIZE*CATEGORY)
    num = num.tolist()
    component = random.sample(num, batch_size)
    return component

def next_test_set(component):
    num = sorted(component)
    lineNum = 0
    cnt = 0
    batch_x = []
    batch_y = []
    while True:
        if(cnt == batch_size):
            break
        else:
            if(int(num[cnt]) == int(trainingImages[lineNum, 0])):
                image = testImages[lineNum, :]
                label = testLabels[lineNum, :]
                batch_x.append(image)
                batch_y.append(label)
                cnt += 1
        lineNum += 1

    return np.array(batch_x), np.array(batch_y)

def read_test_data():
    fileImg = open('./data/testImage256.txt', 'r')
    for i in range(TEST_DATA_SIZE*CATEGORY):
        line = fileImg.readline()
        val = line.split(',')
        testImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    for i in range(TEST_DATA_SIZE*CATEGORY):
        for j in range(IMG_SIZE*IMG_SIZE):
            testImages[i, j] /= 255.0
    
    filelbl = open('./data/testLABEL256.txt', 'r')
    for i in range(TEST_DATA_SIZE*CATEGORY):
        line = filelbl.readline()
        val = line.split(',')
        testLabels[i, :] = val[1:IMG_SIZE*IMG_SIZE*CATEGORY + 1]

def WriteLog(msg):
    with open('./LOG/trace.LOG', mode='a') as f:
        f.write(msg + '\n')

if __name__=='__main__':    
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            with tf.variable_scope("scope_model"):
                x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE]) # inputs(gray image)
                y = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*CATEGORY]) # teacher
                input_size = tf.placeholder(tf.int32)
            
                read_training_data()
                read_test_data()
                if not os.path.exists('./LOG'):
                    os.mkdir('./LOG')
                
                #output = inference(x, 0.0001, is_training)
                output = inference(x, input_size)
                cost = loss(output, y)
                train_op = training(cost)
                eval_op = evaluate(output, y)
                sess = tf.Session()
                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                # Training cycle
                for epoch in range(training_epochs):
                    batch_component = defineBatchComtents()
                    minibatch_x, minibatch_y = next_batch(batch_component)
                    sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, input_size: batch_size})
                    
                    # display logs per step
                    if epoch % display_step == 0:
                        component = defineTestComtents()
                        minitest_x, minitest_y = next_test_set(component)
                        cost_val = sess.run(cost, feed_dict={x: minitest_x, y: minitest_y, input_size: batch_size})
                        accuracy = sess.run(eval_op, feed_dict={x: minitest_x, y: minitest_y, input_size: batch_size})
                        
                        msg = "Epoch: " + str(epoch+1) + ", cost = " + "{:.9f}".format(cost_val) + ", Validation Error = " + "{:.9f}".format(accuracy)
                        print(msg)
                        WriteLog(msg)
                        
                print("Optimizer finished!")

                # generate decoded image with test data
                component = defineTestComtents()
                minitest_x, minitest_y = next_test_set(component)
                result_accuracy = sess.run(eval_op, feed_dict={x: minitest_x, y: minitest_y, input_size: batch_size})
                print('accuracy (test) = ', result_accuracy)

                if not os.path.exists('./result'):
                    os.mkdir('./result')
                for m in range(CATEGORY):
                    decoded_imgs = sess.run(output, feed_dict={x: testImages[TEST_DATA_SIZE*m:TEST_DATA_SIZE*(m + 1), :], y: testLabels[TEST_DATA_SIZE*m:TEST_DATA_SIZE*(m + 1), :], input_size: TEST_DATA_SIZE})
                    decoded_imgs = decoded_imgs.reshape([-1, IMG_SIZE, IMG_SIZE, CATEGORY])
                    for i in range(TEST_DATA_SIZE):
                        for n in range(CATEGORY):
                            np.savetxt('./result/' + str(TEST_DATA_SIZE*m + i) + '_' + str(n) + '.txt', decoded_imgs[i, :, :, n])

                    