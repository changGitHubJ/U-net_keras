import keras
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.merge import concatenate

class MyModel:
    def __init__(self, input_size, batch_size, epochs):
        self.model = ""
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.graph = tf.get_default_graph()

    def create_model(self):
        #x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])
        inputs = Input(self.input_size)

        # encoding ##############
        # with tf.variable_scope("conv1_1"):
        #     conv1_1 = conv2d(x, [3, 3, 1, 64], [64])
        # with tf.variable_scope("conv1_2"):
        #     conv1_2 = conv2d(conv1_1, [3, 3, 64, 64], [64])
        #     pool1 = max_pool(conv1_2)
        conv1_1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(inputs)
        conv1_2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

        # with tf.variable_scope("conv2_1"):
        #     conv2_1 = conv2d(pool1, [3, 3, 64, 128], [128])
        # with tf.variable_scope("conv2_2"):
        #     conv2_2 = conv2d(conv2_1, [3, 3, 128, 128], [128])
        #     pool2 = max_pool(conv2_2)
        conv2_1 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool1)
        conv2_2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        # with tf.variable_scope("conv3_1"):
        #     conv3_1 = conv2d(pool2, [3, 3, 128, 256], [256])
        # with tf.variable_scope("conv3_2"):
        #     conv3_2 = conv2d(conv3_1, [3, 3, 256, 256], [256])
        #     pool3 = max_pool(conv3_2)
        conv3_1 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool2)
        conv3_2 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv3_1)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

        # with tf.variable_scope("conv4_1"):
        #     conv4_1 = conv2d(pool3, [3, 3, 256, 512], [512])
        # with tf.variable_scope("conv4_2"):
        #     conv4_2 = conv2d(conv4_1, [3, 3, 512, 512], [512])
        #     pool4 = max_pool(conv4_2)
        conv4_1 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool3)
        conv4_2 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv4_1)
        #drop4 = Dropout(0.5)(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

        # with tf.variable_scope("conv5_1"):
        #     conv5_1 = conv2d(pool4, [3, 3, 512, 1024], [1024])
        # with tf.variable_scope("conv5_2"):
        #     conv5_2 = conv2d(conv5_1, [3, 3, 1024, 1024], [1024])
        # with tf.variable_scope("conv5_3"):
        #     conv5_3 = conv2dtranspose(conv5_2, [3, 3, 512, 1024], [512], [input_size, 32, 32, 512])
        #     concated1 = tf.concat([conv5_3, conv4_2], axis=3)
        conv5_1 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool4)
        conv5_2 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv5_1)
        #drop5 = Dropout(0.5)(conv5_2)
        conv_up5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv5_2))
        concated5 = concatenate([conv4_2, conv_up5], axis=3)

        # decoding ##############
        # with tf.variable_scope("conv_up6_1"):
        #     conv_up1_1 = conv2d(concated1, [3, 3, 1024, 512], [512])
        # with tf.variable_scope("conv_up6_2"):
        #     conv_up1_2 = conv2d(conv_up1_1, [3, 3, 512, 512], [512])
        # with tf.variable_scope("conv_up6_3"):
        #     conv6_3 = conv2dtranspose(conv_up1_2, [3, 3, 256, 512], [256], [input_size, 64, 64, 256])
        #     concated2 = tf.concat([conv6_3, conv3_2], axis=3)
        conv6_1 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated5)
        conv6_2 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv6_1)
        conv_up6 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv6_2))
        concated6 = concatenate([conv3_2, conv_up6], axis=3)
        #concated6 = concatenate([conv_up6, conv3_2], axis=3)

        # with tf.variable_scope("conv_up7_1"):
        #     conv_up7_1 = conv2d(concated2, [3, 3, 512, 256], [256])
        # with tf.variable_scope("conv_up7_2"):    
        #     conv_up7_2 = conv2d(conv_up7_1, [3, 3, 256, 256], [256])
        # with tf.variable_scope("conv_up7_3"):
        #     conv_up7_3 = conv2dtranspose(conv_up7_2, [3, 3, 128, 256], [128], [input_size, 128, 128, 128])
        #     concated3 = tf.concat([conv_up7_3, conv2_2], axis=3)
        conv7_1 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated6)
        conv7_2 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv7_1)
        conv_up7 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv7_2))
        concated7 = concatenate([conv2_2, conv_up7], axis=3)
        #concated7 = concatenate([conv_up7, conv2_2], axis=3)

        # with tf.variable_scope("conv_up8_1"):
        #     conv_up8_1 = conv2d(concated3, [3, 3, 256, 128], [128])
        # with tf.variable_scope("conv_up8_2"):
        #     conv_up8_2 = conv2d(conv_up8_1, [3, 3, 128, 128], [128])
        # with tf.variable_scope("conv_up8_3"):
        #     conv_up8_3 = conv2dtranspose(conv_up8_2, [3, 3, 64, 128], [64], [input_size, 256, 256, 64])
        #     concated4 = tf.concat([conv_up8_3, conv1_2], axis=3)
        conv8_1 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated7)
        conv8_2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv8_1)
        conv_up8 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv8_2))
        concated8 = concatenate([conv1_2, conv_up8], axis=3)
        #concated8 = concatenate([conv_up8, conv1_2], axis=3)

        # with tf.variable_scope("conv9_1"):
        #     conv9_1 = conv2d(concated4, [3, 3, 128, 64], [64])
        # with tf.variable_scope("conv9_2"):
        #     conv9_2 = conv2d(conv9_1, [3, 3, 64, 64], [64])
        # with tf.variable_scope("output"):
        #     outputs = conv2d_no_activation(conv9_2, [1, 1, 64, CATEGORY], [CATEGORY])
        conv9_1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated8)
        conv9_2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv9_1)
        outputs = Conv2D(1, 1, activation="sigmoid")(conv9_2)
        #outputs = Conv2D(1, 1)(conv9_2)

        self.model = Model(input=inputs, output=outputs)

        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr = 1e-4), metrics=['accuracy'])

    def softmax_cross_entropy(self, y_true, y_pred):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = tf.reduce_mean(cross_entropy, axis=1)
        loss = tf.reduce_mean(loss, axis=1)
        return loss

    def predict(self, y_pred, batch_size):
        return self.model.predict(y_pred, batch_size)

    def loss(self, y_true, y_pred):
        return self.softmax_cross_entropy(y_true, y_pred)

    def training(self, x_train, y_train, x_test, y_test):
        history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True, validation_data=(x_test, y_test))
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save(self, output_filename):
        self.model.save(output_filename)