import numpy as np

class MyLoadData:
    IMG_SIZE = 0
    OUTPUT_SIZE = 0

    def __init__(self, IMG_SIZE, OUTPUT_SIZE):
        self.IMG_SIZE = IMG_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE

    def read_images(self, filename, DATA_SIZE):
        ImgOrg = np.loadtxt(filename, delimiter=',')
        Images = np.zeros((DATA_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1))
        for i in range(DATA_SIZE):
            Images[i, :, :, 0] = ImgOrg[i, 1:self.IMG_SIZE*self.IMG_SIZE + 1].reshape(self.IMG_SIZE, self.IMG_SIZE)
        Images = Images.astype('float32')
        Images /= 255
        return Images

    def read_labels(self, filename, DATA_SIZE):
        LblOrg = np.loadtxt(filename, delimiter=',')
        Labels = np.zeros((DATA_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.bool)
        for i in range(DATA_SIZE):
            Labels[i, :, :, 0] = LblOrg[i, 1:self.IMG_SIZE*self.IMG_SIZE + 1].reshape(self.IMG_SIZE, self.IMG_SIZE)
        Labels = Labels.astype('bool')
        return Labels
        
    # filewgh = open('./data/trainWEIGHT.txt', 'r')
    # for i in range(TRAIN_DATA_SIZE):
    #     line = filewgh.readline()
    #     val = line.split(',')
    #     trainingWeights[i, :] = val