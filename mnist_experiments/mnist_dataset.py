#! /usr/bin/python3

import numpy as np

class ImageSet:
    def __init__(self, filename):
        self.f = open(filename, "rb")

        self.f.seek(4)
        self.N_images = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(8)
        self.rows = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(12)
        self.cols = int.from_bytes(self.f.read(4), byteorder="big")

        self.next_index = 1

    def getImageAsBytes(self, index):
        if (index < 1 or index > self.N_images):
            return None
        else:
            self.f.seek(16 + (index-1)*self.rows*self.cols)
            return self.f.read(self.rows*self.cols)

    def getImageAsFloatArray(self, index):
        return np.reshape([float(x)/255.0 for x in self.getImageAsBytes(index)],
                          (28, 28))

class LabelSet:
    def __init__(self, filename):
        self.f = open(filename, "rb")
        self.f.seek(4)
        self.N_labels = int.from_bytes(self.f.read(4), byteorder="big")

    def getLabel(self, index):
        if (index < 1 or index > self.N_labels):
            return None
        else:
            self.f.seek(8 + (index-1))
            return int.from_bytes(self.f.read(1), byteorder="big")

    def getOneHotLabel(self, index):
        label = self.getLabel(index)
        one_hot = np.zeros(10)
        one_hot[label] = 1
        return one_hot

class ImageAndLabelSet:
    def __init__(self, filename_images, filename_labels):
        self.image_set = ImageSet(filename_images)
        self.label_set = LabelSet(filename_labels)

        self.N_images = self.image_set.N_images
        self.next_index = 1

    def getNextBatch(self, batchSize):
        image_batch = np.reshape([], (0, 784))
        label_batch = np.reshape([], (0, 10))
        remaining_images = batchSize
        while (remaining_images > 0):
            image_batch = np.concatenate([image_batch, np.reshape(self.image_set.getImageAsFloatArray(self.next_index), (1, 784))], axis=0)
            label_batch = np.concatenate([label_batch, np.reshape(self.label_set.getOneHotLabel(self.next_index), (1, 10))], axis=0)
            remaining_images -= 1
            self.next_index = self.next_index + 1
            if (self.next_index > self.N_images):
                self.next_index = 1
        return image_batch, label_batch

    def getAll(self):
        saved_index = self.next_index
        self.next_index = 1
        batch_for_return = self.getNextBatch(self.N_images)
        self.next_index = saved_index
        return batch_for_return

def get_training_set():
    return ImageAndLabelSet("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte")

def get_test_set():
    return ImageAndLabelSet("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte")
