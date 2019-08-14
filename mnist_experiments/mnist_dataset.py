#! /usr/bin/python3

import numpy as np

class ImageSet:
    def __init__(self, filename):
        self.f = open(filename, "rb")

        self.f.seek(4)
        self._N_images = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(8)
        self.rows = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(12)
        self.cols = int.from_bytes(self.f.read(4), byteorder="big")

        self.next_index = 1

    def getImageAsBytes(self, index):
        if (index < 1 or index > self._N_images):
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
    def __init__(self, filename_images, filename_labels, training_fraction=1.0):
        self.image_set = ImageSet(filename_images)
        self.label_set = LabelSet(filename_labels)

        self._N_images = self.image_set._N_images
        self.next_index = 0

        if training_fraction < 0.0 or training_fraction > 1.0:
            raise ValueError('training_fraction must be between 0.0 and 1.0')
        self._training_fraction = training_fraction
        self._N_training = int(self._N_images * self._training_fraction)

        # Select pseudo-randomly some images to use for validation
        all_indices = np.arange(1, self._N_images + 1)
        rng = np.random.RandomState(12345)
        rng.shuffle(all_indices)
        self.training_indices = all_indices[:self._N_training]
        self.validation_indices = all_indices[self._N_training:]

    def get_next_training_batch(self, batch_size=32):
        image_batch = np.reshape([], (0, 784))
        label_batch = np.reshape([], (0, 10))
        remaining_images = batch_size
        while (remaining_images > 0):
            training_index = self.training_indices[self.next_index]
            image_batch = np.concatenate([image_batch, np.reshape(self.image_set.getImageAsFloatArray(training_index), (1, 784))], axis=0)
            label_batch = np.concatenate([label_batch, np.reshape(self.label_set.getOneHotLabel(training_index), (1, 10))], axis=0)
            remaining_images -= 1
            self.next_index = self.next_index + 1
            if (self.next_index == self._N_training):
                self.next_index = 0
        return image_batch, label_batch

    def get_validation_batches(self, batch_size=32):
        remaining_images_in_valiadtion_set = self._N_images - self._N_training
        next_index = 0 # Note  - different from self.next_index

        while remaining_images_in_valiadtion_set > 0:
            image_batch = np.reshape([], (0, 784))
            label_batch = np.reshape([], (0, 10))

            for _ in range(min(remaining_images_in_valiadtion_set, batch_size)):
                validation_index = self.validation_indices[next_index]
                image_batch = np.concatenate([image_batch, np.reshape(self.image_set.getImageAsFloatArray(validation_index), (1, 784))], axis=0)
                label_batch = np.concatenate([label_batch, np.reshape(self.label_set.getOneHotLabel(validation_index), (1, 10))], axis=0)

                remaining_images_in_valiadtion_set -= 1
                next_index += 1

            yield image_batch, label_batch

    def getAll(self):
        saved_index = self.next_index
        self.next_index = 1
        batch_for_return = self.getNextBatch(self._N_images)
        self.next_index = saved_index
        return batch_for_return

def get_training_set():
    return ImageAndLabelSet("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte")

def get_test_set():
    return ImageAndLabelSet("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte")
