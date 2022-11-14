from datetime import datetime
from time import time
import numpy as np 
import tensorflow as tf

class CIFAR:
    rng = None
    data_path = None
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None
    BoS = None
    key_label = None
    extra_images =  False
    

    def __init__(self, BoS=100, rng=None, extra_images=False):
        self.BoS = BoS
        self.rng = rng
        self.extra_images = extra_images

        if self.BoS == 10:
            self.data_path = "data/cifar-10-batches"
            self.label_key = b'labels'
        elif self.BoS == 100:
            self.data_path = "data/cifar-100-python"
            self.label_key = b'fine_labels'
        else: 
            raise ValueError("Invalud Big or Small Value")

    def load(self, nums=None):
        import pickle
        import os
        files = os.listdir(self.data_path)
        X = []
        Y = []
        for file in files:
            with open(self.data_path + "/" + file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                if(file[0] == 't'):
                    self.Y_test = np.array(dict[self.label_key])
                    self.X_test = dict[b'data'].reshape(len(dict[self.label_key]), 3, 32, 32).transpose(0, 2, 3, 1)
                else:
                    Y.append(dict[self.label_key])
                    X.append(dict[b'data'].reshape(len(dict[self.label_key]), 3, 32, 32).transpose(0, 2, 3, 1))
        
        X = np.concatenate(X)
        Y = np.concatenate(Y)

        # Data Augmentation based on 
        # https://medium.com/mlearning-ai/lessons-learned-from-reproducing-resnet-and-densenet-on-cifar-10-dataset-6e25b03328da
        # https://arxiv.org/pdf/1512.03385.pdf

        # Random Crop
        

        # X = X / 128.0
        # X = X - 0.5
        
        # self.X_test = self.X_test / 128.0
        # self.X_test = self.X_test - 0.5


        import math
        rd = np.random.permutation(len(X), )
        train_index =  tf.constant(rd[:-math.floor(len(X) * 0.1)], dtype=tf.int32)
        val_index = tf.constant(rd[-math.floor(len(X) * 0.1):], dtype=tf.int32)

        self.X_val = tf.gather(X, val_index)
        self.Y_val = tf.gather(Y, val_index)

        self.X_train = tf.gather(X, train_index)
        self.Y_train = tf.gather(Y, train_index)


        if self.extra_images:
            X = self.X_train
            Y = self.Y_train

            X_rand = tf.image.random_flip_left_right(X)
            X_rand = tf.image.random_brightness(X_rand, 0.2)
            X_rand = tf.image.random_contrast(X_rand, 0.8, 1.2)
            X_rand = tf.image.random_saturation(X_rand, 0.8, 1.2)

            X_rand1 = tf.image.random_flip_left_right(X)
            X_rand1 = tf.image.random_brightness(X_rand1, 0.2)
            X_rand1 = tf.image.random_contrast(X_rand1, 0.8, 1.2)
            X_rand1 = tf.image.random_saturation(X_rand1, 0.8, 1.2)

            X_rand2 = tf.image.random_flip_left_right(X)
            X_rand2 = tf.image.random_brightness(X_rand2, 0.2)
            X_rand2 = tf.image.random_contrast(X_rand2, 0.8, 1.2)
            X_rand2 = tf.image.random_saturation(X_rand2, 0.8, 1.2)

            X_rand3 = tf.image.random_flip_left_right(X)
            X_rand3 = tf.image.random_brightness(X_rand3, 0.2)
            X_rand3 = tf.image.random_contrast(X_rand3, 0.8, 1.2)
            X_rand3 = tf.image.random_saturation(X_rand3, 0.8, 1.2)


            self.X_train = np.concatenate((X, X_rand, X_rand1, X_rand2, X_rand3))
            self.Y_train = np.concatenate((Y, Y, Y, Y, Y))


            if nums is None:
                rd = np.random.permutation(len(self.X_train))
            else:
                rd = np.array(nums)

            now = datetime.now()
            np.savetxt("rd"+now.strftime("%m%d%Y%H%M%S")+".csv", rd, delimiter=",")

            self.X_train = tf.gather(self.X_train, rd)
            self.Y_train = tf.gather(self.Y_train, rd)

            


    def getBatch(self, batch_size):
        index = self.rng.integers(0, len(self.X_train), size=batch_size)
        
        return tf.gather(self.X_train, index), tf.gather(self.Y_train, index)

    def getVal(self, part, whole):
        # I needed to to this bc of RAM
        import math
        num_parts =  math.floor(len(self.X_val) / whole)
        index = np.arange(part * num_parts, (part + 1) * num_parts)
        return tf.gather(self.X_val, index), tf.gather(self.Y_val, index)

    def getTest(self, part, whole):
        import math
        num_parts =  math.floor(len(self.X_test) / whole)
        index = np.arange(part * num_parts, (part + 1) * num_parts)
        return tf.gather(self.X_test, index), tf.gather(self.X_test, index)