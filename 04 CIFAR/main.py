#!/bin/env python3.8
from ast import Add
from cgitb import handler
from datetime import datetime
import enum
from logging import raiseExceptions
from pickletools import optimize
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa


from tqdm import trange

rng = np.random.default_rng(seed=4363)
DROPOUT = False
batch_norm = True
extra_images = False
aa = -1

def plotImg(img: np.ndarray, label: str):
    plt.imshow(img)
    plt.title(label)
    plt.show() 
    plt.savefig("./image_debugger.jpg")

class CFAR:
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

    def __init__(self, BoS=100, rng=None):
        self.BoS = BoS
        self.rng = rng

        if self.BoS == 10:
            self.data_path = "data/cifar-10-batches"
            self.label_key = b'labels'
        elif self.BoS == 100:
            self.data_path = "data/cifar-100-python"
            self.label_key = b'fine_labels'
        else: 
            raise ValueError("Invalud Big or Small Value")

    def load(self):
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
        if extra_images:
            X_rand = tf.image.random_flip_left_right(X)
            X_rand = tf.image.random_brightness(X_rand, 0.2)
            X_rand = tf.image.random_contrast(X_rand, 0.8, 1.2)
            X_rand = tf.image.random_saturation(X_rand, 0.8, 1.2)

            X_rand1 = tf.image.random_flip_left_right(X)
            X_rand1 = tf.image.random_brightness(X_rand1, 0.2)
            X_rand1 = tf.image.random_contrast(X_rand1, 0.8, 1.2)
            X_rand1 = tf.image.random_saturation(X_rand1, 0.8, 1.2)

            X = np.concatenate((X, X_rand, X_rand1))
            Y = np.concatenate((Y, Y, Y))

        X = tf.image.resize(X, [36,36])
        X = tf.image.random_crop(X, size=(len(X), 32,32,3))

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

c = CFAR(100,rng=rng)

c.load()
# https://www.analyticsvidhya.com/blog/2021/09/building-resnet-34-model-using-pytorch-a-guide-for-beginners/
gg = 1
l2 = tf.keras.regularizers.L2(l2=0.01)

def identity_block(x, filters):
    x_skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)    
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Add()([x, x_skip])
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)     
    return x

def conv_block(x, filters):
    x_skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer='l2')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)  

    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same",kernel_regularizer='l2')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)   

    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)


    x_skip = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_regularizer='l2')(x_skip)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)      
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)


    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation("relu")(x)
    return x

def resnet_builder(output_dim=10):
    x_IN = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.cast(x_IN, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x_IN)
    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    block_layers = [2,2,2]
    filters = [64,128,256]
    for i in range(len(block_layers)):
        if i == 0: 
            for _ in range(block_layers[i]):
                x = identity_block(x, filters[i])
        else:
            x = conv_block(x, filters[i])
            for _ in range(block_layers[i]-1):
                x = identity_block(x, filters[i])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(output_dim, activation="softmax")(x)
    model = tf.keras.models.Model(inputs = x_IN, outputs = x, name = "ResNet34")
    return model

dropout = 0.20
# model = tf.keras.models.load_model("logs/model_041541_finished_7050")
# model = resnet_builder(100)
loss = tf.keras.losses.SparseCategoricalCrossentropy()


#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
starter_learning_rate = 0.01
end_learning_rate = 0.001
decay_steps = 10000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)
#optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#tensor board
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



# Training code here
x_IN = tf.keras.layers.Input(shape=(32, 32, 3))

model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=x_IN, pooling='avg', classes=100, classifier_activation='softmax') # resnet_builder(100)
print(model.summary())

top_1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy')
top_5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
top_10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_accuracy')



model.compile(optimizer=optimizer, loss=loss, metrics=[top_1,top_5, top_10])

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("logs/model_{}.hd5".format(epoch))

print("Training...")
print(tf.one_hot(c.Y_train, 100).shape)
print(c.X_test.shape)
saver = CustomSaver()

# get CIFAR-100 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# model.load_weights("logs/model_040443_finished_0")
model.fit(x=x_train,y=y_train, epochs=3000, validation_split=0.1, callbacks=[tensorboard_callback, saver],
          batch_size=256, shuffle=True)


print("Evaluating...")

model.evaluate(x=c.X_test,y=c.Y_test, batch_size=128)

print("Saving model...")
model.save("logs/model_"+datetime.now().strftime("%H%M%S")+"_finished_"+str(int(top_1.result().numpy()*10000)))



# bar = trange(20000)
# for epoch in bar:
#     with tf.profiler.experimental.Trace("train", step_num=epoch, _r=1):
#         try:
#             x, y = c.getBatch(128)
#             with tf.GradientTape() as tape:
#                 y_hat = model(x, training=True)
#                 loss_value = loss(y, y_hat)

#             gpu_devices = tf.config.list_physical_devices('GPU')
#             if gpu_devices:
#                 memu = tf.config.experimental.get_memory_usage('GPU:0')

#             board.log(epoch, loss_value, board.train_accuracy(y, y_hat), board.train_top_5_accuracy(y, y_hat),split="train", memu=memu)
                
#             grads = tape.gradient(loss_value, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#             if epoch % 50 == 0:
#                 l = 0
#                 groups = 10
#                 for i in range(1, groups):
#                     x, y = c.getVal(i,groups)
#                     y_hat = model(x, training=False)
#                     l += loss(y, y_hat)
#                     board.val_accuracy(y, y_hat)
#                     board.val_top_5_accuracy(y, y_hat)

#                 board.log(epoch, l/groups, board.val_accuracy.result(),board.val_top_5_accuracy.result() ,split="val")

#             bar.set_description(f"Loss @ {epoch} => {loss_value.numpy():0.8f}")
#             bar.refresh()

            

#             if epoch % 2500 == 0:
#                 now = datetime.now()
#                 current_time = now.strftime("%H%M%S")
#                 model.save("./logs/model_" + str(current_time) + "_running_"+str(epoch))   

#             learning_rate = learning_rate_fn(epoch)


#         except KeyboardInterrupt:
#             print("KeyboardInterrupt")
#             now = datetime.now()
#             current_time = now.strftime("%H%M%S")
#             model.save("./logs/model_" + str(current_time) + "_interrupted_"+str(epoch))        
#             break

#         except Exception as e:
#             print(e)
#             now = datetime.now()
#             current_time = now.strftime("%H%M%S")
#             model.save("./logs/model_" + str(current_time) + "_interrupted_"+str(epoch))        
#             break 

# tf.profiler.experimental.stop()

# now = datetime.now()
# current_time = now.strftime("%H%M%S")
# model.save("./logs/model_" + str(current_time) + "_finished_"+str(epoch))      

# l=0
# y_hat = model(x, training=False)
# for i in range(1, 100):
#             x, y = c.getTest(i,100)
#             y_hat = model(x, training=False)
#             l += loss(y, y_hat)
#             board.test_accuracy(y, y_hat)
#             board.test_top_5_accuracy(y, y_hat)

# board.log(epoch, l/100, board.test_accuracy.result(), board.test_top_5_accuracy.result() ,split="test")

# board.close()
# now = datetime.now()
# current_time = now.strftime("%H%M%S")
# model.save("./logs/model_" + str(current_time))

