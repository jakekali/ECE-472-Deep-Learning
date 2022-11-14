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

from resnet import *
from cifar import *


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

top_1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy')
top_5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
top_10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_accuracy')

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        now = datetime.now()
        if epoch % 1 == 0 and epoch > 10:  # or save after some epoch, each k-th epoch etc.
            self.model.save("logs/"+ now.strftime("%D%H%M%S") + "model_{}".format(epoch))

saver = CustomSaver()

n = 3
drop_out = 0.35
tf.keras.regularizers.L2(l2=0.001)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

# model.add(tf.keras.layers.Conv2D(32, (n, n),  padding='same', kernel_regularizer='l2'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_out))
# model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

# model.add(tf.keras.layers.Conv2D(64, (n, n),  padding='same', kernel_regularizer='l2'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_out))

# model.add(tf.keras.layers.Conv2D(64, (n, n),  padding='same', kernel_regularizer='l2'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_out))
# model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

# model.add(tf.keras.layers.Conv2D(128, (n, n),  padding='same', kernel_regularizer='l2'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_out))

# model.add(tf.keras.layers.Conv2D(128, (n, n),  padding='same', kernel_regularizer='l2'))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(drop_out))

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(100))
# model.add(tf.keras.layers.Dropout(drop_out))
# model.add(tf.keras.layers.Activation('softmax'))

model = resnet_builder(100)

cifar100 = CIFAR(100, extra_images=True)
cifar100.load()
loss = tf.keras.losses.CategoricalCrossentropy()
model.save("logs/"+ model.name + datetime.now().strftime("%Y%m%d-%H%M%S") + "_100_clases_model")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=loss, metrics=[top_1, top_5, top_10])
model.fit(cifar100.X_train, tf.keras.utils.to_categorical(cifar100.Y_train), epochs=10, batch_size=128, validation_data=(cifar100.X_test, tf.keras.utils.to_categorical(cifar100.Y_test)), callbacks=[tensorboard_callback, saver])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=loss, metrics=[top_1, top_5, top_10])
model.fit(cifar100.X_train, tf.keras.utils.to_categorical(cifar100.Y_train), epochs=1, batch_size=128, validation_data=(cifar100.X_test, tf.keras.utils.to_categorical(cifar100.Y_test)), callbacks=[tensorboard_callback, saver])

print(model.evaluate(cifar100.X_test, tf.keras.utils.to_categorical(cifar100.Y_test)))

## loss: 0.5841 - top_1_accuracy: 0.8436 - top_5_accuracy: 0.9772 - top_10_accuracy: 0.9917 - val_loss: 1.5767 
# - val_top_1_accuracy: 0.6260 - val_top_5_accuracy: 0.8797 - val_top_10_accuracy: 0.9407


reduce_lr =tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                               patience=10, min_lr=0.0000001, verbose=1)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.8, nesterov=False)
loss = tf.keras.losses.CategoricalCrossentropy()


cifar10 = CIFAR(10, extra_images=True)
cifar10.load()

drop_out = 0.00

model10 = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (n, n),  padding='same', kernel_regularizer='l2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax')

    ]
)



model10.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


model10.fit(
    cifar10.X_train, tf.keras.utils.to_categorical(cifar10.Y_train), 
    epochs=350, 
    batch_size=128, 
    validation_data=((cifar10.X_val), tf.keras.utils.to_categorical(cifar10.Y_val)), 
    callbacks=[tensorboard_callback, saver, reduce_lr]
    )

print(model10.evaluate(cifar10.X_test, tf.keras.utils.to_categorical(cifar10.Y_test)))

# Epoch 34/350
# 23s 13ms/step - loss: 0.1192 - accuracy: 1.0000 - val_loss: 0.4689 - val_accuracy: 0.9074 - lr: 4.0000e-04

#  313/313 - 3s - loss: 0.5056 - accuracy: 0.8950 - 3s/epoch - 9ms/step