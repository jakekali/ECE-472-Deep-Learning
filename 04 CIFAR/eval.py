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

model = tf.keras.models.load_model('logs/10/12/22160510model_34')
model.summary()

cifar10 = CIFAR(10)
cifar10.load()


model.evaluate(cifar10.X_test, tf.keras.utils.to_categorical(cifar10.Y_test), verbose=2)