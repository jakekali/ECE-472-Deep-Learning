#!/bin/env python3.8

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import math

from tqdm import trange


script_path = os.path.dirname(os.path.realpath(__file__))
rng = np.random.default_rng(seed=55)


class Data:
    def __init__(self, ns, sig, range):

        self.index = np.arange(2 * ns)
        self.ns = ns
        self.sig = sig
        self.range = range

        self.r_1 = rng.uniform(range[0], range[1], size=ns)
        self.r_2 = rng.uniform(range[0], range[1], size=ns)

        self.x_1 = self.r_1 * tf.math.cos(self.r_1)
        self.y_1 = self.r_1 * tf.math.sin(self.r_1)

        self.x_2 = -self.r_2 * tf.math.cos(self.r_2)
        self.y_2 = -self.r_2 * tf.math.sin(self.r_2)

        self.x_1 += rng.normal(0, sig, (ns))
        self.y_1 += rng.normal(0, sig, (ns))

        self.x_2 += rng.normal(0, sig, (ns))
        self.y_2 += rng.normal(0, sig, (ns))

        self.data_1 = [self.x_1, self.y_1]
        self.data_2 = [self.x_2, self.y_2]
        self.data_in = np.concatenate([self.data_1, self.data_2], axis=1).T
        self.data_out = np.concatenate(([0] * ns, [1] * ns))

    def plot(self):
        plt.scatter(self.x_1, self.y_1, label="1")
        plt.scatter(self.x_2, self.y_2, label="2")
        plt.legend()
        plt.savefig(f"{script_path}/fit1.pdf")

    def get_batch(self, batch_size):
        choices = np.array(rng.choice(self.index, size=batch_size))
        return self.data_in[choices], self.data_out[choices]


class DenseLayer(tf.Module):
    def __init__(self, n_in, n_out):
        self.w = tf.Variable(rng.uniform(-1, 1, (n_in, n_out)), dtype=tf.float32)
        self.b = tf.Variable(rng.normal(-1, 1, (n_out)), dtype=tf.float32)

    def __call__(self, x):
        return tf.nn.sigmoid((x @ self.w) + self.b)


class Model(tf.Module):
    def __init__(self):
        self.D1 = DenseLayer(2, 32)
        self.D2 = DenseLayer(32, 32)
        self.D3 = DenseLayer(32, 32)
        self.D4 = DenseLayer(32, 32)
        self.D5 = DenseLayer(32, 32)
        self.D6 = DenseLayer(32, 1)

    def __call__(self, x):
        l1 = self.D1(x)
        l2 = self.D2(l1)
        l3 = self.D3(l2)
        l4 = self.D4(l3)
        l5 = self.D5(l4)
        l6 = self.D6(l5)
        return l6


data = Data(500, 0.25, (0.75, 15))
data.plot()

model = Model()

optimizer = tf.optimizers.SGD(learning_rate=0.03)
bar = trange(10)
for i in bar:
    with tf.GradientTape() as tape:
        x, y = data.get_batch(32)
        y_hat = model(x)
        print("Hat")
        print(y_hat)
        print("Y")
        print(y)
        loss = tf.reduce_mean(y * tf.math.log(y_hat) + (1 - y) * tf.math.log(1 - y_hat))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
    bar.refresh()

fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=200)

# ax[0].set_title("Linear Combination of Gaussians")
# ax[0].set_xlabel("x")
# ax[0].set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
# h = ax[0].set_ylabel("y", labelpad=10)
# h.set_rotation(0)

# xs = np.linspace(0, 2, 100)
# xs = xs[:, np.newaxis]
# print(tf.shape(np.squeeze(xs)))
# print(tf.shape(model(np.squeeze(xs))))
# ax[0].plot(xs, np.squeeze(model(xs)), "--", label="model")
# ax[0].plot(np.squeeze(data.x), data.y, "o", label="training data")
# ax[0].plot(xs, np.squeeze((tf.math.sin(2 * math.pi * xs))), label="training data")
# ax[0].plot()

# ax[1].set_title("Linear Combination of Gaussians")
# ax[1].set_xlabel("x")
# ax[1].set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)

# for mu_i in range(tf.shape(model.mu)[0]):
#     theta_j = tf.math.exp(-((xs - model.mu[mu_i]) ** 2) / (model.sigma[mu_i]) ** 2)
#     ax[1].plot(xs, theta_j)
# ax[0].plot()
# ax[1].plot()


# plt.tight_layout()
# plt.savefig(f"{script_path}/fit.pdf")
