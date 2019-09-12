



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Chase Gaudet
# code based on work by Chiheb Trabelsi
# on Deep Complex Networks git source

import sys
from KernelPCA import *
sys.setrecursionlimit(10000)
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from dataset import *
from numpy import  *
sys.setrecursionlimit(10000)
import logging as L
import numpy as np
from complex_layers.utils import GetReal, GetImag
from complex_layers.conv import ComplexConv2D
from complex_layers.bn import ComplexBatchNormalization
from quaternion_layers.utils import Params, GetR, GetI, GetJ, GetK
from quaternion_layers.conv import QuaternionConv2D
from quaternion_layers.bn import QuaternionBatchNormalization
import keras
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.layers import Layer, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, Flatten, \
    Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')

PICTURE_SIZE=512
# Callbacks:
# Print a newline after each epoch.
class PrintNewlineAfterEpochCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.write("\n")


# Also evaluate performance on test set at each epoch end.
class TestErrorCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.loss_history = []
        self.acc_history = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data

        L.getLogger("train").info("Epoch {:5d} Evaluating on test set...".format(epoch + 1))
        test_loss, test_acc = self.model.evaluate(x, y, verbose=0)
        L.getLogger("train").info("                                      complete.")

        self.loss_history.append(test_loss)
        self.acc_history.append(test_acc)

        L.getLogger("train").info(
            "Epoch {:5d} train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}, test_loss: {}, test_acc: {}".format(
                epoch + 1,
                logs["loss"], logs["acc"],
                logs["val_loss"], logs["val_acc"],
                test_loss, test_acc))


# Keep a history of the validation performance.
class TrainValHistory(Callback):
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


class LrDivisor(Callback):
    def __init__(self, patience=float(50000), division_cst=10.0, epsilon=1e-03, verbose=1, epoch_checkpoints={41, 61}):
        super(Callback, self).__init__()
        self.patience = patience
        self.checkpoints = epoch_checkpoints
        self.wait = 0
        self.previous_score = 0.
        self.division_cst = division_cst
        self.epsilon = epsilon
        self.verbose = verbose
        self.iterations = 0

    def on_batch_begin(self, batch, logs={}):
        self.iterations += 1

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        divide = False
        if (epoch + 1) in self.checkpoints:
            divide = True
        elif (
                current_score >= self.previous_score - self.epsilon and current_score <= self.previous_score + self.epsilon):
            self.wait += 1
            if self.wait == self.patience:
                divide = True
        else:
            self.wait = 0
        if divide == True:
            K.set_value(self.model.optimizer.lr, self.model.optimizer.lr.get_value() / self.division_cst)
            self.wait = 0
            if self.verbose > 0:
                L.getLogger("train").info("Current learning rate is divided by" + str(
                    self.division_cst) + ' and his values is equal to: ' + str(self.model.optimizer.lr.get_value()))
        self.previous_score = current_score


def schedule(epoch):
    if epoch >= 0 and epoch < 10:
        lrate = 0.01
        if epoch == 0:
            L.getLogger("train").info("Current learning rate value is " + str(lrate))
    elif epoch >= 10 and epoch < 100:
        lrate = 0.01
        if epoch == 10:
            L.getLogger("train").info("Current learning rate value is " + str(lrate))
    elif epoch >= 100 and epoch < 120:
        lrate = 0.01
        if epoch == 100:
            L.getLogger("train").info("Current learning rate value is " + str(lrate))
    elif epoch >= 120 and epoch < 150:
        lrate = 0.001
        if epoch == 120:
            L.getLogger("train").info("Current learning rate value is " + str(lrate))
    elif epoch >= 150:
        lrate = 0.0001
        if epoch == 150:
            L.getLogger("train").info("Current learning rate value is " + str(lrate))
    return lrate


def learnVectorBlock(I, featmaps, filter_size, act, bnArgs):
    """Learn initial vector component for input."""

    O = BatchNormalization(**bnArgs)(I)
    O = Activation(act)(O)
    O = Convolution2D(featmaps, filter_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=l2(0.0001))(O)

    O = BatchNormalization(**bnArgs)(O)
    O = Activation(act)(O)
    O = Convolution2D(featmaps, filter_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=l2(0.0001))(O)

    return O


def getResidualBlock(I, mode, filter_size, featmaps, activation, shortcut, convArgs, bnArgs):
    """Get residual block."""

    if mode == "real":
        O = BatchNormalization(**bnArgs)(I)
    elif mode == "complex":
        O = ComplexBatchNormalization(**bnArgs)(I)
    elif mode == "quaternion":
        O = QuaternionBatchNormalization(**bnArgs)(I)
    O = Activation(activation)(O)

    if shortcut == 'regular':
        if mode == "real":
            O = Conv2D(featmaps, filter_size, **convArgs)(O)
        elif mode == "complex":
            O = ComplexConv2D(featmaps, filter_size, **convArgs)(O)
        elif mode == "quaternion":
            O = QuaternionConv2D(featmaps, filter_size, **convArgs)(O)
    elif shortcut == 'projection':
        if mode == "real":
            O = Conv2D(featmaps, filter_size, strides=(2, 2), **convArgs)(O)
        elif mode == "complex":
            O = ComplexConv2D(featmaps, filter_size, strides=(2, 2), **convArgs)(O)
        elif mode == "quaternion":
            O = QuaternionConv2D(featmaps, filter_size, strides=(2, 2), **convArgs)(O)

    if mode == "real":
        O = BatchNormalization(**bnArgs)(O)
        O = Activation(activation)(O)
        O = Conv2D(featmaps, filter_size, **convArgs)(O)
    elif mode == "complex":
        O = ComplexBatchNormalization(**bnArgs)(O)
        O = Activation(activation)(O)
        O = ComplexConv2D(featmaps, filter_size, **convArgs)(O)
    elif mode == "quaternion":
        O = QuaternionBatchNormalization(**bnArgs)(O)
        O = Activation(activation)(O)
        O = QuaternionConv2D(featmaps, filter_size, **convArgs)(O)

    if shortcut == 'regular':
        O = Add()([O, I])
    elif shortcut == 'projection':
        if mode == "real":
            X = Conv2D(featmaps, (1, 1), strides=(2, 2), **convArgs)(I)
            O = Concatenate(1)([X, O])
        elif mode == "complex":
            X = ComplexConv2D(featmaps, (1, 1), strides=(2, 2), **convArgs)(I)
            O_real = Concatenate(1)([GetReal()(X), GetReal()(O)])
            O_imag = Concatenate(1)([GetImag()(X), GetImag()(O)])
            O = Concatenate(1)([O_real, O_imag])
        elif mode == "quaternion":
            X = QuaternionConv2D(featmaps, (1, 1), strides=(2, 2), **convArgs)(I)
            O_r = Concatenate(1)([GetR()(X), GetR()(O)])
            O_i = Concatenate(1)([GetI()(X), GetI()(O)])
            O_j = Concatenate(1)([GetJ()(X), GetJ()(O)])
            O_k = Concatenate(1)([GetK()(X), GetK()(O)])
            O = Concatenate(1)([O_r, O_i, O_j, O_k])

    return O


def getModel(params):
    mode = params.mode
    n = params.num_blocks
    sf = params.start_filter
    dataset = params.dataset
    activation = params.act
    inputShape = (3,PICTURE_SIZE,PICTURE_SIZE)
    channelAxis = 1
    filsize = (3, 3)
    convArgs = {
        "padding": "same",
        "use_bias": False,
        "kernel_regularizer": l2(0.0001),
    }
    bnArgs = {
        "axis": channelAxis,
        "momentum": 0.9,
        "epsilon": 1e-04
    }

    convArgs.update({"kernel_initializer": params.init})

    # Create the vector channels
    R = Input(shape=inputShape)

    if mode != "quaternion":
        I = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        O = concatenate([R, I], axis=channelAxis)
    else:
        I = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        J = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        K = learnVectorBlock(R, 3, filsize, 'relu', bnArgs)
        O = concatenate([R, I, J, K], axis=channelAxis)

    if mode == "real":
        O = Conv2D(sf, filsize, **convArgs)(O)
        O = BatchNormalization(**bnArgs)(O)
    elif mode == "complex":
        O = ComplexConv2D(sf, filsize, **convArgs)(O)
        O = ComplexBatchNormalization(**bnArgs)(O)
    else:
        O = QuaternionConv2D(64, (3,3), **convArgs)(O)
        O = QuaternionBatchNormalization(**bnArgs)(O)
    O = MaxPooling2D(pool_size=(3, 3))(O)
    #conv2
    O = QuaternionConv2D(128, (3,3), **convArgs)(O)
    O = QuaternionBatchNormalization(**bnArgs)(O)
    O = MaxPooling2D(pool_size=(3, 3))(O)
    #conv3
    O = QuaternionConv2D(256, (3,3), **convArgs)(O)
    O = QuaternionBatchNormalization(**bnArgs)(O)
    O = MaxPooling2D(pool_size=(3, 3))(O)
    #conv4
    O = QuaternionConv2D(512, (3,3), **convArgs)(O)
    O = QuaternionBatchNormalization(**bnArgs)(O)
    O = MaxPooling2D(pool_size=(3, 3))(O)
    #conv5
    O = QuaternionConv2D(16, (3,3), **convArgs)(O)
    O = QuaternionBatchNormalization(**bnArgs)(O)
    O = MaxPooling2D(pool_size=(3, 3))(O)

    # Flatten
    O = Flatten()(O)

    # Dense
    O = Dense(1000, activation='relu', kernel_regularizer=l2(0.0001))(O)

    model = Model(R, O)
    return model


def euclidean_distance(vects):
    x, y = vects
    x = hash(pca_data(x))
    y = hash(pca_data(y))

    y_ = tf.reduce_mean(tf.abs(tf.subtract(x, y)), axis=1, name='distance')
    return y_
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss(y, y_):
    l1 = tf.multiply(y, tf.square(y_))
    l2 = tf.multiply(tf.subtract(1.0, y), tf.pow(tf.maximum(tf.subtract(1.0, y_), 0), 2))
    loss = tf.reduce_mean(tf.add(l1, l2))
    return loss

def get_im_cv2(paths, img_rows=512, img_cols=512, color_type=3, normalize=True):
    imgs = []
    for path in paths:
        if color_type == 1:
            img = Image.open(path)
        elif color_type == 3:
            if '.JPEG' in path:
                #img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                img = Image.open(path)
                img =img.resize((PICTURE_SIZE,PICTURE_SIZE),Image.ANTIALIAS)
                img = np.asarray(img, dtype='float32')
                # Reduce size
                # resized = cv2.resize(img, (img_cols, img_rows))

                img = img.astype('float32')
                img /= 255.
                # resized -= 1.
                imgs.append(img)
                # .reshape(len(paths),3, PICTURE_SIZE, PICTURE_SIZE)
                #.transpose([0, 3, 2, 1])
    imgs=np.array(imgs).transpose([0, 3, 2, 1])
    return imgs

def get_train_batch(left, right, simi, batch_size):
    while 1:
        for i in range(0, len(left), batch_size):
            x1 = get_im_cv2(left[i:i + batch_size])
            x2 = get_im_cv2(right[i:i + batch_size])
            y = simi[i:i + batch_size]
            y=np.asarray(y)[:, np.newaxis]
            yield  [x1,  x2],  y

param_dict = {"mode": "quaternion",
                  "num_blocks": 10,
                  "start_filter": 24,
                  "dropout": 0,
                  "batch_size": 32,
                  "num_epochs": 200,
                  "dataset": "cifar10",
                  "act": "relu",
                  "init": "quaternion",
                  "lr": 1e-3,
                  "momentum": 0.9,
                  "decay": 0,
                  "clipnorm": 1.0
                  }

params = Params(param_dict)


img_a = Input(shape=(3,PICTURE_SIZE, PICTURE_SIZE))
img_b = Input(shape=(3,PICTURE_SIZE, PICTURE_SIZE))
base_network = getModel(params)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
model = Model(input=[img_a, img_b], output=distance)

epochs = 100


opt = SGD (lr = params.lr,
               momentum = params.momentum,
               decay = params.decay,
               nesterov = True,
               clipnorm = params.clipnorm)

model.compile(opt,loss=contrastive_loss, metrics=['accuracy'])
model.summary()


testErrCb = TestErrorCallback((left_dev, right_dev,similar_dev))
trainValHistCb = TrainValHistory()
lrSchedCb = LearningRateScheduler(schedule)
callbacks = [
    ModelCheckpoint('{}_weights.hd5'.format('quaternion'), monitor='val_loss', verbose=0, save_best_only=True),
    testErrCb,
    trainValHistCb
]

model.fit_generator(generator=get_train_batch(left_train, right_train, similar_train,10),
                             steps_per_epoch=25000,
                             epochs=10,
                             verbose=1,
                             validation_data=([get_im_cv2(left_dev), get_im_cv2(right_dev)],similar_dev),
                             validation_steps=52,
                             callbacks=callbacks)