import re
import numpy as np
from PIL import Image
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
from keras.datasets import cifar10, cifar100
from keras.layers import Layer, AveragePooling2D, AveragePooling3D, add, Add, concatenate, Concatenate, Input, Flatten, \
    Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import cv2
from dataset import *
import sys

#################################################### ENERGY function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
####################################################
size = 22  ## to downsample images
total_sample_size = 10000
############################################################# returns numpy array by reading image
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
        x,y,z = self.test_data
        x = get_im_cv2(x,512,512,3)
        y = get_im_cv2(y)
        L.getLogger("train").info("Epoch {:5d} Evaluating on test set...".format(epoch + 1))
        test_loss, test_acc = self.model.evaluate([x, y],z, verbose=0)
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
# 根据分组的进行程调整学习率
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
#
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
        # 学习率下降的条件：
        # 1，轮回次数到了
        # 2，或者分数到了一定程度 暂时未知
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
def get_im_cv2(paths, img_rows=512, img_cols=512, color_type=3, normalize=True):
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
            # Reduce size
            resized = cv2.resize(img, (img_cols, img_rows))
            if normalize:
                resized = resized.astype('float32')
                resized /= 127.5
                resized -= 1.
                imgs.append(resized)
    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)
def get_train_batch(left, right, simi, batch_size, img_w, img_h, color_type, is_argumentation):
    while 1:
        for i in range(0, len(left), batch_size):
            x1 = get_im_cv2(left[i:i + batch_size], img_w, img_h, color_type)
            x2 = get_im_cv2(right[i:i + batch_size], img_w, img_h, color_type)
            y = simi[i:i + batch_size]
            yield  [x1,  x2],  y

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

def build_base_network():
    inputs = Input(shape=(512, 512, 3), name='in_layer')
    # convolutional layer 1
    conv1 = Conv2D(6, (3, 3), padding="same", activation="relu")(inputs)
    maxp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(12, (3, 3), padding="same", activation="relu")(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    naxp2 = Dropout(0.25)(maxp2)
    F = Flatten()(maxp2)
    d1 = Dense(128, activation='relu')(F)
    d1 = Dropout(0.1)(d1)
    d2 = Dense(50, activation='relu')(d1)
    model = Model(inputs, d2)
    return model

img_a = Input(shape=(512, 512, 3))
img_b = Input(shape=(512, 512, 3))
base_network = build_base_network()
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
model = Model(input=[img_a, img_b], output=distance)

epochs = 100
param_dict = {"mode": "quaternion",
                  "num_blocks": 10,
                  "start_filter": 24,
                  "dropout": 0,
                  "batch_size": 2,
                  "num_epochs": 10,
                  "dataset": "cifar10",
                  "act": "relu",
                  "init": "quaternion",
                  "lr": 1e-3,
                  "momentum": 0.9,
                  "decay": 0,
                  "clipnorm": 1.0
                  }

params = Params(param_dict)


opt = SGD(lr=params.lr,
              momentum=params.momentum,
              decay=params.decay,
              nesterov=True,
              clipnorm=params.clipnorm)

model.compile(opt,loss=contrastive_loss, metrics=['accuracy'])  ### NADAM optimizer
model.summary()

testErrCb = TestErrorCallback((left_dev, right_dev,similar_dev))
trainValHistCb = TrainValHistory()
lrSchedCb = LearningRateScheduler(schedule)
callbacks = [
    ModelCheckpoint('{}_weights.hd5'.format('quaternion'), monitor='val_loss', verbose=0, save_best_only=True),
    testErrCb,
    lrSchedCb,
    trainValHistCb]
model.fit_generator(generator=get_train_batch(left_train, right_train, similar_train,10, 512, 512, 3, True),
                             steps_per_epoch=20050,
                             epochs=50,
                             verbose=1,
                             validation_data=([get_im_cv2(left_dev,512,512,3), get_im_cv2(right_dev,512,512,3)],similar_dev),
                             validation_steps=52,
                             callbacks=callbacks)
################################




