import re
import sys
from keras.models import load_model
sys.setrecursionlimit(10000)
from keras import initializers
from complex_layers.bn import ComplexBatchNormalization
import logging as L
from complex_layers.utils import GetReal, GetImag
from complex_layers.conv import ComplexConv2D
from quaternion_layers.utils import Params, GetR, GetI, GetJ, GetK
from quaternion_layers.conv import QuaternionConv2D
from quaternion_layers.bn import QuaternionBatchNormalization
import numpy as np
from keras.layers import Layer, AveragePooling2D,Add, concatenate, Concatenate,\
    Convolution2D, BatchNormalization,ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from sklearn.decomposition import IncrementalPCA as IPCA
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
import cv2
from dataset import *
import sys
from numpy import  *
from sklearn.decomposition import PCA
import tensorflow as tf

def pca1(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:, i]) for i in range(n_features)])  #normalization
  norm_X=X-mean  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data



K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')


def pca_data(data):
    # data=K.reshape(data,[-1,128])
    pca = PCA(n_components=4)
    pca.fit(data)  # 训练
    newdata = pca.fit_transform(data)  # 降维后的数据

    # PCA(copy=True, n_components=2, whiten=False)
    return newdata
def hammingDistance(x,y):
    distance=0
    for i in range(x.shape[1]):
        if x[0][i]!=y[0][i]:
            distance+=1
    return distance/x.shape[1]
####################################################尝试加入PCA
def pca_distance(vects):
    x, y = vects
    x = pca1(x, 128)
    y = pca1(y, 128)
    y_ = K.sqrt(K.sum(K.pow(np.subtract(x, y), 2), axis=1))
    return y_

    # distances=np.array([])
    #     # ddd=K.get_variable_shape(x)[0]
    #     # for i in range(5):
    #     #     x[i]=pca(x[i])
    #     #     y[i]=pca(y[i])
    #     #     distance = hammingDistance(x[i],y[i])
    #     #     distances[i]=distance
    #     # return distances

#################################################### ENERGY function
def euclidean_distance(vects):
    x, y = vects

    x = pca_data(x)

    y = pca_data(y)

    y_ = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(x, y), 2), 1, keep_dims=True), name='distance')
    return y_
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss(y, y_):
    l1 = tf.multiply(y, tf.square(y_))
    l2 = tf.multiply(tf.subtract(1.0, y), tf.pow(tf.maximum(tf.subtract(1.0, y_), 0), 2))
    loss = tf.reduce_mean(tf.add(l1, l2))
    return loss
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
            #img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            img = Image.open(path)
            img = np.asarray(img, dtype='float32')
            # Reduce size
            # resized = cv2.resize(img, (img_cols, img_rows))
            if normalize:
                img = img.astype('float32')
                img /= 255.
                # resized -= 1.
                imgs.append(img)
                # .reshape(len(paths), img_cols, img_rows, color_type)
    imgs=np.array(imgs)
    return imgs
def get_train_batch(left, right, simi, batch_size):
    while 1:
        for i in range(0, len(left), batch_size):
            x1 = get_im_cv2(left[i:i + batch_size])
            x2 = get_im_cv2(right[i:i + batch_size])
            y = simi[i:i + batch_size]
            y=np.asarray(y)[:, np.newaxis]
            yield  [x1,  x2],  y

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.6].mean()
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
    inputShape = (3, 512, 512)
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
        O = \
            ComplexConv2D(sf, filsize, **convArgs)(O)
        O = ComplexBatchNormalization(**bnArgs)(O)
    else:
        O = QuaternionConv2D(sf, filsize, **convArgs)(O)
        O = QuaternionBatchNormalization(**bnArgs)(O)
    O = Activation(activation)(O)
    # O = MaxPooling2D(pool_size=(2, 2))(O)
    for i in range(n):
        O = getResidualBlock(O, mode, filsize, sf, activation, 'regular', convArgs, bnArgs)
    O = MaxPooling2D(pool_size=(2, 2))(O)
    O = getResidualBlock(O, mode, filsize, sf, activation, 'projection', convArgs, bnArgs)
    for i in range(n - 1):
        O = getResidualBlock(O, mode, filsize, sf * 2, activation, 'regular', convArgs, bnArgs)
    O = MaxPooling2D(pool_size=(2, 2))(O)
    O = getResidualBlock(O, mode, filsize, sf * 2, activation, 'projection', convArgs, bnArgs)
    for i in range(n - 1):
        O = getResidualBlock(O, mode, filsize, sf * 4, activation, 'regular', convArgs, bnArgs)
    O = AveragePooling2D(pool_size=(8, 8))(O)
    # Flatten
    O = Flatten()(O)

    # Dense
    O = Dense(128, activation='softmax', kernel_regularizer=l2(0.0001))(O)

    model = Model(R, O)
    return model



def build_base_network():
    inputs = Input(shape=(512, 512,3), name='in_layer')
    # convolutional layer 1
    conv1 = Conv2D(6, (3, 3), padding="same", activation="relu")(inputs)
    maxp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(12, (3, 3), padding="same", activation="relu")(maxp1)
    maxp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    naxp2 = Dropout(0.25)(maxp2)
    F = Flatten()(maxp2)
    d1 = Dense(128, activation='relu')(F)
    d1 = Dropout(0.1)(d1)
    d2 = Dense(128, activation='relu')(d1)




    model = Model(inputs, d2)
    return model

def bulid_VGG_network():
    input=Input(shape=(512,512,3),name='in_layer')

    y_=Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='SAME',activation='relu')(input)
    y_=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='SAME')(y_)

    y_ = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='SAME', activation='relu')(y_)
    y_ = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='SAME')(y_)

    y_=Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='SAME',activation='relu')(y_)
    y_=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='SAME')(y_)

    y_=Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='SAME',activation='relu')(y_)
    y_=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='SAME')(y_)

    y_=Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='SAME',activation='relu')(y_)
    y_=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='SAME')(y_)

    y_ = Flatten()(y_)

    d1 = Dense(128, activation='relu',kernel_initializer=initializers.random_normal(stddev=0.1),bias_initializer=initializers.Constant(value=0.1))(y_)
    # d1 = pca(d1, batch=params.batch_size)
    model = Model(input, d1)
    return model




param_dict = {"mode": "quaternion",
                  "num_blocks": 10,
                  "start_filter": 24,
                  "dropout": 0,
                  "batch_size": 10,
                  "num_epochs": 10,
                  "dataset": "cifar10",
                  "act": "relu",
                  "init": "quaternion",
                  "lr": 1e-6,
                  "momentum": 0.9,
                  "decay": 0,
                  "clipnorm": 1.0
                  }

params = Params(param_dict)


img_a = Input(shape=(512, 512, 3))
img_b = Input(shape=(512, 512, 3))
base_network = bulid_VGG_network()
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
                             steps_per_epoch=20050,
                             epochs=10,
                             verbose=1,
                             validation_data=([get_im_cv2(left_dev), get_im_cv2(right_dev)],similar_dev),
                             validation_steps=52,
                             callbacks=callbacks)
###############################
# model_pre=load_model('quaternion_weights.hd5',custom_objects={'contrastive_loss': contrastive_loss})
# x=model_pre.predict([get_im_cv2(left_dev), get_im_cv2(right_dev)])
# print(x)



