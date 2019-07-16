from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import tensorflow as tf
from KernelPCA import *
import re
import sys
from keras.models import load_model
sys.setrecursionlimit(10000)
from keras import initializers
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

# model = ResNet50(weights='imagenet')
#
# img_path = 'C:\\Users\\Administrator\\Desktop\\cata\\[Beautyleg]美女Avy黑丝诱惑写真.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# # 将结果解码为元组列表 (class, description, probability)
# # (一个列表代表批次中的一个样本）
# print('Predicted:', decode_predictions(preds, top=1)[0])



model = VGG16(weights='imagenet', include_top=False)

img_path = 'C:\\Users\\Administrator\\Desktop\\cata\\[Beautyleg]美女Avy黑丝诱惑写真.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print(features)

