import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def build_generator():
    input_shape = (96, 96, 3)
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 9, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(64, 3, strides=2, padding='same', activation='relu')(conv1)
    conv3 = Conv2D(128, 3, padding='same', activation='relu')(conv2)
    conv4 = Conv2D(128, 3, strides=2, padding='same', activation='relu')(conv3)
    conv5 = Conv2D(256, 3, padding='same', activation='relu')(conv4)
    conv6 = Conv2D(256, 3, strides=2, padding='same', activation='relu')(conv5)
    conv7 = Conv2D(512, 3, padding='same', activation='relu')(conv6)
    conv8 = Conv2D(512, 3, padding='same', activation='relu')(conv7)
    up1 = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(conv8)
    up2 = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(up1)
    conv9 = Conv2D(3, 9, padding='same', activation='tanh')(up2)
    outputs = (conv9 + 1) * 127.5
    return Model(inputs, outputs)

def build_discriminator():
    input_shape = (384, 384, 3)
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(64, 3, strides=2, padding='same', activation='relu')(conv1)
    conv3 = Conv2D(128, 3, padding='same', activation='relu')(conv2)
    conv4 = Conv2D(128, 3, strides=2, padding='same', activation='relu')(conv3)
    conv5 = Conv2D(256, 3, padding='same', activation='relu')(conv4)
    conv6 = Conv2D(256, 3, strides=2, padding='same', activation='relu')(conv5)
    conv7 = Conv2D(512, 3, padding='same', activation='relu')(conv6)
    conv8 = Conv2D
