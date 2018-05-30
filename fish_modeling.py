import os
import shutil
import numpy as np
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers import Input, GlobalAveragePooling2D, Add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils


batch_size = 64
epochs = 100
data_augmentation = True


def fishhead_loss(ybody_pred):
    """mean squared err loss
    L = (yhead_true - ybody_pred * yhead_pred)**2
    where yhead_true belongs to [-1, 0, 1]
          -1: fishhead heads left
          0: nonfishhead
          +1: fishhead heads right
    and where ybody_true belongs to [0, 1]
           0: nonfish
           1: fish
    :param ybody_pred: the predicted fishhead orientation
    :return: the loss
    """
    def fish_head_err(y_true, y_pred):
        # print('y_true shape {}, y_pred shape {}'.format(K.int_shape(y_true), K.int_shape(y_pred)))
        return K.mean(K.square(y_true - ybody_pred * y_pred), axis=1)
    return fish_head_err


def hdacc(y_true, y_pred):
    return K.mean(K.cast((y_true - y_pred) * y_true < 0.5, y_true.dtype))


def conv2d(x, filters, kernel_size=3, strides=1, padding='same', activation='relu', name=None):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,
               activation=activation, kernel_initializer='he_normal', name=name)(x)
    return x


def maxpool2d(x, pool_size=3, strides=2, padding='same', name=None):
    x = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(x)
    return x


def block_conv2d(x, layers, filters, kernel_size=3, padding='same', activation='relu', block_name=None):
    for i in range(layers):
        x = conv2d(x, filters, kernel_size, strides=1, padding=padding, activation=activation,
                   name='{}_conv{}_{}x{}'.format(block_name, str(i + 1), kernel_size, kernel_size))
    return x


def modelling_normal(input_shape=(40, 80, 3), model_name='fishnet_normal'):
    """13 conv layers in total"""
    img_in = Input(shape=input_shape, name='input_layer')
    x = block_conv2d(img_in, layers=2, filters=16, kernel_size=3, block_name='block1')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool1_3x3')  # 20x40x16
    x = block_conv2d(x, layers=3, filters=32, kernel_size=3, block_name='block2')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool2_3x3')  # 10x20x32
    x = block_conv2d(x, layers=3, filters=64, kernel_size=3, block_name='block3')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool3_3x3')  # 5x10x64
    x = block_conv2d(x, layers=3, filters=64, kernel_size=3, block_name='block4')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool4_3x3')  # 3x5x128
    x = block_conv2d(x, layers=2, filters=128, kernel_size=3, block_name='block5')
    features = GlobalAveragePooling2D(name='gap1')(x)  # 128

    body_dense = Dense(64, activation='relu', name='fishbody_fc1')(features)
    body_drop = Dropout(rate=0.5, name='fishbody_dropout')(body_dense)
    body_out = Dense(1, activation='sigmoid', name='body')(body_drop)

    head_dense = Dense(64, activation='relu', name='fishhead_fc1')(features)
    head_drop = Dropout(rate=0.5, name='fishhead_dropout')(head_dense)
    head_out = Dense(1, activation='tanh', name='head')(head_drop)

    model = Model(inputs=img_in, outputs=[body_out, head_out], name=model_name)
    model.compile(optimizer='adadelta', loss={'body': 'binary_crossentropy', 'head': fishhead_loss(body_out)},
                  loss_weights={'body': 1.0, 'head': 1.0}, metrics=['accuracy', hdacc])
    model.summary()

    return model


def fit_model(x_train, y_train, x_test, y_test, model_name='fishnet_normal'):
    model = modelling_normal(x_train.shape[1:], model_name=model_name)

    log_dir = datetime.now().strftime('data\\model\\{}_model_%Y%m%d%H'.format(model_name))
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)
    es = EarlyStopping(monitor='val_loss', patience=20)
    mc = ModelCheckpoint(log_dir + '\\FISHNET-EP{epoch:02d}-BODYACC{val_body_acc:.4f}-'
                                   'HEADACC{val_head_hdacc:.4f}.h5',
                         monitor='val_loss', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

    start = time()
    if not data_augmentation:
        print('Not using data augmentation.')
        h = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                      validation_data=(x_test, y_test), callbacks=[es, mc, tb])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        # Fit the model on the batches generated by datagen.flow().
        # h = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
        #                         steps_per_epoch=x_train.shape[0] // batch_size, workers=4,
        #                         validation_data=(x_test, y_test),
        #                         callbacks=[es, mc, tb])
        h = model.fit(x=x_train, y=[y_train[0], y_train[1]], batch_size=batch_size, epochs=epochs,
                      validation_data=(x_test, [y_test[0], y_test[1]]),
                      callbacks=[es, mc, tb])

    print('Total Time Spent: {:.2f} min'.format((time() - start) / 60))
    loss, val_loss = h.history['loss'], h.history['val_loss']
    m_loss, m_val_loss = np.argmax(loss), np.argmax(val_loss)
    print("Best Training Loss: {:.2f} achieved at EP #{}.".format(loss[m_loss], m_loss + 1))
    print("Best Testing Loss:  {:.2f} achieved at EP #{}.".format(val_loss[m_val_loss], m_val_loss + 1))
    # history_curve(h)


if __name__ == '__main__':
    # The data, shuffled and split between train and test sets:
    data = np.load('data\\fishset_20180407.npz')
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    print('x_train shape: {};  y_train shape: {}'.format(x_train.shape, [yt.shape for yt in y_train]))
    x_mean = np.mean(x_train, axis=0)
    x_train = ((x_train - x_mean) / 127).astype('float32')
    x_test = ((x_test - x_mean) / 127).astype('float32')
    fit_model(x_train, y_train, x_test, y_test)


