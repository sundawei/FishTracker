import os
import shutil
import cv2
import numpy as np
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, concatenate, Flatten
from keras.layers import Input, GlobalAveragePooling2D, Add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils

batch_size = 64
epochs = 200
data_augmentation = True


def history_curve(history):
    h = history.history
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(len(h['loss'])), h['loss'], label='Train')
    plt.plot(range(len(h['val_loss'])), h['val_loss'], label='Test')
    plt.title('Loss over ' + str(len(h['loss'])) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(len(h['acc'])), h['acc'], label='Train')
    plt.plot(range(len(h['val_acc'])), h['val_acc'], label='Test')
    plt.title('Accuracy over ' + str(len(h['acc'])) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


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


def naming(scope_name, branch_name, layer_name):
    return "{0}_{1}_{2}".format(scope_name, branch_name, layer_name)


def inception(x, i, scope_name):
    filters = int(32 * i)
    branch_0 = conv2d(x, filters, 1, name=naming(scope_name, 'branch0', 'conv0_1x1'))
    branch_1 = conv2d(x, filters, 1, name=naming(scope_name, 'branch1', 'conv1a_1x1'))
    branch_1 = conv2d(branch_1, filters, 3, name=naming(scope_name, 'branch1', 'conv1b_3x3'))
    branch_2 = conv2d(x, filters, 1, name=naming(scope_name, 'branch2', 'conv2a_1x1'))
    branch_2 = conv2d(branch_2, filters, 3, name=naming(scope_name, 'branch2', 'conv2b_3x3'))
    branch_2 = conv2d(branch_2, filters, 3, name=naming(scope_name, 'branch2', 'conv2c_3x3'))
    branch_3 = maxpool2d(x, 3, 1, name=naming(scope_name, 'branch3', 'maxpool3a_3x3'))
    branch_3 = conv2d(branch_3, filters, 1, name=naming(scope_name, 'branch3', 'conv3b_1x1'))
    branches = concatenate([branch_0, branch_1, branch_2, branch_3], name=scope_name + '_' + 'concat_branches')
    return branches


def modelling_normal(input_shape=(40, 80, 3), model_name='fishid_normal'):
    """15 conv layers in total"""
    img_in = Input(shape=input_shape, name='input_layer')
    x = block_conv2d(img_in, layers=2, filters=32, kernel_size=3, block_name='block1')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool1_3x3')  # 20x40x16
    x = block_conv2d(x, layers=2, filters=64, kernel_size=3, block_name='block2')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool2_3x3')  # 10x20x32
    x = block_conv2d(x, layers=3, filters=64, kernel_size=3, block_name='block3')
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool3_3x3')  # 5x10x64
    x = inception(x, 1, scope_name='inception1')  # 5x10
    x = inception(x, 1.5, scope_name='inception2')  # 5x10
    x = maxpool2d(x, 3, 2, padding='same', name='maxpool4_3x3')  # 3x5
    x = inception(x, 2, scope_name='inception3')  # 3x5
    # x = inception(x, 2, scope_name='inception4')  # 3x5
    features = GlobalAveragePooling2D(name='gap1')(x)  # 256
    features = Dropout(rate=0.5)(features)
    dense_out = Dense(20, activation='softmax', name='fc1')(features)

    model = Model(inputs=img_in, outputs=dense_out, name=model_name)
    model.compile(optimizer=Adam(lr=5e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def fit_model(x_train, y_train, x_test, y_test, model_name='fishid_normal'):
    model = modelling_normal(x_train.shape[1:], model_name=model_name)

    log_dir = datetime.now().strftime('data\\model\\{}_model_%Y%m%d%H'.format(model_name))
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)
    es = EarlyStopping(monitor='val_acc', patience=30)
    mc = ModelCheckpoint(log_dir + '\\FISHID-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
                         monitor='val_acc', save_best_only=True)
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
            # rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        # Fit the model on the batches generated by datagen.flow().
        h = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                steps_per_epoch=x_train.shape[0] // batch_size, workers=8,
                                validation_data=(x_test, y_test),
                                callbacks=[es, mc, tb])
        # h = model.fit(x=x_train, y=[y_train[0], y_train[1]], batch_size=batch_size, epochs=epochs,
        #               validation_data=(x_test, [y_test[0], y_test[1]]),
        #               callbacks=[es, mc, tb])

    print('Total Time Spent: {:.2f} min'.format((time() - start) / 60))
    acc, val_acc = h.history['acc'], h.history['val_acc']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("Best Training Accuracy: {:.2f}% achieved at EP #{}.".format(acc[m_acc] * 100, m_acc + 1))
    print("Best Testing Accuracy:  {:.2f}% achieved at EP #{}.".format(val_acc[m_val_acc] * 100, m_val_acc + 1))
    history_curve(h)


if __name__ == '__main__':
    # The data, shuffled and split between train and test sets:
    data = np.load('data\\fishid_20_20171228_20180413.npz')
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    print('x_train shape: {};  x_test shape: {}'.format(x_train.shape, x_test.shape))
    x_mean = np.mean(x_train, axis=0)
    cv2.imwrite('data\\fishid_mean.jpg', x_mean.astype(np.uint8))
    x_train = ((x_train - x_mean) / 127).astype('float32')
    x_test = ((x_test - x_mean) / 127).astype('float32')
    y_train = np_utils.to_categorical(y_train, 20)
    y_test = np_utils.to_categorical(y_test, 20)
    fit_model(x_train, y_train, x_test, y_test)
