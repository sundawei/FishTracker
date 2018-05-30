import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from datetime import datetime
import time, os, shutil
from keras.layers import PReLU
import matplotlib.pyplot as plt


model_path = 'data\\model\\FISHID-EP90-ACC0.9505.npz'


def inception(net, base_num, scope):
    with tf.variable_scope(scope):
        # branch_0
        branch_0 = Conv2d(net, base_num, filter_size=(1, 1), act=tf.nn.relu, name='conv1_1')
        # branch_1
        branch_1 = Conv2d(net, base_num, filter_size=(1, 1), act=tf.nn.relu, name='conv2_1')
        branch_1 = Conv2d(branch_1, base_num, filter_size=(3, 3), act=tf.nn.relu, name='conv2_2')
        # branch_2
        branch_2 = Conv2d(net, base_num, filter_size=(1, 1), act=tf.nn.relu, name='conv3_1')
        branch_2 = Conv2d(branch_2, base_num, filter_size=(3, 3), act=tf.nn.relu, name='conv3_2')
        branch_2 = Conv2d(branch_2, base_num, filter_size=(3, 3), act=tf.nn.relu, name='conv3_3')
        # branch_3
        branch_3 = MaxPool2d(net, filter_size=(3, 3), strides=(1, 1), name='pool1')
        branch_3 = Conv2d(branch_3, base_num, filter_size=(1, 1), act=tf.nn.relu, name='conv4_1')
        # concat
        concat_net = ConcatLayer(layers=[branch_0, branch_1, branch_2, branch_3], concat_dim=-1, name='concat')
    return concat_net


def model(x):
    """ fishid modeling using tl
    :param x:  x is tf.placeholder for net inputs
    :return: constructed net and corresponding net outputs
    """
    w_init = tf.truncated_normal_initializer(stddev=0.1)
    b_init = tf.constant_initializer(0.1)

    net = InputLayer(x, name='input')
    net = Conv2d(net, 32, (3, 3), act=tf.nn.relu, W_init=w_init, b_init=b_init, name='conv1_1')
    net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')  # 20x40
    net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, W_init=w_init, b_init=b_init, name='conv2_1')
    net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, W_init=w_init, b_init=b_init, name='conv2_2')
    net = MaxPool2d(net, (3, 3), (2, 2), name='pool2')  # 10x20
    net = inception(net, base_num=32, scope='inception1')
    net = inception(net, base_num=48, scope='inception2')
    net = MaxPool2d(net, (3, 3), (2, 2), name='pool3')  # 5x10
    net = Conv2d(net,  64, (1, 1), act=tf.nn.relu, W_init=w_init, b_init=b_init, name='conv3_1')
    net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, W_init=w_init, b_init=b_init, name='conv3_2')
    net = MaxPool2d(net, (3, 3), (2, 2), name='pool4')  # 3x5
    net = GlobalMeanPool2d(net, name='gap')
    net = DenseLayer(net, 48, act=tf.identity, W_init=w_init, b_init=b_init, name='1fc')
    feature_net = LambdaLayer(net, fn=lambda x_in: PReLU()(x_in), name='Lambda1')
    net = DropoutLayer(feature_net, keep=0.5)
    net = DenseLayer(net, 20, act=tf.identity, W_init=w_init, b_init=b_init, name='2fc')

    features = feature_net.outputs
    y_pred = net.outputs
    return net, features, y_pred


class FishidDescriptor(object):
    def __init__(self, patch_shape=(40, 80, 3)):
        self.x = tf.placeholder(tf.float32, shape=(None, ) + patch_shape, name='img_inputs')
        self.network, self.feature_op, _ = model(self.x)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        tl.files.load_and_assign_npz(sess=self.sess, name=model_path, network=self.network)

        self.network.print_layers()
        self.network.print_params()

    def extract_features(self, batch_patches):
        feed_dict = {self.x: batch_patches, }
        feed_dict.update(tl.utils.dict_to_one(self.network.all_drop))
        features = self.sess.run(self.feature_op, feed_dict=feed_dict)
        return features

    def compute_feature_distances(self, features):
        """
        compute euclidean distances among feature vectors in feature space.
        :param features: N * m array, N is number of examples, m is feature dimensions
        :return: a normalized distance matrix (NxN) where each element is the distance between two feature vectors
        """
        diff_features = features[:, None, :] - features[None, :, :]  # Nx1xm - 1xNxm = NxNxm
        distances = np.linalg.norm(diff_features, axis=2)  # NxN
        max_dist = np.max(distances, axis=1)
        distances = distances / max_dist[:, None]
        return distances

