from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.misc
from utils.data_util import img2cell
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO

def image_summary(tag, images, row_num=10, col_num=10, margin_syn=2):
    cell_images = img2cell(images, row_num=row_num, col_num=col_num, margin_syn=margin_syn)
    cell_image = cell_images[0]
    try:
        s = StringIO()
    except:
        s = BytesIO()
    scipy.misc.toimage(cell_image).save(s, format="png")
    img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=cell_image.shape[0], width=cell_image.shape[1])
    return tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])


def batch_norm(x, training=True, name=None):
    return tf.layers.batch_normalization(x, training=training, name=name)

def instance_norm(x, epsilon=1e-6, training=True, name=None):
    # with tf.variable_scope(name):
    #     mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    #     scale = tf.get_variable('scale', [x.get_shape()[-1]],
    #                             initializer=tf.truncated_normal_initializer(
    #                                 mean=1.0, stddev=0.02))
    #     offset = tf.get_variable(
    #         'offset', [x.get_shape()[-1]],
    #         initializer=tf.constant_initializer(0.0)
    #     )
    #     inv = tf.rsqrt(var + epsilon)
    #     out = scale * (x - mean) * inv + offset
    #     return out
    return tf.contrib.layers.instance_norm(x, epsilon=epsilon, scope=name)

def leaky_relu(input_, leakiness=0.2):
    assert leakiness <= 1
    return tf.maximum(input_, leakiness * input_)



def conv2d(input_, output_dim, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, stddev=0.02, name="conv2d"):
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    else:
        k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides

    with tf.variable_scope(name):
        if type(padding) == list or type(padding) == tuple:
            padding = [0] + list(padding) + [0]
            input_ = tf.pad(input_, [[p, p] for p in padding], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding.upper())
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if activate_fn:
            conv = activate_fn(conv)
        return conv


def fully_connected(input_, output_dim, stddev=0.02, name="fc"):
    shape = input_.shape
    return conv2d(input_, output_dim, kernal=list(shape[1:3]), strides=(1, 1), padding="VALID", stddev=stddev, name=name)


def build_residual_block(input_, dim, norm_layer=batch_norm, use_dropout=False, init_gain=0.02, name="residule_block"):
    with tf.variable_scope(name):
        conv_block = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_block = conv2d(conv_block, dim, kernal=(3, 3), strides=(1, 1), padding="VALID", stddev=init_gain, name=name + "_c1")
        conv_block = tf.nn.relu(norm_layer(conv_block, name=name + "_bn1"))
        if use_dropout:
            conv_block = tf.nn.dropout(conv_block, 0.5)
        conv_block = tf.pad(conv_block, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_block = conv2d(conv_block, dim, kernal=(3, 3), strides=(1, 1), padding="VALID", stddev=init_gain, name=name + "_c2")
        conv_block = norm_layer(conv_block, name=name + "_bn2")
        return conv_block + input_

def deconv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, kernel_initializer=None, name=None):
    with tf.variable_scope(name):
        new_shape = tf.shape(inputs)[1:3] * strides
        input_ = tf.image.resize_images(
            images=inputs, size=new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.conv2d(input_, filters, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer)


def convt2d(input_, output_dim, kernel=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, stddev=0.02, name="convt2d"):
    assert type(kernel) in [list, tuple, int]
    assert type(strides) in [list, tuple, int]
    assert type(padding) in [list, tuple, int, str]
    if type(kernel) == list or type(kernel) == tuple:
        [k_h, k_w] = list(kernel)
    else:
        k_h = k_w = kernel
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides

    with tf.variable_scope(name):
        if type(padding) in [tuple, list, int]:
            if type(padding) == int:
                p_h = p_w = padding
            else:
                [p_h, p_w] = list(padding)
            pad_ = [0, p_h, p_w, 0]
            input_ = tf.pad(input_, [[p, p] for p in pad_], "CONSTANT")
            padding = 'VALID'

        h, w = input_.get_shape()[1:3]
        output_shape = [tf.shape(input_)[0], h * d_h, w * d_w, output_dim]

        w = tf.get_variable('w', [k_h, k_w, output_dim, input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        convt = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1],
                                       padding=padding.upper())
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(convt, biases)
        if activate_fn:
            convt = activate_fn(convt)
        return convt
