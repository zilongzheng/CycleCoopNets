try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
from model.custom_ops import *


def get_norm_layer(norm_type):
    if norm_type == 'batch_norm':
        return batch_norm
    elif norm_type == 'instance_norm':
        return instance_norm
    elif norm_type == 'none':
        return lambda x: x
    else:
        raise NotImplementedError(
            'Normalization type %s is not implemented.' % norm_type)


def get_weight_initializer(init_type='random_normal'):
    if init_type in ['random_normal', 'truncated_normal', 'he_normal']:
        return getattr(tf.initializers, init_type)
    else:
        raise NotImplementedError(
            'Initialization type %s is not implemented.' % init_type)


class TFModule():
    def __init__(self, name=None):
        self.name = name
        self.training = True
        self.module_list = []

    def test(self):
        self.training = False
        for m in self.module_list:
            m.test()

    def train(self):
        self.training = True
        for m in self.module_list:
            m.train()

    def __call__(self):
        raise NotImplementedError()


class ConvEBM(TFModule):
    def __init__(self, net_type='3_layer', weight_init='random_normal', init_gain=0.02, name='conv_ebm'):
        super(ConvEBM, self).__init__(name)
        self.init_gain = init_gain
        self.net_type = net_type
        self.weight_init = get_weight_initializer(weight_init)

    def __call__(self, inputs, training):
        if self.net_type == '3_layer':
            return self.ebm_3_layer(inputs, training)
        elif self.net_type == '4_layer':
            return self.ebm_4_layer(inputs, training)
        else:
            raise ValueError(
                "Undefined descriptor type: {}".format(self.net_type))

    def ebm_3_layer(self, inputs, training):
        ndf = 64
        kernel_initializer = self.weight_init(stddev=self.init_gain)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            out = tf.layers.conv2d(inputs, ndf, (5, 5), strides=(
                2, 2), padding="same", kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv1")
            out = tf.layers.conv2d(out, ndf * 2, (3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv2")

            out = tf.layers.conv2d(out, ndf * 4, (3, 3), strides=(1, 1), padding="same",
                                   kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv3")

            out = tf.layers.conv2d(out, 100, out.get_shape().as_list()[1:3], strides=(
                1, 1), padding="valid", kernel_initializer=kernel_initializer, name="fc")
        return out


    def ebm_4_layer(self, inputs, training):
        ndf = 64
        kernel_initializer = self.weight_init(stddev=self.init_gain)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            out = tf.layers.conv2d(inputs, ndf, (3, 3), strides=(
                1, 1), padding="same", kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv1")
            out = tf.layers.conv2d(out, ndf * 2, (4, 4), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv2")
            out = tf.layers.conv2d(out, ndf * 4, (4, 4), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv3")
            out = tf.layers.conv2d(out, ndf * 8, (4, 4), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_initializer, activation=tf.nn.leaky_relu, name="conv4")

            out = tf.layers.conv2d(out, 100, out.get_shape().as_list()[1:3], strides=(
                1, 1), padding="valid", kernel_initializer=kernel_initializer, name="fc")
        return out


def conv_batch_relu(input_, output_dim, kernal=(4, 4), strides=(2, 2), padding='SAME', norm_layer=batch_norm, activate_fn=tf.nn.relu, name="layer"):
    with tf.variable_scope(name or "conv_batch_relu"):
        conv = conv2d(input_, output_dim, kernal=kernal,
                      strides=strides, padding=padding, name="conv")
        # conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training)
        if norm_layer != None:
            conv = norm_layer(conv)
        if activate_fn != None:
            conv = activate_fn(conv)

        return conv


def convt_batch_relu(input_, output_shape, kernal=(4, 4), strides=(2, 2), padding='SAME', norm_layer=batch_norm, activate_fn=tf.nn.relu, dropout=0, name="layer"):
    with tf.variable_scope(name or "convt_batch_relu"):
        convt = convt2d(input_, output_shape, kernal=kernal,
                        strides=strides, padding=padding, name="convt")
        # conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training)
        if norm_layer != None:
            convt = norm_layer(convt)
        if activate_fn != None:
            convt = activate_fn(convt)
        if dropout > 0:
            convt = tf.nn.dropout(convt, dropout)
        return convt


def generator_unet256(input_, noise=None, norm_layer=batch_norm, name='gen_res', reuse=False):
    ngf = 64
    with tf.variable_scope(name, reuse=reuse):
        e1 = conv_batch_relu(input_, ngf, norm_layer=None,
                             activate_fn=tf.nn.leaky_relu, name="enc1")
        e2 = conv_batch_relu(e1, ngf * 2, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc2")
        e3 = conv_batch_relu(e2, ngf * 4, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc3")
        e4 = conv_batch_relu(e3, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc4")
        e5 = conv_batch_relu(e4, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc5")
        e6 = conv_batch_relu(e5, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc6")
        e7 = conv_batch_relu(e6, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc7")
        e8 = conv_batch_relu(e7, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc8")

        # if noise is not None:
        #     noise = tf.reshape(noise, [-1, 1, 1, noise.get_shape()[1]])
        #     e8 = tf.concat((e8, noise), axis=3)

        d1 = convt_batch_relu(e8, (None, 2, 2, ngf * 8), norm_layer=norm_layer,
                              dropout=0.5, activate_fn=tf.nn.relu, name="dec1")
        d1 = tf.concat([d1, e7], axis=3)
        d2 = convt_batch_relu(d1, (None, 4, 4, ngf * 8), norm_layer=norm_layer,
                              dropout=0.5, activate_fn=tf.nn.relu, name="dec2")
        d2 = tf.concat([d2, e6], axis=3)
        d3 = convt_batch_relu(d2, (None, 8, 8, ngf * 8), norm_layer=norm_layer,
                              dropout=0.5, activate_fn=tf.nn.relu, name="dec3")
        d3 = tf.concat([d3, e5], axis=3)
        d4 = convt_batch_relu(d3, (None, 16, 16, ngf * 8),
                              norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec4")
        d4 = tf.concat([d4, e4], axis=3)
        d5 = convt_batch_relu(d4, (None, 32, 32, ngf * 4),
                              norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec5")
        d5 = tf.concat([d5, e3], axis=3)
        d6 = convt_batch_relu(d5, (None, 64, 64, ngf * 2),
                              norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec6")
        d6 = tf.concat([d6, e2], axis=3)
        d7 = convt_batch_relu(d6, (None, 128, 128, ngf),
                              norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec7")
        d7 = tf.concat([d7, e1], axis=3)
        d8 = convt_batch_relu(d7, (None, 256, 256, 3),
                              norm_layer=None, activate_fn=tf.tanh, name="dec8")
        return d8


def generator_unet32(input_, noise=None, norm_layer=batch_norm, name='gen_res', reuse=False):
    ngf = 64
    with tf.variable_scope(name, reuse=reuse):
        e1 = conv_batch_relu(input_, ngf, norm_layer=None,
                             activate_fn=tf.nn.leaky_relu, name="enc1")
        e2 = conv_batch_relu(e1, ngf * 2, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc2")
        e3 = conv_batch_relu(e2, ngf * 4, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc3")
        e4 = conv_batch_relu(e3, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc4")
        e5 = conv_batch_relu(e4, ngf * 8, norm_layer=norm_layer,
                             activate_fn=tf.nn.leaky_relu, name="enc5")

        # if noise is not None:
        #     noise = tf.reshape(noise, [-1, 1, 1, noise.get_shape()[1]])
        #     e8 = tf.concat((e8, noise), axis=3)

        d1 = convt_batch_relu(e5, (None, 2, 2, ngf * 8), norm_layer=norm_layer,
                              dropout=0.5, activate_fn=tf.nn.relu, name="dec1")
        d1 = tf.concat([d1, e4], axis=3)
        d2 = convt_batch_relu(d1, (None, 4, 4, ngf * 4),
                              norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec2")
        d2 = tf.concat([d2, e3], axis=3)
        d3 = convt_batch_relu(d2, (None, 8, 8, ngf * 2),
                              norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec3")
        d3 = tf.concat([d3, e2], axis=3)
        d4 = convt_batch_relu(
            d3, (None, 16, 16, ngf), norm_layer=norm_layer, activate_fn=tf.nn.relu, name="dec4")
        d4 = tf.concat([d4, e1], axis=3)
        d5 = convt_batch_relu(
            d4, (None, 32, 32, 3), norm_layer=None, activate_fn=tf.nn.tanh, name="dec5")
        return d5


def build_residual_block(input_, dim, norm_layer=batch_norm, use_dropout=False, init_gain=0.02, training=True, name="residual_block"):

    kernel_initializer = tf.initializers.random_normal(stddev=init_gain)
    with tf.variable_scope(name):
        conv_block = tf.pad(
            input_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_block = tf.layers.conv2d(conv_block, dim, (3, 3), strides=(1, 1), padding="valid",
                                      kernel_initializer=kernel_initializer, name=name + "_c1")
        conv_block = tf.nn.relu(norm_layer(
            conv_block, training=training, name=name + "_bn1"))
        if use_dropout:
            conv_block = tf.layers.dropout(conv_block, 0.5, training=training)
        conv_block = tf.pad(
            conv_block, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_block = tf.layers.conv2d(conv_block, dim, (3, 3), strides=(1, 1), padding="valid",
                                      kernel_initializer=kernel_initializer, name=name + "_c2")
        conv_block = norm_layer(
            conv_block, training=training, name=name + "_bn2")
        return conv_block + input_


class ResidualBlock(TFModule):
    def __init__(self, dim, norm_layer=batch_norm, use_dropout=False, init_gain=0.02, name="residual_block"):
        super(ResidualBlock, self).__init__(name)
        self.dim = dim
        self.use_dropout = use_dropout
        self.init_gain = init_gain
        self.norm_layer = norm_layer

    def __call__(self, inputs, training):
        ndf = 64
        kernel_initializer = tf.initializers.random_normal(
            stddev=self.init_gain)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            conv_block = tf.pad(
                inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            conv_block = tf.layers.conv2d(conv_block, self.dim, (3, 3), strides=(1, 1), padding="valid",
                                          kernel_initializer=kernel_initializer)
            conv_block = tf.nn.relu(norm_layer(
                conv_block, training=training))
            if self.use_dropout:
                conv_block = tf.layers.dropout(
                    conv_block, 0.5, training=training)
            conv_block = tf.pad(
                conv_block, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            conv_block = tf.layers.conv2d(conv_block, self.dim, (3, 3), strides=(1, 1), padding="valid",
                                          kernel_initializer=kernel_initializer)
            conv_block = norm_layer(conv_block, training=training)
        return conv_block + inputs


class ResnetGeneratorMulti(TFModule):
    def __init__(self, img_nc=3, noise_type='all', norm_type='instance_norm', num_blocks=2, init_gain=0.02, use_dropout=False, name='gen_res'):
        super(ResnetGeneratorMulti, self).__init__(name)
        self.use_dropout = use_dropout
        self.init_gain = init_gain
        self.norm_type = norm_type
        self.img_nc = img_nc
        self.num_blocks = num_blocks
        self.noise_type = noise_type

    def __call__(self, inputs, noise=None, training=None):
        ngf = 64
        norm_layer = get_norm_layer(self.norm_type)
        kernel_initializer = tf.initializers.random_normal(
            stddev=self.init_gain)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # if noise is not None:
            #     inputs = tf.tile(inputs, [tf.shape(noise)[0], 1, 1, 1])
            if self.noise_type == 'input' or self.noise_type == 'all':
                z_inp = tf.reshape(noise, [-1, 1, 1, noise.get_shape()[-1]])
                z_inp = tf.tile(z_inp, [1, inputs.get_shape()[
                                1], inputs.get_shape()[2], 1])
                inputs = tf.concat([inputs, z_inp], axis=-1)
            padded_input = tf.pad(
                inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

            conv1 = tf.layers.conv2d(padded_input, ngf, (7, 7), strides=(1, 1), padding="valid",
                                     kernel_initializer=kernel_initializer, name="conv1")
            conv1 = norm_layer(conv1, training=training, name="conv1_bn")
            conv1 = tf.nn.relu(conv1)

            if self.noise_type == 'all':
                z_inp = tf.reshape(noise, [-1, 1, 1, noise.get_shape()[-1]])
                z_inp = tf.tile(z_inp, [1, conv1.get_shape()[
                                1], conv1.get_shape()[2], 1])
                conv1 = tf.concat([conv1, z_inp], axis=-1)

            conv2 = tf.layers.conv2d(conv1, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                     kernel_initializer=kernel_initializer, name="conv2")
            conv2 = norm_layer(conv2, training=training, name="conv2_bn")
            conv2 = tf.nn.relu(conv2)

            if self.noise_type == 'all':
                z_inp = tf.reshape(noise, [-1, 1, 1, noise.get_shape()[-1]])
                z_inp = tf.tile(z_inp, [1, conv2.get_shape()[
                                1], conv2.get_shape()[2], 1])
                conv2 = tf.concat([conv2, z_inp], axis=-1)
            conv3 = tf.layers.conv2d(conv2, ngf * 4, (3, 3), strides=(2, 2), padding="same",
                                     kernel_initializer=kernel_initializer, name="conv3")
            conv3 = norm_layer(conv3, training=training, name="conv3_bn")
            conv3 = tf.nn.relu(conv3)

            res_out = conv3
            for r in range(self.num_blocks):
                res_out = build_residual_block(
                    res_out, ngf * 4, norm_layer=norm_layer, use_dropout=self.use_dropout, init_gain=self.init_gain, training=training, name="res" + str(r))

            convt1 = tf.layers.conv2d_transpose(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                                kernel_initializer=kernel_initializer, name="convt1")
            # convt1 = deconv2d(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
            #                                     kernel_initializer=kernel_initializer, name="convt1")

            convt1 = norm_layer(
                convt1, training=training, name="convt1_bn")
            convt1 = tf.nn.relu(convt1)

            convt2 = tf.layers.conv2d_transpose(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
                                                kernel_initializer=kernel_initializer, name="convt2")
            # convt2 = deconv2d(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
            #                                     kernel_initializer=kernel_initializer, name="convt2")
            convt2 = norm_layer(
                convt2, training=training, name="convt2_bn")
            convt2 = tf.nn.relu(convt2)

            padded_output = tf.pad(
                convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            output = tf.layers.conv2d(padded_output, self.img_nc, (7, 7), strides=(1, 1), padding="valid",
                                      kernel_initializer=kernel_initializer, activation=tf.tanh, name="output")
            return output


class ResnetGenerator(TFModule):
    def __init__(self, img_nc=3, norm_type='instance_norm', num_blocks=2, init_gain=0.02, use_dropout=False, name='gen_res'):
        super(ResnetGenerator, self).__init__(name)
        self.use_dropout = use_dropout
        self.init_gain = init_gain
        self.norm_type = norm_type
        self.img_nc = img_nc
        self.num_blocks = num_blocks

    def __call__(self, inputs, noise=None, training=None):
        ngf = 64
        norm_layer = get_norm_layer(self.norm_type)
        kernel_initializer = tf.initializers.random_normal(
            stddev=self.init_gain)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            padded_input = tf.pad(
                inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

            conv1 = tf.layers.conv2d(padded_input, ngf, (7, 7), strides=(1, 1), padding="valid",
                                     kernel_initializer=kernel_initializer, name="conv1")
            conv1 = norm_layer(conv1, training=training, name="conv1_bn")
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(conv1, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                     kernel_initializer=kernel_initializer, name="conv2")
            conv2 = norm_layer(conv2, training=training, name="conv2_bn")
            conv2 = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(conv2, ngf * 4, (3, 3), strides=(2, 2), padding="same",
                                     kernel_initializer=kernel_initializer, name="conv3")
            conv3 = norm_layer(conv3, training=training, name="conv3_bn")
            conv3 = tf.nn.relu(conv3)

            res_out = conv3
            for r in range(self.num_blocks):
                res_out = build_residual_block(
                    res_out, ngf * 4, norm_layer=norm_layer, use_dropout=self.use_dropout, init_gain=self.init_gain, training=training, name="res" + str(r))

            convt1 = tf.layers.conv2d_transpose(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                                kernel_initializer=kernel_initializer, name="convt1")
            # convt1 = deconv2d(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
            #                                     kernel_initializer=kernel_initializer, name="convt1")

            convt1 = norm_layer(
                convt1, training=training, name="convt1_bn")
            convt1 = tf.nn.relu(convt1)

            convt2 = tf.layers.conv2d_transpose(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
                                                kernel_initializer=kernel_initializer, name="convt2")
            # convt2 = deconv2d(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
            #                                     kernel_initializer=kernel_initializer, name="convt2")
            convt2 = norm_layer(
                convt2, training=training, name="convt2_bn")
            convt2 = tf.nn.relu(convt2)

            padded_output = tf.pad(
                convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            output = tf.layers.conv2d(padded_output, self.img_nc, (7, 7), strides=(1, 1), padding="valid",
                                      kernel_initializer=kernel_initializer, activation=tf.tanh, name="output")
            return output


def generator_resnet128(input_, noise=None, norm_type='instance_norm', num_blocks=6, skip=False, name='gen_res', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        ngf = 64
        padded_input = tf.pad(
            input_, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        conv1 = conv2d(padded_input, ngf, kernal=(7, 7),
                       strides=(1, 1), padding="VALID", name="conv1")
        conv1 = norm_layer(conv1, name="conv1_bn")
        conv1 = tf.nn.relu(conv1)

        conv2 = conv2d(conv1, ngf * 2, kernal=(3, 3),
                       strides=(2, 2), padding="SAME", name="conv2")
        conv2 = norm_layer(conv2, name="conv2_bn")
        conv2 = tf.nn.relu(conv2)

        conv3 = conv2d(conv2, ngf * 4, kernal=(3, 3),
                       strides=(2, 2), padding="SAME", name="conv3")
        conv3 = norm_layer(conv3, name="conv3_bn")
        conv3 = tf.nn.relu(conv3)

        res_out = conv3
        for r in range(num_blocks):
            res_out = build_residual_block(
                res_out, ngf * 4, norm_layer=norm_layer, use_dropout=False, name="res" + str(r))

        convt1 = convt2d(res_out, (None, 64, 64, ngf * 2), kernal=(3, 3),
                         strides=(2, 2), padding="SAME", name="convt1")
        convt1 = norm_layer(convt1, name="convt1_bn")
        convt1 = tf.nn.relu(convt1)

        convt2 = convt2d(convt1, (None, 128, 128, ngf), kernal=(
            3, 3), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        convt2 = tf.nn.relu(convt2)

        padded_output = tf.pad(
            convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        output = conv2d(padded_output, 3, kernal=(7, 7),
                        strides=(1, 1), padding="VALID", name="output")

        if skip:
            output = output + input_
        output = tf.tanh(output)

        return output


def generator_resnet(input_, noise=None, norm_type='instance_norm', num_blocks=9, init_gain=0.02, use_dropout=False, name='gen_res'):

    kernel_initializer = tf.initializers.random_normal(stddev=init_gain)

    norm_layer = get_norm_layer(norm_type)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ngf = 64
        padded_input = tf.pad(
            input_, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        conv1 = tf.layers.conv2d(padded_input, ngf, (7, 7), strides=(1, 1), padding="valid",
                                 kernel_initializer=kernel_initializer, name="conv1")
        conv1 = norm_layer(conv1, name="conv1_bn")
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                 kernel_initializer=kernel_initializer, name="conv2")
        conv2 = norm_layer(conv2, name="conv2_bn")
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, ngf * 4, (3, 3), strides=(2, 2), padding="same",
                                 kernel_initializer=kernel_initializer, name="conv3")
        conv3 = norm_layer(conv3, name="conv3_bn")
        conv3 = tf.nn.relu(conv3)

        res_out = conv3
        for r in range(num_blocks):
            res_out = build_residual_block(
                res_out, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, init_gain=init_gain, name="res" + str(r))

        convt1 = tf.layers.conv2d_transpose(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer=kernel_initializer, name="convt1")
        # convt1 = deconv2d(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
        #                                     kernel_initializer=kernel_initializer, name="convt1")

        convt1 = norm_layer(convt1, name="convt1_bn")
        convt1 = tf.nn.relu(convt1)

        convt2 = tf.layers.conv2d_transpose(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer=kernel_initializer, name="convt2")
        # convt2 = deconv2d(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
        #                                     kernel_initializer=kernel_initializer, name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        convt2 = tf.nn.relu(convt2)

        padded_output = tf.pad(
            convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        output = tf.layers.conv2d(padded_output, 3, (7, 7), strides=(1, 1), padding="valid",
                                  kernel_initializer=kernel_initializer, activation=tf.tanh, name="output")
        return output


def generator_resnet32(input_, noise=None, nc=3, norm_type='instance_norm', num_blocks=2, init_gain=0.02, use_dropout=False, name='gen_res'):

    kernel_initializer = tf.initializers.random_normal(stddev=init_gain)

    norm_layer = get_norm_layer(norm_type)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ngf = 64
        padded_input = tf.pad(
            input_, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        conv1 = tf.layers.conv2d(padded_input, ngf, (7, 7), strides=(1, 1), padding="valid",
                                 kernel_initializer=kernel_initializer, name="conv1")
        conv1 = norm_layer(conv1, name="conv1_bn")
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                 kernel_initializer=kernel_initializer, name="conv2")
        conv2 = norm_layer(conv2, name="conv2_bn")
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, ngf * 4, (3, 3), strides=(2, 2), padding="same",
                                 kernel_initializer=kernel_initializer, name="conv3")
        conv3 = norm_layer(conv3, name="conv3_bn")
        conv3 = tf.nn.relu(conv3)

        res_out = conv3
        for r in range(num_blocks):
            res_out = build_residual_block(
                res_out, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout, init_gain=init_gain, name="res" + str(r))

        convt1 = tf.layers.conv2d_transpose(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer=kernel_initializer, name="convt1")
        # convt1 = deconv2d(res_out, ngf * 2, (3, 3), strides=(2, 2), padding="same",
        #                                     kernel_initializer=kernel_initializer, name="convt1")

        convt1 = norm_layer(convt1, name="convt1_bn")
        convt1 = tf.nn.relu(convt1)

        convt2 = tf.layers.conv2d_transpose(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer=kernel_initializer, name="convt2")
        # convt2 = deconv2d(convt1, ngf, (3, 3), strides=(2, 2), padding="same",
        #                                     kernel_initializer=kernel_initializer, name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        convt2 = tf.nn.relu(convt2)

        padded_output = tf.pad(
            convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        output = tf.layers.conv2d(padded_output, nc, (7, 7), strides=(1, 1), padding="valid",
                                  kernel_initializer=kernel_initializer, activation=tf.tanh, name="output")
        return output
