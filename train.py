from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
from model.cyclecoopnets import CycleCoopNets
from data.unaligned_data import UnalignedDataLoader

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('random_seed', 1, 'Random seed for experiments')

tf.flags.DEFINE_integer('load_size', 286, 'Image size to load images')
tf.flags.DEFINE_integer('img_size', 256, 'Image size to crop images')
tf.flags.DEFINE_integer('img_nc', 3, 'Image channels')

tf.flags.DEFINE_integer('batch_size', 1, 'Batch size of training images')
tf.flags.DEFINE_integer('num_vis', 1, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train')
tf.flags.DEFINE_integer('epoch_start_decay', 50, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 10, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 10, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')

# parameters for ebmNet
tf.flags.DEFINE_float('e_lr', 0.0002, 'Initial learning rate for ebm')
tf.flags.DEFINE_float('ebm_refsig', 0.016, 'Standard deviation for reference distribution of ebm')
tf.flags.DEFINE_integer('ebm_sample_steps', 20, 'Sample steps for Langevin dynamics of ebm') # 15
tf.flags.DEFINE_float('ebm_step_size', 0.002, 'Step size for ebm Langevin dynamics') # 0.002
tf.flags.DEFINE_integer('gpu', 0, 'Gpu Device')

tf.flags.DEFINE_float('init_gain', 0.002, 'Scaling factor for weight initialization')

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0002, 'Initial learning rate for generator')
tf.flags.DEFINE_integer('gen_num_blocks', 9, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_boolean('no_dropout', False, 'True if in testing mode')

tf.flags.DEFINE_string('norm_type', 'instance_norm', '[batch_norm|instance_norm|none]')
tf.flags.DEFINE_float('lambdaA', 10, 'lambda for cycle')
tf.flags.DEFINE_float('lambdaB', 10, 'lambda for cycle')
tf.flags.DEFINE_float('lambda_identity', 0, 'lambda for identity loss')

tf.flags.DEFINE_string('dataroot', './input', 'The data directory')
tf.flags.DEFINE_string('category', 'summer2winter_yosemite', 'The name of dataset')
tf.flags.DEFINE_string('output', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_interval', 100, 'Number of iterations to save output results')
tf.flags.DEFINE_integer('save_step', 1, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_boolean('debug', False, 'True if in debug mode')

tf.flags.DEFINE_string('trainA_postfix', 'trainA', 'The data directory')
tf.flags.DEFINE_string('trainB_postfix', 'trainB', 'The data directory')
tf.flags.DEFINE_string('testA_postfix', 'testA', 'The data directory')
tf.flags.DEFINE_string('testB_postfix', 'testB', 'The data directory')


def main(_):
    random_seed = FLAGS.random_seed
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    category = FLAGS.category

    if FLAGS.debug:
        output_root = os.path.join(FLAGS.output, 'debug')
    elif FLAGS.test:
        output_root = FLAGS.output
    else:
        output_root = os.path.join(FLAGS.output, category)

    model = CycleCoopNets(output_root=output_root, isTrain=not FLAGS.test)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > FLAGS.gpu:
    #     tf.config.experimental.set_visible_devices(physical_devices[FLAGS.gpu], 'GPU')
    #     tf.config.experimental.set_memory_growth(physical_devices[FLAGS.gpu], True)
    gpu_options = tf.GPUOptions(visible_device_list=str(FLAGS.gpu), allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        input_dir_A = os.path.join(FLAGS.dataroot, category, FLAGS.testA_postfix)
        input_dir_B = os.path.join(FLAGS.dataroot, category, FLAGS.testB_postfix)

        test_data = UnalignedDataLoader(input_dir_A, input_dir_B, no_flip=True, max_dataset_size=100 if FLAGS.debug else 'inf',
            load_size=FLAGS.img_size, crop_size=FLAGS.img_size,  shuffle=False, serial_batches=True)
        # ckpt = '%s/checkpoints/model.ckpt-%d' % (FLAGS.output_dir, FLAGS.ckpt)

        input_dir_A = os.path.join(FLAGS.dataroot, category, FLAGS.trainA_postfix)
        input_dir_B = os.path.join(FLAGS.dataroot, category, FLAGS.trainB_postfix)

        if FLAGS.test:
            ckpt = '%s/checkpoints/model.ckpt-%s' % (output_dir, FLAGS.ckpt)
            model.inference(sess, test_data, ckpt)
        else:
            train_data = UnalignedDataLoader(input_dir_A, input_dir_B,  max_dataset_size=100 if FLAGS.debug else 'inf',
                load_size=FLAGS.load_size, crop_size=FLAGS.img_size, shuffle=True, serial_batches=True)

            model.train(sess, train_data, test_data, FLAGS.ckpt)


if __name__ == '__main__':
    tf.app.run()
