from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from model.cyclecoopnets import CycleCoopNets
from data.unaligned_data import UnalignedDataLoader

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('load_size', 286, 'Image size to load images')
tf.flags.DEFINE_integer('image_size', 256, 'Image size to crop images')

tf.flags.DEFINE_integer('batch_size', 1, 'Batch size of training images')
tf.flags.DEFINE_integer('num_vis', 1, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train')
tf.flags.DEFINE_integer('epoch_start_decay', 50, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 10, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 10, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')

# parameters for descriptorNet
tf.flags.DEFINE_float('d_lr', 0.0005, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 30, 'Sample steps for Langevin dynamics of descriptor') # 15
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics') # 0.002
tf.flags.DEFINE_integer('gpu', 0, 'Gpu Device')

tf.flags.DEFINE_float('init_gain', 0.002, 'Step size for descriptor Langevin dynamics')

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0002, 'Initial learning rate for generator')
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 0, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0, 'Step size for generator Langevin dynamics')
tf.flags.DEFINE_integer('gen_num_blocks', 9, 'Sample steps for Langevin dynamics of generator')

tf.flags.DEFINE_string('norm_type', 'instance_norm', '[batch_norm|instance_norm|none]')
tf.flags.DEFINE_integer('lambdaA', 10, 'lambda for cycle')
tf.flags.DEFINE_integer('lambdaB', 10, 'lambda for cycle')

tf.flags.DEFINE_string('dataroot', './input', 'The data directory')
tf.flags.DEFINE_string('category', 'summer2winter_yosemite', 'The name of dataset')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 1, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_boolean('debug', False, 'True if in testing mode')
tf.flags.DEFINE_boolean('eval_psnr', False, 'True if in testing mode')
tf.flags.DEFINE_boolean('eval_all', False, 'True if in testing mode')

tf.flags.DEFINE_string('trainA_postfix', 'trainA', 'The data directory')
tf.flags.DEFINE_string('trainB_postfix', 'trainB', 'The data directory')
tf.flags.DEFINE_string('testA_postfix', 'testA', 'The data directory')
tf.flags.DEFINE_string('testB_postfix', 'testB', 'The data directory')


def main(_):
    RANDOM_SEED = 1
    tf.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    category = FLAGS.category

    if FLAGS.debug:
        output_dir = os.path.join(FLAGS.output_dir, 'debug')
    elif FLAGS.test:
        output_dir = FLAGS.output_dir
    else:
        output_dir = os.path.join(FLAGS.output_dir, '{}_{}'.format(category, time.strftime('%Y-%m-%d_%H-%M-%S')))

    # output_dir = os.path.join(FLAGS.output_dir, '{}_{}'.format(category, time.strftime('%Y-%m-%d_%H-%M-%S')))

    model = CycleCoopNets(
        num_epochs=FLAGS.num_epochs,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        beta1=FLAGS.beta1, epoch_start_decay=FLAGS.epoch_start_decay,
        nTileRow=FLAGS.nTileRow, nTileCol=FLAGS.nTileCol,
        d_lr=FLAGS.d_lr, g_lr=FLAGS.g_lr, init_gain=FLAGS.init_gain,
        des_refsig=FLAGS.des_refsig, gen_refsig=FLAGS.gen_refsig,
        des_step_size=FLAGS.des_step_size, gen_step_size=FLAGS.gen_step_size, gen_num_blocks=FLAGS.gen_num_blocks,
        des_sample_steps=FLAGS.des_sample_steps, gen_sample_steps=FLAGS.gen_sample_steps,
        norm_type=FLAGS.norm_type, lambdaA = FLAGS.lambdaA, lambdaB = FLAGS.lambdaB, num_vis=FLAGS.num_vis,
        log_step=FLAGS.log_step, output_dir=output_dir, isTrain=not FLAGS.test, args=FLAGS.__flags
    )

    gpu_options = tf.GPUOptions(visible_device_list=str(FLAGS.gpu), allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        input_dir_A = os.path.join(FLAGS.dataroot, category, FLAGS.testA_postfix)
        input_dir_B = os.path.join(FLAGS.dataroot, category, FLAGS.testB_postfix)
        print(input_dir_A, input_dir_B)
        test_data = UnalignedDataLoader(input_dir_A, input_dir_B, no_flip=True, max_dataset_size=100 if FLAGS.overfit else 'inf',
            load_size=FLAGS.image_size, crop_size=FLAGS.image_size,  shuffle=False, serial_batches=True)
        # ckpt = '%s/checkpoints/model.ckpt-%d' % (FLAGS.output_dir, FLAGS.ckpt)
        # model.inference(sess, dataset, ckpt)

        input_dir_A = os.path.join(FLAGS.dataroot, category, FLAGS.trainA_postfix)
        input_dir_B = os.path.join(FLAGS.dataroot, category, FLAGS.trainB_postfix)
        print(input_dir_A, input_dir_B)

        if FLAGS.test:
            ckpt = '%s/checkpoints/model.ckpt-%s' % (output_dir, FLAGS.ckpt)
            model.inference(sess, test_data, ckpt)
        else:
            train_data = UnalignedDataLoader(input_dir_A, input_dir_B,  max_dataset_size=100 if FLAGS.overfit else 'inf',
                load_size=FLAGS.load_size, crop_size=FLAGS.image_size, shuffle=True, serial_batches=True)

            model.train(sess, train_data, test_data, FLAGS.ckpt)


if __name__ == '__main__':
    tf.app.run()
