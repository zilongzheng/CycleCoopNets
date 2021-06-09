from __future__ import division
from __future__ import print_function

import os
import time
import math
import numpy as np
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
from model import modules
from utils.data_util import saveSampleImages, mkdir
from utils.logger import Logger, numpy_to_image
from data.unaligned_data import preprocess_image
from PIL import Image
import json

FLAGS = tf.app.flags.FLAGS


class CycleCoopNets(object):
    def __init__(self, output_root='./output', isTrain=True):
        self.isTrain = isTrain

        # model params
        self.img_size = FLAGS.img_size
        self.img_nc = FLAGS.img_nc
        self.norm_type = FLAGS.norm_type
        self.use_dropout = not FLAGS.no_dropout
        self.init_gain = FLAGS.init_gain
        self.gen_num_blocks = FLAGS.gen_num_blocks

        # mcmc params
        self.mcmc_delta = FLAGS.ebm_step_size
        self.mcmc_refsig = FLAGS.ebm_refsig
        self.mcmc_T = FLAGS.ebm_sample_steps

        self.output_dir = output_root
        self.log_dir = os.path.join(output_root, 'log')
        self.sample_dir = os.path.join(output_root, 'synthesis')
        self.model_dir = os.path.join(output_root, 'checkpoints')

        self.lambda_A = FLAGS.lambdaA  # weight for cycle loss (A -> B -> A)
        self.lambda_B = FLAGS.lambdaB  # weight for cycle loss (B -> A -> B)
        # for photo generation from paintings
        self.lambda_identity = FLAGS.lambda_identity


        self.gen_A = modules.ResnetGenerator(img_nc=self.img_nc, norm_type=self.norm_type,
                                             num_blocks=self.gen_num_blocks, init_gain=self.init_gain, use_dropout=self.use_dropout, name='gen_A')
        self.gen_B = modules.ResnetGenerator(img_nc=self.img_nc, norm_type=self.norm_type,
                                             num_blocks=self.gen_num_blocks, init_gain=self.init_gain, use_dropout=self.use_dropout, name='gen_B')
        self.ebm_A = modules.ConvEBM(
            net_type='3_layer', init_gain=self.init_gain, name='ebm_A')
        self.ebm_B = modules.ConvEBM(
            net_type='3_layer', init_gain=self.init_gain, name='ebm_B')

        if isTrain:
            # training params
            self.num_vis = FLAGS.num_vis

            self.batch_size = FLAGS.batch_size
            self.num_epochs = FLAGS.num_epochs
            self.log_interval = FLAGS.log_interval
            self.save_step = FLAGS.save_step

            # optim params
            self.e_lr = FLAGS.e_lr
            self.g_lr = FLAGS.g_lr
            self.beta1 = FLAGS.beta1
            self.epoch_start_decay = FLAGS.epoch_start_decay

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            with open("%s/config.txt" % self.output_dir, "w") as f:
                for k in self.__dict__:
                    f.write(str(k) + ':' + str(getattr(self, k)) + '\n')

        self.build_model()

    def build_model(self):
        self.obs_a = tf.placeholder(
            tf.float32, [None, self.img_size, self.img_size, self.img_nc], name='obs_a')
        self.obs_b = tf.placeholder(
            tf.float32, [None, self.img_size, self.img_size, self.img_nc], name='obs_b')

        self.training = tf.placeholder(tf.bool, name='training')

        self.gen_res_b = self.gen_A(
            self.obs_a, training=self.training)  # G_A: A -> B
        self.gen_res_a = self.gen_B(
            self.obs_b, training=self.training)  # G_B: B -> A

        # G_B(G_A(A)): A -> B -> A
        self.rec_a = self.gen_B(self.gen_res_b, training=self.training)
        # G_A(G_B(A)): B -> A -> B
        self.rec_b = self.gen_A(self.gen_res_a, training=self.training)

        # symbolic langevins
        self.syn_a = self.langevin_dynamics(
            self.gen_res_a, ebm_net=self.ebm_A, name='mcmc_A')  # syn_a = gen_res_a
        self.syn_b = self.langevin_dynamics(
            self.gen_res_b, ebm_net=self.ebm_B, name='mcmc_B')  # syn_b = gen_res_b

        obs_res_a = self.ebm_A(self.obs_a, training=self.training)  # F_A
        # F_A(syn_res_a) = F_A(obs_res_a)
        syn_res_a = self.ebm_A(self.syn_a, training=self.training)
        self.grad_a_op = tf.reduce_mean(
            tf.abs(tf.gradients(obs_res_a, self.obs_a)[0]))

        obs_res_b = self.ebm_B(self.obs_b, training=self.training)  # F_B
        # F_B(syn_res_b) = F_B(obs_res_b)
        syn_res_b = self.ebm_B(self.syn_b, training=self.training)
        self.grad_b_op = tf.reduce_mean(
            tf.abs(tf.gradients(obs_res_b, self.obs_b)[0]))

        # ebm variables
        ebm_A_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='ebm_A')
        ebm_B_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='ebm_B')
        # generator variables
        gen_A_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_A')
        gen_B_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_B')

        self.var_list = ebm_A_vars + ebm_B_vars + gen_A_vars + gen_B_vars

        if self.isTrain:
            # with open("%s/ebmcriptor.txt" % self.output_dir, "w") as f:
            #     f.write(ebm_net)

            global_step = tf.Variable(0, trainable=False)

            self.ebm_loss_A = tf.reduce_mean(tf.subtract(tf.reduce_mean(
                syn_res_a, axis=0), tf.reduce_mean(obs_res_a, axis=0)))
            self.ebm_loss_B = tf.reduce_mean(tf.subtract(tf.reduce_mean(
                syn_res_b, axis=0), tf.reduce_mean(obs_res_b, axis=0)))
            # self.ebm_loss_mean, self.ebm_loss_update = tf.contrib.metrics.streaming_mean(self.ebm_loss)
            tf.summary.scalar('ebm_loss_A', self.ebm_loss_A)
            tf.summary.scalar('ebm_loss_B', self.ebm_loss_B)

            self.e_lr_decay = tf.train.polynomial_decay(
                self.e_lr, global_step=global_step, decay_steps=self.num_epochs-self.epoch_start_decay, end_learning_rate=1e-5)
            ebm_optim = tf.train.AdamOptimizer(
                self.e_lr_decay, beta1=self.beta1)
            self.ebm_A_step = ebm_optim.minimize(
                self.ebm_loss_A, var_list=ebm_A_vars)
            self.ebm_B_step = ebm_optim.minimize(
                self.ebm_loss_B, var_list=ebm_B_vars)

            # gen_A_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen_A')]
            # gen_B_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen_B')]

            self.gen_loss_A = tf.reduce_mean(
                tf.square(self.syn_b - self.gen_res_b))  # syn_a = langevin(G(A))
            self.gen_loss_B = tf.reduce_mean(
                tf.square(self.syn_a - self.gen_res_a))
            # self.gen_loss_A = tf.reduce_mean(tf.square(self.syn_b - self.gen_res_b))
            # self.gen_loss_B = tf.reduce_mean(tf.square(self.syn_a - self.gen_res_a))

            tf.summary.scalar('gen_loss_A', self.gen_loss_A)
            tf.summary.scalar('gen_loss_B', self.gen_loss_B)

            # self.gen_loss_A = tf.reduce_mean(tf.square(self.syn_b - self.gen_res_b)) # syn_a = langevin(G(A))
            # self.gen_loss_B = tf.reduce_mean(tf.square(self.syn_a - self.gen_res_a))

            self.loss_cycle_A = tf.reduce_mean(
                tf.abs(self.obs_a - self.rec_a)) * self.lambda_A  # ||G_B(G_A(A)) - A||
            self.loss_cycle_B = tf.reduce_mean(
                tf.abs(self.obs_b - self.rec_b)) * self.lambda_B  # ||G_A(G_B(B)) - B||

            tf.summary.scalar('loss_cycle_A', self.loss_cycle_A)
            tf.summary.scalar('loss_cycle_B', self.loss_cycle_B)

            if self.lambda_identity > 0:
                self.idt_a = self.gen_A(self.obs_b, self.z_a)  # ||G_A(B)- B||
                self.loss_idt_A = tf.reduce_mean(
                    tf.abs(self.idt_a - self.obs_b)) * self.lambda_B * self.lambda_identity
                self.idt_b = self.gen_B(self.obs_a, self.z_b)  # ||G_B(A) - A||
                self.loss_idt_B = tf.reduce_mean(
                    tf.abs(self.idt_b - self.obs_a)) * self.lambda_A * self.lambda_identity
            else:
                self.loss_idt_A = 0.0
                self.loss_idt_B = 0.0

            self.loss_G = self.gen_loss_A + self.gen_loss_B + self.loss_cycle_A + self.loss_cycle_B \
                + self.loss_idt_A + self.loss_idt_B
            tf.summary.scalar('loss_G', self.loss_G)

            self.g_lr_decay = tf.train.polynomial_decay(
                self.g_lr, global_step=global_step, decay_steps=self.num_epochs-self.epoch_start_decay, end_learning_rate=0.0)
            gen_optim = tf.train.AdamOptimizer(
                self.g_lr_decay, beta1=self.beta1)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.gen_step = gen_optim.minimize(
                    self.loss_G, var_list=gen_A_vars + gen_B_vars)

            self.update_lr = tf.assign_add(global_step, 1)
            self.summary_op = tf.summary.merge_all()

    def langevin_dynamics(self, syn_arg, ebm_net, name="langevin_dynamics"):
        def cond(i, syn):
            return tf.less(i, self.mcmc_T)

        def body(i, syn):
            noise = tf.random_normal(shape=tf.shape(
                syn), name='noise', stddev=self.mcmc_delta)
            syn_res = ebm_net(syn, training=self.training)
            grad = tf.gradients(
                syn_res, syn, stop_gradients=syn, name='grad_ebm')[0]
            # syn = syn + 0.5 * self.delta1 * self.delta1 * grad + noise
            syn = syn - 0.5 * self.mcmc_delta * self.mcmc_delta * \
                (syn / self.mcmc_refsig / self.mcmc_refsig - grad)
            return tf.add(i, 1), syn

        with tf.name_scope(name):
            i = tf.constant(0)
            i, syn = tf.while_loop(
                cond, body, [i, syn_arg], parallel_iterations=1)
            return syn

    def train(self, sess, train_data, test_data, ckpt=None):
        """Training Function"""

        num_data = len(train_data)

        num_batches = num_data // self.batch_size

        # initialize training
        # sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        sess.run(tf.global_variables_initializer())

        # make graph immutable
        tf.get_default_graph().finalize()

        ref_A, ref_B = test_data[np.arange(self.num_vis)]

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        saveSampleImages(ref_A, "%s/ref_A.png" % self.sample_dir,
                         row_num=int(np.sqrt(self.num_vis)), col_num=int(np.sqrt(self.num_vis)))
        saveSampleImages(ref_B, "%s/ref_B.png" % self.sample_dir,
                         row_num=int(np.sqrt(self.num_vis)), col_num=int(np.sqrt(self.num_vis)))
        sample_A_dir = os.path.join(self.sample_dir, 'sample_A')
        sample_B_dir = os.path.join(self.sample_dir, 'sample_B')
        rec_A_dir = os.path.join(self.sample_dir, 'rec_A')
        rec_B_dir = os.path.join(self.sample_dir, 'rec_B')

        iters = 0
        self.logger = Logger(self.log_dir)

        log_history = []

        # train
        for epoch in range(self.num_epochs):
            start_time = time.time()
            log = {}

            sample_results_A = []
            sample_results_B = []
            gen_results_A = []
            gen_results_B = []
            rec_results_A = []
            rec_results_B = []

            print('Train on epoch {}'.format(epoch))

            total_time = 0.0

            for i in range(num_batches):
                index = np.arange(i * self.batch_size,
                                  (i + 1) * self.batch_size)
                obs_a, obs_b = train_data[index]

                st_time = time.time()

                feed_dict = {
                    self.obs_a: obs_a,
                    self.obs_b: obs_b,
                    self.training: True
                }

                # A -> B
                # I_B' = G_A(I_A)
                # I_B'' = Langevin_B(I_B')
                [gen_res_b, syn_b] = sess.run(
                    [self.gen_res_b, self.syn_b], feed_dict=feed_dict)

                # Update D_B
                # min(D_B(I_B'') - D_A(I_B))
                ebm_loss_B = sess.run(
                    [self.ebm_loss_B,  self.ebm_B_step], feed_dict=feed_dict)[0]

                # B -> A
                # I_A' = G_B(I_B)
                # I_A'' = Langevin_A(I_A')
                [gen_res_a, syn_a] = sess.run(
                    [self.gen_res_a, self.syn_a], feed_dict=feed_dict)

                # Update
                # min(D_A(I_A'') - D_A(I_A))
                ebm_loss_A = sess.run(
                    [self.ebm_loss_A,  self.ebm_A_step], feed_dict=feed_dict)[0]

                # A -> B -> A
                # Rec_A = G_B(G_A(I_B))
                # B -> A -> B
                # Rec_B = G_A(G_B(I_A))
                rec_a, rec_b = sess.run(
                    [self.rec_a, self.rec_b], feed_dict=feed_dict)

                # Step G2: update G net
                # lambdaA * || Rec_A - I_A|| + || I_B'' - I_B'||
                # lambdaB * || Rec_B - I_B|| + || I_A'' - I_A'||
                g_loss, _ = sess.run(
                    [self.loss_G, self.gen_step], feed_dict=feed_dict)
                # print(g_loss)

                ed_time = time.time()
                total_time += ed_time - st_time

                sample_results_A.append(syn_a)
                sample_results_B.append(syn_b)
                gen_results_A.append(gen_res_a)
                gen_results_B.append(gen_res_b)
                rec_results_A.append(rec_a)
                rec_results_B.append(rec_b)

                iters += 1

                if iters % self.log_interval == 0:
                    print('[Iter {:06d}][ebm loss A: {:.4f}][ebm loss B: {:.4f}][gen loss: {:.4f}]'.format(
                        iters, ebm_loss_A, ebm_loss_B, g_loss
                    ))
                    summary = sess.run(self.summary_op, feed_dict=feed_dict)
                    self.logger.add_summary(summary, iters)
                    # test_img_a, test_img_b = train_data.sample_image_pair()
                    # self.eval(sess, iters, test_img_a, test_img_b)
                    rand_int = np.random.randint(
                        0, len(obs_b), size=self.num_vis)
                    selected_ba = [np.split(img, img.shape[0], axis=0) for img in [
                        obs_b[rand_int], syn_a[rand_int], gen_res_a[rand_int], rec_b[rand_int]]]
                    selected_ba = [np.concatenate(
                        img_arr, axis=1).squeeze() for img_arr in selected_ba]

                    selected_ab = [np.split(img, img.shape[0], axis=0) for img in [
                        obs_a[rand_int], syn_b[rand_int], gen_res_b[rand_int], rec_a[rand_int]]]
                    selected_ab = [np.concatenate(
                        img_arr, axis=1).squeeze() for img_arr in selected_ab]

                    combined_image_a = [np.array(Image.fromarray(
                        numpy_to_image(img))) for img in selected_ba]
                    combined_image_b = [np.array(Image.fromarray(
                        numpy_to_image(img))) for img in selected_ab]

                    combined_image_a = np.concatenate(combined_image_a, axis=1)
                    combined_image_b = np.concatenate(combined_image_b, axis=1)

                    self.logger.image_summary(
                        'B2A (input, syn, gen, rec)', combined_image_a, iters)
                    self.logger.image_summary(
                        'A2B (input, syn, gen, rec)', combined_image_b, iters)

            train_data.shuffle()

            print('Total time: {}'.format(total_time))

            e_lr, g_lr = sess.run([self.e_lr_decay, self.g_lr_decay])

            if epoch >= self.epoch_start_decay:
                sess.run(self.update_lr)

            log['e_lr'] = float(e_lr)
            log['g_lr'] = float(g_lr)

            # log['epoch'] = epoch
            log_history.append(log)

            end_time = time.time()
            log_msg = '[Epoch {:3d}]'.format(epoch)
            for k in sorted(log):
                log_msg += '[{}: {:.4f}]'.format(k, log[k])
            log_msg += '[time: {:.2f}s]'.format(end_time - start_time)
            print(log_msg)

            with open('%s/log.json' % self.log_dir, 'w') as f:
                json.dump(log_history, f, ensure_ascii=False)

            if np.isnan(log['gen_loss_avg']) or log['gen_loss_avg'] > 100:
                break

            if epoch % self.save_step == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" %
                           (self.model_dir, 'model.ckpt'), global_step=epoch)

    def inference(self, sess, test_data, ckpt):
        """ Testing Function"""
        assert ckpt is not None, 'no valid checkpoint provided'

        test_dir = os.path.join(self.output_dir, 'test')

        rec_A_dir = os.path.join(test_dir, 'rec_A')
        rec_B_dir = os.path.join(test_dir, 'rec_B')
        sample_A_dir = os.path.join(test_dir, 'sample_A2B')
        sample_B_dir = os.path.join(test_dir, 'sample_B2A')
        obs_A_dir = os.path.join(test_dir, 'obs_A')
        obs_B_dir = os.path.join(test_dir, 'obs_B')

        saver = tf.train.Saver()

        saver.restore(sess, ckpt)
        print('Successfully load checkpoint {}'.format(ckpt))

        for bi in range(len(test_data)):
            obs_a, obs_b = test_data[bi]
            obs_a = np.expand_dims(obs_a, axis=0)
            obs_b = np.expand_dims(obs_b, axis=0)

            [gen_res_b, syn_b] = sess.run(
                [self.gen_res_b, self.syn_b], feed_dict={self.obs_a: obs_a})

            [gen_res_a, syn_a] = sess.run(
                [self.gen_res_a, self.syn_a], feed_dict={self.obs_b: obs_b})

            rec_a, rec_b = sess.run([self.rec_a, self.rec_b], feed_dict={
                self.obs_a: obs_a, self.obs_b: obs_b
            })
