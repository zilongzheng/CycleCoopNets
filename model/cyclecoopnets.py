from __future__ import division
from __future__ import print_function

import os
import time
import math
from six.moves import xrange
import numpy as np
import tensorflow as tf
from model import modules
from utils.data_util import saveSampleImages, mkdir
from utils.logger import Logger, numpy_to_image
from data.unaligned_data import preprocess_image
from PIL import Image
from utils.data_util import eval_psnr, eval_ssim
import json

FLAGS = tf.app.flags.FLAGS

class CycleCoopNets(object):
    def __init__(self, num_epochs=200, epoch_start_decay=100, image_size=64, batch_size=100, nTileRow=12, nTileCol=12, z_size=100,
                 d_lr=0.001, g_lr=0.0001, beta1=0.5, init_gain=0.002, use_dropout=False,
                 des_step_size=0.002, des_sample_steps=10, des_refsig=0.016, num_vis = 1, img_nc=3,
                 gen_step_size=0.1, gen_sample_steps=10, gen_refsig=0.3, gen_num_blocks=9,
                 norm_type='batch_norm', lambdaA=20, lambdaB=20, lambda_sem=1, log_step=10, isTrain=True,
                 output_dir='./output_dir', args=None):

        self.args = args
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.nTileRow = nTileRow
        self.nTileCol = nTileCol
        self.num_chain = nTileRow * nTileCol
        self.beta1 = beta1
        self.init_gain = init_gain
        self.epoch_start_decay = epoch_start_decay

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.delta1 = des_step_size
        self.sigma1 = des_refsig
        self.delta2 = gen_step_size
        self.sigma2 = gen_refsig
        self.t1 = des_sample_steps
        self.t2 = gen_sample_steps

        self.log_step = log_step

        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'log')
        self.sample_dir = os.path.join(output_dir, 'synthesis')
        self.model_dir = os.path.join(output_dir, 'checkpoints')

        self.img_nc = img_nc
        self.num_fc = 1
        self.norm_type = norm_type
        self.use_dropout = use_dropout


        self.z_size = z_size
        self.z_scale = 0.003
        self.lambda_A = lambdaA # weight for cycle loss (A -> B -> A)
        self.lambda_B = lambdaB # weight for cycle loss (B -> A -> B)
        self.lambda_identity = 0 # for photo generation from paintings
        self.log_interval = 100
        self.eval_step = 5
        self.num_vis = num_vis
        self.des_type = '3_layer'

        self.gen_A = modules.ResnetGenerator(img_nc=self.img_nc, norm_type=self.norm_type, num_blocks=gen_num_blocks, init_gain=self.init_gain, use_dropout=self.use_dropout, name='gen_A')
        self.gen_B = modules.ResnetGenerator(img_nc=self.img_nc, norm_type=self.norm_type, num_blocks=gen_num_blocks, init_gain=self.init_gain, use_dropout=self.use_dropout, name='gen_B')
        self.des_A = modules.Descriptor(self.num_fc, self.init_gain, net_type=self.des_type, name='des_A')
        self.des_B = modules.Descriptor(self.num_fc, self.init_gain, net_type=self.des_type, name='des_B')

        self.isTrain = isTrain

        if isTrain:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            with open("%s/config.txt" % self.output_dir, "w") as f:
                for k in self.__dict__:
                    if not k == 'args':
                        f.write(str(k) + ':' + str(getattr(self, k)) + '\n')
                
                for k in self.args:
                    if k not in self.__dict__:
                        f.write(str(k) + ':' + str(self.args[k]) + '\n')

        self.build_model()

    def build_model(self):
        self.obs_a = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.img_nc], name='obs_a')
        self.obs_b = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.img_nc], name='obs_b')

        self.z_a = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z_a')
        self.z_b = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z_b')
        self.training = tf.placeholder(tf.bool, name='training')

        # self.gen_res_b = self.generator(self.obs_a, self.z_a, name='gen_A') # G_A: A -> B
        # self.gen_res_a = self.generator(self.obs_b, self.z_b, name='gen_B') # G_B: B -> A
        self.gen_res_b = self.gen_A(self.obs_a, self.z_a, training=self.training) # G_A: A -> B
        self.gen_res_a = self.gen_B(self.obs_b, self.z_b, training=self.training) # G_B: B -> A

        self.rec_a = self.gen_B(self.gen_res_b, self.z_b, training=self.training) # G_B(G_A(A)): A -> B -> A
        self.rec_b = self.gen_A(self.gen_res_a, self.z_a, training=self.training) # G_A(G_B(A)): B -> A -> B

        # symbolic langevins
        self.syn_a = self.langevin_dynamics_descriptor(self.gen_res_a, des_net=self.des_A, name='langevin_A') # syn_a = gen_res_a
        self.syn_b = self.langevin_dynamics_descriptor(self.gen_res_b, des_net=self.des_B, name='langevin_B') # syn_b = gen_res_b
    
        obs_res_a = self.des_A(self.obs_a, training=self.training) # F_A
        syn_res_a = self.des_A(self.syn_a, training=self.training) # F_A(syn_res_a) = F_A(obs_res_a)
        self.grad_a_op = tf.reduce_mean(tf.abs(tf.gradients(obs_res_a, self.obs_a)[0]))

        obs_res_b = self.des_B(self.obs_b, training=self.training) # F_B
        syn_res_b = self.des_B(self.syn_b, training=self.training) # F_B(syn_res_b) = F_B(obs_res_b)
        self.grad_b_op = tf.reduce_mean(tf.abs(tf.gradients(obs_res_b, self.obs_b)[0]))

        # descriptor variables
        des_A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='des_A')
        des_B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='des_B')
        # generator variables
        gen_A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_A')
        gen_B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_B')

        self.var_list = des_A_vars + des_B_vars + gen_A_vars + gen_B_vars


        if self.isTrain:            
            # with open("%s/descriptor.txt" % self.output_dir, "w") as f:
            #     f.write(des_net)

            global_step = tf.Variable(0, trainable=False)

            self.des_loss_A = tf.reduce_mean(tf.subtract(tf.reduce_mean(syn_res_a, axis=0), tf.reduce_mean(obs_res_a, axis=0)))
            self.des_loss_B = tf.reduce_mean(tf.subtract(tf.reduce_mean(syn_res_b, axis=0), tf.reduce_mean(obs_res_b, axis=0)))
            # self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(self.des_loss)
            tf.summary.scalar('des_loss_A', self.des_loss_A)
            tf.summary.scalar('des_loss_B', self.des_loss_B)

            self.d_lr_decay = tf.train.polynomial_decay(self.d_lr, global_step=global_step, decay_steps=self.num_epochs-self.epoch_start_decay, end_learning_rate=1e-5)
            des_optim = tf.train.AdamOptimizer(self.d_lr_decay, beta1=self.beta1)
            self.des_A_step = des_optim.minimize(self.des_loss_A, var_list=des_A_vars)
            self.des_B_step = des_optim.minimize(self.des_loss_B, var_list=des_B_vars)

            # gen_A_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen_A')]
            # gen_B_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen_B')]

            self.gen_loss_A = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.syn_b - self.gen_res_b)) # syn_a = langevin(G(A))
            self.gen_loss_B = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.syn_a - self.gen_res_a))
            # self.gen_loss_A = tf.reduce_mean(tf.square(self.syn_b - self.gen_res_b))
            # self.gen_loss_B = tf.reduce_mean(tf.square(self.syn_a - self.gen_res_a))

            tf.summary.scalar('gen_loss_A', self.gen_loss_A)
            tf.summary.scalar('gen_loss_B', self.gen_loss_B)

            # self.gen_loss_A = tf.reduce_mean(tf.square(self.syn_b - self.gen_res_b)) # syn_a = langevin(G(A))
            # self.gen_loss_B = tf.reduce_mean(tf.square(self.syn_a - self.gen_res_a))

            self.loss_cycle_A = tf.reduce_mean(tf.abs(self.obs_a - self.rec_a)) * self.lambda_A # ||G_B(G_A(A)) - A||
            self.loss_cycle_B = tf.reduce_mean(tf.abs(self.obs_b - self.rec_b)) * self.lambda_B # ||G_A(G_B(B)) - B||

            tf.summary.scalar('loss_cycle_A', self.loss_cycle_A)
            tf.summary.scalar('loss_cycle_B', self.loss_cycle_B)

            if self.lambda_identity > 0:
                self.idt_a = self.gen_A(self.obs_b, self.z_a) # ||G_A(B)- B||
                self.loss_idt_A = tf.reduce_mean(tf.abs(self.idt_a - self.obs_b)) * self.lambda_B * self.lambda_identity
                self.idt_b = self.gen_B(self.obs_a, self.z_b) # ||G_B(A) - A||
                self.loss_idt_B = tf.reduce_mean(tf.abs(self.idt_b - self.obs_a)) * self.lambda_A * self.lambda_identity
            else:
                self.loss_idt_A = 0.0
                self.loss_idt_B = 0.0


            self.loss_G = self.gen_loss_A + self.gen_loss_B + self.loss_cycle_A + self.loss_cycle_B \
                            + self.loss_idt_A + self.loss_idt_B
            tf.summary.scalar('loss_G', self.loss_G)

            self.g_lr_decay = tf.train.polynomial_decay(self.g_lr, global_step=global_step, decay_steps=self.num_epochs-self.epoch_start_decay, end_learning_rate=0.0)
            gen_optim = tf.train.AdamOptimizer(self.g_lr_decay, beta1=self.beta1)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.gen_step = gen_optim.minimize(self.loss_G, var_list=gen_A_vars + gen_B_vars)

            self.update_lr = tf.assign_add(global_step, 1)
            self.summary_op = tf.summary.merge_all()


    def langevin_dynamics_descriptor(self, syn_arg, des_net, name="langevin_dynamics_descriptor"):
        def cond(i, syn):
            return tf.less(i, self.t1)

        def body(i, syn):
            noise = tf.random_normal(shape=tf.shape(syn), name='noise', stddev=self.delta1)
            syn_res = des_net(syn, training=self.training)
            grad = tf.gradients(syn_res, syn, stop_gradients=syn, name='grad_des')[0]
            # syn = syn + 0.5 * self.delta1 * self.delta1 * grad + noise
            syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad)
            return tf.add(i, 1), syn

        with tf.name_scope(name):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg], back_prop=False, parallel_iterations=1)
            return syn


    def train(self, sess, train_data, test_data, ckpt=None):
        """Training Function"""

        num_data = len(train_data)

        num_batches = num_data // self.batch_size

        # initialize training
        # sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        sess.run(tf.global_variables_initializer())

        # if ckpt is not None:
        #     cls_vars = [var for var in tf.trainable_variables() if var.name.startswith(self.cls.name)]
        #     saver_pretrain = tf.train.Saver(cls_vars)
        #     saver_pretrain.restore(sess, ckpt)
        #     print('Successfully restore checkpoint: {}'.format(ckpt))

        # make graph immutable
        tf.get_default_graph().finalize()

        ref_A, ref_B = test_data[np.arange(self.nTileRow * self.nTileCol)]

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        saveSampleImages(ref_A, "%s/ref_A.png" % self.sample_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)
        saveSampleImages(ref_B, "%s/ref_B.png" % self.sample_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)
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

            d_loss_A_epoch, d_loss_B_epoch, g_loss_epoch = [], [], []

            print('Train on epoch {}'.format(epoch))

            total_time = 0.0

            for i in range(num_batches):
                index = np.arange(i * self.batch_size, (i + 1) * self.batch_size)
                obs_a, obs_b = train_data[index]

                st_time = time.time()

                feed_dict = {
                    self.obs_a: obs_a,
                    self.obs_b: obs_b,
                    self.training: True
                }

                ## A -> B
                z_a = np.random.normal(size=(len(obs_a), self.z_size), scale=self.z_scale)
                feed_dict.update({self.z_a: z_a})
                # I_B' = G_A(I_A, Z_A)
                # I_B'' = Langevin_B(I_B')
                [gen_res_b, syn_b] = sess.run([self.gen_res_b, self.syn_b], feed_dict=feed_dict)

                # Update D_B
                # min(D_B(I_B'') - D_A(I_B))
                des_loss_B = sess.run([self.des_loss_B,  self.des_B_step], feed_dict=feed_dict)[0]
            
                # B -> A
                z_b = np.random.normal(size=(len(obs_b), self.z_size), scale=self.z_scale)
                feed_dict.update({self.z_b: z_b})
                # I_A' = G_B(I_B, Z_B)
                # I_A'' = Langevin_A(I_A')
                [gen_res_a, syn_a] = sess.run([self.gen_res_a, self.syn_a], feed_dict=feed_dict)

                # Update 
                # min(D_A(I_A'') - D_A(I_A))
                des_loss_A = sess.run([self.des_loss_A,  self.des_A_step], feed_dict=feed_dict)[0]

                ## A -> B -> A
                # Rec_A = G_B(G_A(I_B))
                ## B -> A -> B
                # Rec_B = G_A(G_B(I_A))
                rec_a, rec_b = sess.run([self.rec_a, self.rec_b], feed_dict=feed_dict)

                # Step G2: update G net
                # lambdaA * || Rec_A - I_A|| + || I_B'' - I_B'||
                # lambdaB * || Rec_B - I_B|| + || I_A'' - I_A'||
                g_loss, _ = sess.run([self.loss_G, self.gen_step], feed_dict=feed_dict)
                # print(g_loss)

                ed_time = time.time()
                print(ed_time - st_time)
                total_time += ed_time - st_time

                d_loss_A_epoch.append(des_loss_A)
                d_loss_B_epoch.append(des_loss_B)
                g_loss_epoch.append(g_loss)

                sample_results_A.append(syn_a)
                sample_results_B.append(syn_b)
                gen_results_A.append(gen_res_a)
                gen_results_B.append(gen_res_b)
                rec_results_A.append(rec_a)
                rec_results_B.append(rec_b)

                iters += 1

                if iters % self.log_interval == 0:
                    print('[Iter {:06d}][des loss A: {:.4f}][des loss B: {:.4f}][gen loss: {:.4f}]'.format(
                        iters, des_loss_A, des_loss_B, g_loss
                    ))
                    # print('[Iter {:06d}][des loss A: {:.4f}][des loss B: {:.4f}][gen loss: {:.4f}][sem loss: {:.4f}]'.format(
                    #     iters, des_loss_A, des_loss_B, g_loss, sem_loss
                    # ))
                    summary = sess.run(self.summary_op, feed_dict=feed_dict)
                    self.logger.add_summary(summary, iters)
                    # test_img_a, test_img_b = train_data.sample_image_pair()
                    # self.eval(sess, iters, test_img_a, test_img_b)
                    rand_int = np.random.randint(0, len(obs_b), size=self.num_vis)
                    selected_ba = [np.split(img, img.shape[0], axis=0) for img in [obs_b[rand_int], syn_a[rand_int], gen_res_a[rand_int], rec_b[rand_int]]]
                    selected_ba = [np.concatenate(img_arr, axis=1).squeeze() for img_arr in selected_ba]

                    selected_ab = [np.split(img, img.shape[0], axis=0) for img in [obs_a[rand_int], syn_b[rand_int], gen_res_b[rand_int], rec_a[rand_int]]]
                    selected_ab = [np.concatenate(img_arr, axis=1).squeeze() for img_arr in selected_ab]

                    combined_image_a = [np.array(Image.fromarray(numpy_to_image(img))) for img in selected_ba]
                    combined_image_b = [np.array(Image.fromarray(numpy_to_image(img))) for img in selected_ab]

                    combined_image_a = np.concatenate(combined_image_a, axis=1)
                    combined_image_b = np.concatenate(combined_image_b, axis=1)

                    self.logger.image_summary('B2A (input, syn, gen, rec)', combined_image_a, iters)
                    self.logger.image_summary('A2B (input, syn, gen, rec)', combined_image_b, iters)

            train_data.shuffle()

            print('Total time: {}'.format(total_time))

            d_lr, g_lr = sess.run([self.d_lr_decay, self.g_lr_decay])

            if epoch >= self.epoch_start_decay:
                sess.run(self.update_lr)

            log['des_loss_A_avg'], log['des_loss_B_avg'], log['gen_loss_avg']= float(np.mean(d_loss_A_epoch)), float(np.mean(d_loss_B_epoch)), float(np.mean(g_loss_epoch))
            log['d_lr'] = float(d_lr)
            log['g_lr'] = float(g_lr)

            # log['epoch'] = epoch
            log_history.append(log)

            end_time = time.time()
            log_msg = '[Epoch {:3d}]'.format(epoch)
            for k in sorted(log):
                log_msg += '[{}: {:.4f}]'.format(k, log[k])
            log_msg += '[time: {:.2f}s]'.format(end_time - start_time)
            # print('Epoch #{:d}, des lr: {:.6f}, avg.descriptor loss A: {:.4f}, avg.descriptor loss B: {:.4f}, gen lr: {:.6f}, avg.generator loss: {:.4f}, cls lr: {:.6f}, avg.classifier loss: {:.4f}, '
            #       'time: {:.2f}s'.format(epoch, d_lr, log['des_loss_A_avg'], log['des_loss_B_avg'], g_lr, log['gen_loss_avg'], c_lr, log['cls_loss_avg'], end_time - start_time))
            print(log_msg)

            with open('%s/log.json' % self.log_dir, 'w') as f:
                json.dump(log_history, f, ensure_ascii=False)

            if np.isnan(log['gen_loss_avg']) or  log['gen_loss_avg'] > 100:
                break

            if epoch % self.log_step == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)
            
            if epoch % self.eval_step == 0:
                self.eval(sess, epoch, test_data)

                # sample_results_A = np.concatenate(sample_results_A, axis=0)
                # sample_results_B = np.concatenate(sample_results_B, axis=0)
                # gen_results_A = np.concatenate(gen_results_A, axis=0)
                # gen_results_B = np.concatenate(gen_results_B, axis=0)

                # rec_results_A = np.concatenate(rec_results_A, axis=0)
                # rec_results_B = np.concatenate(rec_results_B, axis=0)

                # if not os.path.exists(sample_A_dir):
                #     os.makedirs(sample_A_dir)
                # saveSampleImages(sample_results_A, "%s/des_A_%03d.png" % (sample_A_dir, epoch), row_num=self.nTileRow,
                #                  col_num=self.nTileCol)
                # saveSampleImages(gen_results_A, "%s/gen_A_%03d.png" % (sample_A_dir, epoch), row_num=self.nTileRow,
                #                  col_num=self.nTileCol)

                # if not os.path.exists(sample_B_dir):
                #     os.makedirs(sample_B_dir)
                # saveSampleImages(sample_results_B, "%s/des_B_%03d.png" % (sample_B_dir, epoch), row_num=self.nTileRow,
                #                  col_num=self.nTileCol)
                # saveSampleImages(gen_results_B, "%s/gen_B_%03d.png" % (sample_B_dir, epoch), row_num=self.nTileRow,
                #                  col_num=self.nTileCol)

                # if not os.path.exists(rec_A_dir):
                #     os.makedirs(rec_A_dir)
                # saveSampleImages(rec_results_A, "%s/rec_A_%03d.png" % (rec_A_dir, epoch), row_num=self.nTileRow,
                #                  col_num=self.nTileCol)
                # if not os.path.exists(rec_B_dir):
                #     os.makedirs(rec_B_dir)
                # saveSampleImages(rec_results_B, "%s/rec_B_%03d.png" % (rec_B_dir, epoch), row_num=self.nTileRow,
                #                  col_num=self.nTileCol)

    def eval(self, sess, epoch, test_data):
        print('Evaluate on epoch {}'.format(epoch))
        # log_history = []
        sample_A_dir = os.path.join(self.sample_dir, 'sample_A')
        sample_B_dir = os.path.join(self.sample_dir, 'sample_B')
        rec_A_dir = os.path.join(self.sample_dir, 'rec_A')
        rec_B_dir = os.path.join(self.sample_dir, 'rec_B')

        # eval
        sample_results_A = []
        sample_results_B = []
        gen_results_A = []
        gen_results_B = []
        rec_results_A = []
        rec_results_B = []

        num_data = len(test_data) if FLAGS.eval_all else (self.nTileCol * self.nTileRow)
        num_batches = int(math.ceil(num_data / self.batch_size))

        des_A_psnr, des_B_psnr, gen_A_psnr, gen_B_psnr = [], [], [], []

        for i in range(num_batches):
            index = np.arange(i * self.batch_size, min((i + 1) * self.batch_size, num_data))
            obs_a, obs_b = test_data[index]

            feed_dict = {
                self.obs_a: obs_a,
                self.obs_b: obs_b,
                self.training: False
            }

            ## A -> B
            z_a = np.random.normal(size=(len(obs_a), self.z_size), scale=self.z_scale)
            feed_dict.update({self.z_a: z_a})
            # I_B' = G_A(I_A, Z_A)
            # I_B'' = Langevin_B(I_B')
            [gen_res_b, syn_b] = sess.run([self.gen_res_b, self.syn_b], feed_dict=feed_dict)

            # Update D_B
            # min(D_B(I_B'') - D_A(I_B))
            # des_loss_B = sess.run([self.des_loss_B,  self.des_B_step], feed_dict=feed_dict)[0]
        
            # B -> A
            z_b = np.random.normal(size=(len(obs_b), self.z_size), scale=self.z_scale)
            feed_dict.update({self.z_b: z_b})
            # I_A' = G_B(I_B, Z_B)
            # I_A'' = Langevin_A(I_A')
            [gen_res_a, syn_a] = sess.run([self.gen_res_a, self.syn_a], feed_dict=feed_dict)

            # Update 
            # min(D_A(I_A'') - D_A(I_A))
            # des_loss_A = sess.run([self.des_loss_A,  self.des_A_step], feed_dict=feed_dict)[0]

            ## A -> B -> A
            # Rec_A = G_B(G_A(I_B))
            ## B -> A -> B
            # Rec_B = G_A(G_B(I_A))
            rec_a, rec_b = sess.run([self.rec_a, self.rec_b], feed_dict=feed_dict)

            # Step G2: update G net
            # lambdaA * || Rec_A - I_A|| + || I_B'' - I_B'||
            # lambdaB * || Rec_B - I_B|| + || I_A'' - I_A'||
            # g_loss, sem_loss = sess.run([self.loss_G, self.loss_sem, self.gen_step], feed_dict=feed_dict)[:2]
            # print(g_loss)

            # cls_loss = sess.run([self.loss_cls, self.cls_step], feed_dict=feed_dict)[0]

            # d_loss_A_epoch.append(des_loss_A)
            # d_loss_B_epoch.append(des_loss_B)
            # g_loss_epoch.append(g_loss)
            # c_loss_epoch.append(cls_loss)

            sample_results_A.append(syn_a)
            sample_results_B.append(syn_b)
            gen_results_A.append(gen_res_a)
            gen_results_B.append(gen_res_b)
            rec_results_A.append(rec_a)
            rec_results_B.append(rec_b)

            if FLAGS.eval_psnr:
                gen_A_psnr += eval_psnr(obs_a, gen_res_a, a_min=-1, a_max=1)
                gen_B_psnr += eval_psnr(obs_b, gen_res_b, a_min=-1, a_max=1)
                des_A_psnr += eval_psnr(obs_a, syn_a, a_min=-1, a_max=1)
                des_B_psnr += eval_psnr(obs_b, syn_b, a_min=-1, a_max=1)
        
        if FLAGS.eval_psnr:
            print('[Epoch {:03d}][gen_A_psnr: {:.4f}][gen_B_psnr: {:.4f}][des_A_psnr: {:.4f}][des_B_psnr: {:.4f}]'.format(epoch, np.mean(gen_A_psnr), np.mean(gen_B_psnr), np.mean(des_A_psnr), np.mean(des_B_psnr)))

        sample_results_A = np.concatenate(sample_results_A, axis=0)
        sample_results_B = np.concatenate(sample_results_B, axis=0)
        gen_results_A = np.concatenate(gen_results_A, axis=0)
        gen_results_B = np.concatenate(gen_results_B, axis=0)

        rec_results_A = np.concatenate(rec_results_A, axis=0)
        rec_results_B = np.concatenate(rec_results_B, axis=0)

        if not os.path.exists(sample_A_dir):
            os.makedirs(sample_A_dir)
        saveSampleImages(sample_results_A, "%s/des_A_%03d.png" % (sample_A_dir, epoch), row_num=self.nTileRow,
                            col_num=self.nTileCol)
        saveSampleImages(gen_results_A, "%s/gen_A_%03d.png" % (sample_A_dir, epoch), row_num=self.nTileRow,
                            col_num=self.nTileCol)

        if not os.path.exists(sample_B_dir):
            os.makedirs(sample_B_dir)
        saveSampleImages(sample_results_B, "%s/des_B_%03d.png" % (sample_B_dir, epoch), row_num=self.nTileRow,
                            col_num=self.nTileCol)
        saveSampleImages(gen_results_B, "%s/gen_B_%03d.png" % (sample_B_dir, epoch), row_num=self.nTileRow,
                            col_num=self.nTileCol)

        if not os.path.exists(rec_A_dir):
            os.makedirs(rec_A_dir)
        saveSampleImages(rec_results_A, "%s/rec_A_%03d.png" % (rec_A_dir, epoch), row_num=self.nTileRow,
                            col_num=self.nTileCol)
        if not os.path.exists(rec_B_dir):
            os.makedirs(rec_B_dir)
        saveSampleImages(rec_results_B, "%s/rec_B_%03d.png" % (rec_B_dir, epoch), row_num=self.nTileRow,
                            col_num=self.nTileCol)


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

            z_a = np.random.normal(size=(len(obs_a), self.z_size), scale=self.z_scale)
            [gen_res_b, syn_b] = sess.run([self.gen_res_b, self.syn_b], feed_dict={self.obs_a: obs_a, self.z_a: z_a})

            z_b = np.random.normal(size=(len(obs_b), self.z_size), scale=self.z_scale)
            [gen_res_a, syn_a] = sess.run([self.gen_res_a, self.syn_a], feed_dict={self.obs_b: obs_b, self.z_b: z_b})

            rec_a, rec_b = sess.run([self.rec_a, self.rec_b], feed_dict={
                self.obs_a: obs_a, self.z_a: z_a, self.obs_b: obs_b, self.z_b: z_b
            })
