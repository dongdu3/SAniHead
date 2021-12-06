from __future__ import division
import os
from vex2vdis.ops import *

class vex2vdis(object):
    def __init__(self, sess, epoch=50, vexmaps_init=None, vexmaps_recon=None, batch_size=3, sample_size=1, gf_dim=64,
                 image_size=256, output_size=256, input_c_dim=3, output_c_dim=3, lr=5e-5, beta1=0.5,
                 checkpoint_dir='./vex2vdis/checkpoint'):
        """
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.model_name = 'vex2vdis'

        self.sess = sess
        self.epoch = epoch
        self.vexmaps_init = vexmaps_init
        self.vexmaps_recon = vexmaps_recon
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.image_size = image_size
        self.output_size = output_size

        self.gf_dim = gf_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.lr = lr
        self.beta1 = beta1

        self.checkpoint_dir = checkpoint_dir

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.g_bn_pre_e11 = batch_norm(name='g_bn_pre_e11')
        self.g_bn_pre_e12 = batch_norm(name='g_bn_pre_e12')
        self.g_bn_pre_e21 = batch_norm(name='g_bn_pre_e21')
        self.g_bn_pre_e22 = batch_norm(name='g_bn_pre_e22')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.build_model()

    def generator(self, is_reuse=False):
        with tf.variable_scope("generator") as scope:
            if is_reuse is True:
                scope.reuse_variables()

            # added for multi-images
            # 1. for sketch image
            pre_e11 = self.g_bn_pre_e11(
                conv2d(self.in_sket, output_dim=128, k_h=5, k_w=5, d_h=1, d_w=1, name='g_pre_e11'))
            pre_e12 = self.g_bn_pre_e12(
                conv2d(lrelu(pre_e11), output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, name='g_pre_e12'))
            pre_e13 = conv2d(lrelu(pre_e12), output_dim=32, k_h=3, k_w=3, d_h=1, d_w=1, name='g_pre_e13')
            # 2. for vertex map
            if is_reuse is True:
                pre_e21 = self.g_bn_pre_e21(
                    conv2d(self.vexmaps_recon, output_dim=128, k_h=5, k_w=5, d_h=1, d_w=1, name='g_pre_e21'))
            else:
                pre_e21 = self.g_bn_pre_e21(
                    conv2d(self.vexmaps_init, output_dim=128, k_h=5, k_w=5, d_h=1, d_w=1, name='g_pre_e21'))
            pre_e22 = self.g_bn_pre_e22(
                conv2d(lrelu(pre_e21), output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, name='g_pre_e22'))
            pre_e23 = conv2d(lrelu(pre_e22), output_dim=32, k_h=3, k_w=3, d_h=1, d_w=1, name='g_pre_e23')

            pre_input = tf.concat([pre_e13, pre_e23], -1)

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            e1 = conv2d(pre_input, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim * 8], name='g_d1',
                                                     with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim * 8], name='g_d2',
                                                     with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim * 8], name='g_d3',
                                                     with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim * 8], name='g_d4',
                                                     with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim * 4], name='g_d5',
                                                     with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim * 2], name='g_d6',
                                                     with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim], name='g_d7',
                                                     with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim], name='g_d8',
                                                     with_w=True)

            return tf.nn.tanh(self.d8)

    def save(self, step):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name), global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def build_model(self):
        self.in_sket = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                      name='real_sketch_image')

        self.vdismap_init = self.generator(is_reuse=False)
        # self.vdismap_recon = self.generator(is_reuse=True)
        self.vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        # print(self.vars)
        self.saver = tf.train.Saver(var_list=self.vars)