from __future__ import division

import os
from utils import *
import tensorflow as tf
import p2m.models as p2m
import vex2vdis.model as vex2dis
import meshrefine.models as mesh_refine
import meshrenderer.renderer as meshrenderer

class SAniHeadEnd2End(object):
    def __init__(self, sess, face_tempalte="./utils/sphere/face3.obj", checkpoint_dir="./utils/checkpoint"):

        self.sess = sess
        self.face = np.loadtxt(face_tempalte, dtype='|S32')
        self.checkpoint_dir = checkpoint_dir

        ##############################################for model#######################################################
        # pixel2mesh
        self.model_p2m = p2m.SAniHead(sess=self.sess)
        self.output_init = self.model_p2m.vert3

        # mesh_render_1
        self.mesh_renderer = meshrenderer.MeshRenderer()
        self.rot_verts_1, self.proj_verts_1, self.vexmaps_1 = self.mesh_renderer.render_mesh_vexmap(self.output_init)

        # vertex_map2vertex_dis_map_1
        self.model_vex2vdis = vex2dis.vex2vdis(sess=self.sess, vexmaps_init=self.vexmaps_1, batch_size=3)
        self.vdismaps_1 = self.model_vex2vdis.vdismap_init
        self.output_recon_1, self.output_recon_norm_1 = self.mesh_renderer.recon_mesh_with_vdismap(self.rot_verts_1,
                                                                                                   self.proj_verts_1,
                                                                                                   self.vdismaps_1)
        # mesh_refine
        self.model_refine = mesh_refine.MeshRefine(sess=self.sess, init_pcn=self.output_recon_norm_1)
        self.output_refine_1 = self.model_refine.output

        # mesh_render_2
        self.rot_verts_2, self.proj_verts_2, self.vexmaps_2 = self.mesh_renderer.render_mesh_vexmap(self.output_refine_1)
        self.model_vex2vdis.vexmaps_recon = self.vexmaps_2
        self.vdismaps_2 = self.model_vex2vdis.generator(is_reuse=True)
        self.output_recon_2, self.output_recon_norm_2 = self.mesh_renderer.recon_mesh_with_vdismap(self.rot_verts_2,
                                                                                                   self.proj_verts_2,
                                                                                                   self.vdismaps_2)
        # mesh_refine_iter
        self.output_refine_2 = self.model_refine.predict(self.output_recon_norm_2)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}
        # print(self.vars)

        self.saver = tf.train.Saver(self.vars)

    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save(self, epoch):
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(self.sess, os.path.join(self.checkpoint_dir, 'sanihead_'+str(epoch)+'.ckpt'))
        print("Model saved in file: %s" % save_path)

    # def test(self):
    #     if not os.path.exists(self.test_dir):
    #         os.mkdir(self.test_dir)
    #
    #     self.sess.run(tf.global_variables_initializer())
    #     # print(self.vars)
    #     if self.load():
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")
    #
    #     test_num = 30
    #     with open(self.data_root+"test_mesh_list.txt", 'r') as f:
    #         name_list = f.readlines()
    #         num = 0
    #         for name in name_list:
    #             name = name.strip('\n')
    #             sket_batch = []
    #             for j in range(0, 3):
    #                 sket_batch.append(cv2.imread(
    #                     os.path.join(self.data_root, 'sketch', name + '_' + str(j) + '_sc.png')))
    #             sket_batch = np.array(sket_batch)
    #
    #             # sket_side_flip = cv2.flip(sket_side, 1)
    #             gt_pc_with_norm = read_ply(os.path.join(self.data_root, 'pc', name + '.ply'),
    #                                        with_normal=True)
    #             # save front sketch
    #             cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(0) + '_sc.png'), sket_batch[0])
    #             sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
    #             rand_val = np.random.randint(1, 11)
    #             if rand_val <= 5:
    #                 # save side sketch
    #                 cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(1) + '_sc.png'), sket_batch[1])
    #                 sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
    #             else:
    #                 # save side sketch
    #                 cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(2) + '_sc.png'), sket_batch[2])
    #                 sket_side = cv2.resize(sket_batch[2], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
    #
    #             sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)
    #
    #             verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
    #                 [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
    #                  self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
    #                 feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
    #                            self.model_p2m.gt_pcn: gt_pc_with_norm, self.model_vex2vdis.in_sket: sket_batch,
    #                            self.model_refine.gt_pcn: gt_pc_with_norm})
    #
    #             # save vextex maps
    #             for j in range(0, 3):
    #                 cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vexmap_1.png'),
    #                             np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
    #                 cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vdismap_pre_1.png'),
    #                             np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
    #                 cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vexmap_2.png'),
    #                             np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
    #                 cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vdismap_pre_2.png'),
    #                             np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))
    #
    #             verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
    #             mesh_init = np.vstack((verts_init, self.face))
    #             np.savetxt(os.path.join(self.test_dir, name + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')
    #
    #             verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
    #             mesh_recon_1 = np.vstack((verts_recon_1, self.face))
    #             np.savetxt(os.path.join(self.test_dir, name + '_recon_1.obj'), mesh_recon_1, fmt='%s',
    #                        delimiter=' ')
    #
    #             verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
    #             mesh_refine_1 = np.vstack((verts_refine_1, self.face))
    #             np.savetxt(os.path.join(self.test_dir, name + '_refine_1.obj'), mesh_refine_1, fmt='%s',
    #                        delimiter=' ')
    #
    #             verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
    #             mesh_recon_2 = np.vstack((verts_recon_2, self.face))
    #             np.savetxt(os.path.join(self.test_dir, name + '_recon_2.obj'), mesh_recon_2, fmt='%s',
    #                        delimiter=' ')
    #
    #             verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
    #             mesh_refine_2 = np.vstack((verts_refine_2, self.face))
    #             np.savetxt(os.path.join(self.test_dir, name + '_refine_2.obj'), mesh_refine_2, fmt='%s',
    #                        delimiter=' ')
    #
    #             num += 1
    #             if num > test_num:
    #                 break
    #
    #         print('test done!')











