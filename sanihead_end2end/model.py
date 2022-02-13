from __future__ import division

from utils import *
from cd_dist import *
import p2m.models as p2m
import vex2vdis.model as vex2dis
import meshrefine.models as mesh_refine
import meshrenderer.renderer as meshrenderer

class SAniHeadEnd2End(object):
    def __init__(self, sess, epochs=50, batch_size=1, lr=5e-5, beta1=0.5, data_root=None, checkpoint_dir=None,
                 sample_dir=None, log_dir=None, test_dir=None):

        self.sess = sess
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.data_root = data_root
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        self.test_dir = test_dir
        self.face = np.loadtxt('./utils/sphere/face3.obj', dtype='|S32')

        # faces = np.loadtxt('./utils/sphere/face3.obj', dtype='|S32')
        # faces_idx = np.array(faces[:, 1:], dtype=np.int32)
        # self.faces = tf.constant(faces_idx, dtype=tf.int32, shape=(9920, 3), name='faces')

        ##############################################for model#######################################################
        # pixel2mesh
        self.model_p2m = p2m.SAniHead(sess=self.sess, learning_rate=self.lr)
        self.output_init = self.model_p2m.vert3

        # mesh_render_1
        self.mesh_renderer = meshrenderer.MeshRenderer()
        self.rot_verts_1, self.proj_verts_1, self.vexmaps_1 = self.mesh_renderer.render_mesh_vexmap(self.output_init)

        # vertex_map2vertex_dis_map_1
        self.model_vex2vdis = vex2dis.vex2vdis(sess=self.sess, vexmaps_init=self.vexmaps_1,
                                               batch_size=self.batch_size * 3, lr=self.lr)
        self.vdismaps_1 = self.model_vex2vdis.vdismap_init
        self.output_recon_1, self.output_recon_norm_1 = self.mesh_renderer.recon_mesh_with_vdismap(self.rot_verts_1,
                                                                                                   self.proj_verts_1,
                                                                                                   self.vdismaps_1)
        # mesh_refine
        self.model_refine = mesh_refine.MeshRefine(sess=self.sess, init_pcn=self.output_recon_norm_1,
                                                   learning_rate=self.lr)
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

        ##############################################for loss#######################################################
        self.p2m_loss = self.model_p2m.total_loss

        dist1_1_recon, _, dist2_1_recon, _ = nn_distance(self.model_p2m.gt_pcn[:, :3], self.output_recon_1)
        # self.recon_loss_1 = (tf.reduce_mean(dist1_1_recon) + 0.55 * tf.reduce_mean(dist2_1_recon)) * 3000 * 2.5
        self.recon_loss_1 = (tf.reduce_mean(dist1_1_recon) + 0.55 * tf.reduce_mean(dist2_1_recon)) * 3000 * 2
        dist1_2_recon, _, dist2_2_recon, _ = nn_distance(self.model_p2m.gt_pcn[:, :3], self.output_recon_2)
        # self.recon_loss_2 = (tf.reduce_mean(dist1_2_recon) + 0.55 * tf.reduce_mean(dist2_2_recon)) * 3000 * 2.5
        self.recon_loss_2 = (tf.reduce_mean(dist1_2_recon) + 0.55 * tf.reduce_mean(dist2_2_recon)) * 3000 * 2

        self.refine_loss_1 = self.model_refine.refine_loss(self.output_recon_norm_1, self.output_refine_1)
        self.refine_loss_2 = self.model_refine.refine_loss(self.output_recon_norm_2, self.output_refine_2)

        self.total_loss = self.p2m_loss + self.recon_loss_1 + self.recon_loss_2 + self.refine_loss_1 + self.refine_loss_2

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}
        # print(self.vars)

        self.saver = tf.train.Saver(self.vars)

    def reload(self):
        self.model_p2m.load()
        self.model_vex2vdis.load()
        self.model_refine.load()
    
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

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.total_loss)
        self.sess.run(tf.global_variables_initializer())
        # self.reload()
        self.load()     # 2019-09-23, for demo training(finetune)

        self.sum_p2m_loss = tf.summary.scalar('p2m_loss', self.p2m_loss)
        self.sum_recon_loss = tf.summary.scalar('recon_loss', self.recon_loss_1 + self.recon_loss_2)
        self.sum_refine_loss = tf.summary.scalar('refine_loss', self.refine_loss_1 + self.refine_loss_2)
        self.sum_total_loss = tf.summary.scalar('total_loss', self.total_loss)

        self.sum_train = tf.summary.merge(
            [self.sum_p2m_loss, self.sum_recon_loss, self.sum_refine_loss, self.sum_total_loss])
        self.sum_test = tf.summary.merge(
            [self.sum_p2m_loss, self.sum_recon_loss, self.sum_refine_loss, self.sum_total_loss])

        self.writer_train = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.writer_test = tf.summary.FileWriter(self.log_dir+'/test', self.sess.graph)

        # load file names
        train_name_list = []
        # with open(self.data_root + "train_mesh_list.txt", 'r') as f:
        with open(self.data_root + 'mesh_aug_list_sel.txt', 'r') as f:    # sel for demo
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                train_name_list.append(name)
            f.close()
        with open(self.data_root + 'mesh_gen_inter_sel_aug_list.txt', 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                train_name_list.append(name)
            f.close()
        print("train_data_num:", len(train_name_list))

        test_name_list = []
        # with open(self.data_root + "test_mesh_list.txt", 'r') as f:
        with open(self.data_root + "test_mesh_aug_list_sel.txt", 'r') as f:     # sel for demo
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                test_name_list.append(name)
            f.close()
        print("test_data_num:", len(test_name_list))

        train_num = len(train_name_list)
        train_idx = range(0, train_num)
        for epoch in range(1, self.epochs+1):
            np.random.shuffle(train_idx)
            iter_train = 0
            for i in train_idx:
                tmps = train_name_list[i].split('_')

                sket_batch = []
                if len(tmps) > 4:
                    for j in range(0, 2):
                        sket_batch.append(cv2.imread(os.path.join(self.data_root, 'sketch_mesh_gen_inter_sel_aug',
                                                                  train_name_list[i] + '_' + str(j) + '_sc.png')))
                    sket_batch.append(cv2.flip(sket_batch[1], 1))
                    sket_batch = np.array(sket_batch)
                    gt_pc_with_norm = read_ply(os.path.join(self.data_root, 'pc_mesh_gen_inter_sel_aug',
                                                            train_name_list[i] + '.ply'), with_normal=True)
                elif len(tmps) > 2:
                    rand_val = np.random.randint(1, 11)
                    if rand_val > 5:
                        for j in range(0, 2):
                            sket_batch.append(cv2.imread(
                                os.path.join(self.data_root, 'sketch', train_name_list[i] + '_' + str(j) + '_sc.png')))
                    else:
                        for j in range(0, 2):
                            sket_batch.append(cv2.imread(os.path.join(self.data_root, 'sketch_noise_aug_sel',
                                                                      train_name_list[i] + '_' + str(j) + '_sc.png')))

                    sket_batch.append(cv2.flip(sket_batch[1], 1))
                    sket_batch = np.array(sket_batch)
                    gt_pc_with_norm = read_ply(os.path.join(self.data_root, 'pc', train_name_list[i] + '.ply'),
                                               with_normal=True)

                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                sket_batch = sket_batch/127.5 - 1.

                _, dists, sum_train = self.sess.run([optim, self.total_loss, self.sum_train],
                                                    feed_dict={self.model_p2m.in_sket_front: sket_front,
                                                               self.model_p2m.in_sket_side: sket_side,
                                                               self.model_p2m.gt_pcn: gt_pc_with_norm,
                                                               self.model_vex2vdis.in_sket: sket_batch,
                                                               self.model_refine.gt_pcn: gt_pc_with_norm})

                iter_train += 1
                if iter_train % 10 == 0:
                    self.writer_train.add_summary(sum_train, (epoch-1) * train_num + iter_train)
                    print('Epoch %d, Iteration %d/%d, iter loss = %f' % (epoch, iter_train, train_num, dists))

            # Sample and save model for every 2 epochs
            if epoch % 2 == 0:
                self.sample(test_name_list, epoch)
                self.save(epoch)

        print('Training Finished!')

    def sample(self, test_name_list, epoch_id, test_num=10):
        save_path = os.path.join(self.sample_dir, str(epoch_id))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(0, test_num):
            sket_batch = []
            for j in range(0, 3):
                sket_batch.append(cv2.imread(
                    os.path.join(self.data_root, 'sketch', test_name_list[i] + '_' + str(j) + '_sc.png')))
            sket_batch = np.array(sket_batch)

            # sket_side_flip = cv2.flip(sket_side, 1)
            gt_pc_with_norm = read_ply(os.path.join(self.data_root, 'pc', test_name_list[i] + '.ply'),
                                       with_normal=True)
            # save front sketch
            cv2.imwrite(os.path.join(save_path, test_name_list[i] + '_' + str(0) + '_sc.png'), sket_batch[0])
            sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
            cv2.imwrite(os.path.join(save_path, test_name_list[i] + '_' + str(1) + '_sc.png'), sket_batch[1])
            sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

            sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)

            verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                 self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                           self.model_p2m.gt_pcn: gt_pc_with_norm, self.model_vex2vdis.in_sket: sket_batch,
                           self.model_refine.gt_pcn: gt_pc_with_norm})


            # save vextex maps
            for j in range(0, 3):
                cv2.imwrite(os.path.join(save_path, test_name_list[i] + '_' + str(j) + '_vexmap_1.png'),
                            np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
                cv2.imwrite(os.path.join(save_path, test_name_list[i] + '_' + str(j) + '_vdismap_pre_1.png'),
                            np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
                cv2.imwrite(os.path.join(save_path, test_name_list[i] + '_' + str(j) + '_vexmap_2.png'),
                            np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
                cv2.imwrite(os.path.join(save_path, test_name_list[i] + '_' + str(j) + '_vdismap_pre_2.png'),
                            np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))

            verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
            mesh_init = np.vstack((verts_init, self.face))
            np.savetxt(os.path.join(save_path, test_name_list[i] + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')

            verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
            mesh_recon_1 = np.vstack((verts_recon_1, self.face))
            np.savetxt(os.path.join(save_path, test_name_list[i] + '_recon_1.obj'), mesh_recon_1, fmt='%s',
                       delimiter=' ')

            verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
            mesh_refine_1 = np.vstack((verts_refine_1, self.face))
            np.savetxt(os.path.join(save_path, test_name_list[i] + '_refine_1.obj'), mesh_refine_1, fmt='%s',
                       delimiter=' ')

            verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
            mesh_recon_2 = np.vstack((verts_recon_2, self.face))
            np.savetxt(os.path.join(save_path, test_name_list[i] + '_recon_2.obj'), mesh_recon_2, fmt='%s',
                       delimiter=' ')

            verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
            mesh_refine_2 = np.vstack((verts_refine_2, self.face))
            np.savetxt(os.path.join(save_path, test_name_list[i] + '_refine_2.obj'), mesh_refine_2, fmt='%s',
                       delimiter=' ')


    def test(self):
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.sess.run(tf.global_variables_initializer())
        # print(self.vars)
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            self.reload()
            print(" [!] Load failed...")

        test_num = 30
        # with open(self.data_root+"test_mesh_list.txt", 'r') as f:
        with open(self.data_root + "test_mesh_aug_list_sel.txt", 'r') as f:     # sel for demo
            name_list = f.readlines()
            num = 0
            for name in name_list:
                name = name.strip('\n')
                sket_batch = []
                for j in range(0, 3):
                    sket_batch.append(cv2.imread(
                        os.path.join(self.data_root, 'sketch', name + '_' + str(j) + '_sc.png')))
                sket_batch = np.array(sket_batch)

                # sket_side_flip = cv2.flip(sket_side, 1)
                gt_pc_with_norm = read_ply(os.path.join(self.data_root, 'pc', name + '.ply'),
                                           with_normal=True)
                # save front sketch
                cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(0) + '_sc.png'), sket_batch[0])
                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                # save side sketch
                cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(1) + '_sc.png'), sket_batch[1])
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)

                verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                    [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                     self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                    feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                               self.model_p2m.gt_pcn: gt_pc_with_norm, self.model_vex2vdis.in_sket: sket_batch,
                               self.model_refine.gt_pcn: gt_pc_with_norm})

                # save vextex maps
                for j in range(0, 3):
                    cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vexmap_1.png'),
                                np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vdismap_pre_1.png'),
                                np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vexmap_2.png'),
                                np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(self.test_dir, name + '_' + str(j) + '_vdismap_pre_2.png'),
                                np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))

                verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
                mesh_init = np.vstack((verts_init, self.face))
                np.savetxt(os.path.join(self.test_dir, name + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')

                verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
                mesh_recon_1 = np.vstack((verts_recon_1, self.face))
                np.savetxt(os.path.join(self.test_dir, name + '_recon_1.obj'), mesh_recon_1, fmt='%s',
                           delimiter=' ')

                verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
                mesh_refine_1 = np.vstack((verts_refine_1, self.face))
                np.savetxt(os.path.join(self.test_dir, name + '_refine_1.obj'), mesh_refine_1, fmt='%s',
                           delimiter=' ')

                verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
                mesh_recon_2 = np.vstack((verts_recon_2, self.face))
                np.savetxt(os.path.join(self.test_dir, name + '_recon_2.obj'), mesh_recon_2, fmt='%s',
                           delimiter=' ')

                verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
                mesh_refine_2 = np.vstack((verts_refine_2, self.face))
                np.savetxt(os.path.join(self.test_dir, name + '_refine_2.obj'), mesh_refine_2, fmt='%s',
                           delimiter=' ')

                num += 1
                if num > test_num:
                    break

            print('test done!')


    def test_wo_end2end(self, save_path, test_list_file):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        # print(self.vars)
        self.reload()

        with open(test_list_file, 'r') as f:     # sel for demo
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                sket_batch = []
                for j in range(0, 2):
                    sket_batch.append(cv2.imread(
                        os.path.join(self.data_root, 'sketch', name + '_' + str(j) + '_sc.png')))

                sket_batch.append(cv2.flip(sket_batch[1], 1))
                sket_batch = np.array(sket_batch)

                gt_pc_with_norm = read_ply(os.path.join(self.data_root, 'pc', name + '.ply'),
                                           with_normal=True)
                # save front sketch
                cv2.imwrite(os.path.join(save_path, name + '_' + str(0) + '_sc.png'), sket_batch[0])
                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                cv2.imwrite(os.path.join(save_path, name + '_' + str(1) + '_sc.png'), sket_batch[1])
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)

                verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                    [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                     self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                    feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                               self.model_p2m.gt_pcn: gt_pc_with_norm, self.model_vex2vdis.in_sket: sket_batch,
                               self.model_refine.gt_pcn: gt_pc_with_norm})

                # save vextex maps
                for j in range(0, 3):
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_1.png'),
                                np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_1.png'),
                                np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_2.png'),
                                np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_2.png'),
                                np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))

                verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
                mesh_init = np.vstack((verts_init, self.face))
                np.savetxt(os.path.join(save_path, name + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')

                verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
                mesh_recon_1 = np.vstack((verts_recon_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_1.obj'), mesh_recon_1, fmt='%s',
                           delimiter=' ')

                verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
                mesh_refine_1 = np.vstack((verts_refine_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_1.obj'), mesh_refine_1, fmt='%s',
                           delimiter=' ')

                verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
                mesh_recon_2 = np.vstack((verts_recon_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_2.obj'), mesh_recon_2, fmt='%s',
                           delimiter=' ')

                verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
                mesh_refine_2 = np.vstack((verts_refine_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_2.obj'), mesh_refine_2, fmt='%s',
                           delimiter=' ')

            print('test done!')


    def predict(self, in_sket_root, file_list, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        with open(file_list, 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                sket_batch = []
                for j in range(0, 3):
                    sket_batch.append(cv2.imread(
                        os.path.join(in_sket_root, name + '_' + str(j) + '_sc.png')))
                sket_batch = np.array(sket_batch)

                # save front sketch
                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                rand_val = np.random.randint(1, 11)
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)

                verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                    [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                     self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                    feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                               self.model_vex2vdis.in_sket: sket_batch})

                # save vextex maps
                for j in range(0, 3):
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_1.png'),
                                np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_1.png'),
                                np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_2.png'),
                                np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_2.png'),
                                np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))

                verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
                mesh_init = np.vstack((verts_init, self.face))
                np.savetxt(os.path.join(save_path, name + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')

                verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
                mesh_recon_1 = np.vstack((verts_recon_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_1.obj'), mesh_recon_1, fmt='%s',
                           delimiter=' ')

                verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
                mesh_refine_1 = np.vstack((verts_refine_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_1.obj'), mesh_refine_1, fmt='%s',
                           delimiter=' ')

                verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
                mesh_recon_2 = np.vstack((verts_recon_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_2.obj'), mesh_recon_2, fmt='%s',
                           delimiter=' ')

                verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
                mesh_refine_2 = np.vstack((verts_refine_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_2.obj'), mesh_refine_2, fmt='%s',
                           delimiter=' ')
            print('predict done!')


    def generate(self, in_sket_root, file_list, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        with open(file_list, 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                sket_batch = []
                for j in range(0, 3):
                    sket_batch.append(cv2.imread(
                        os.path.join(in_sket_root, name + '_' + str(j) + '_sc.png')))
                sket_batch = np.array(sket_batch)

                # save front sketch
                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)

                verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                    [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                     self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                    feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                               self.model_vex2vdis.in_sket: sket_batch})

                verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
                mesh_refine_2 = np.vstack((verts_refine_2, self.face))
                np.savetxt(os.path.join(save_path, name + '.obj'), mesh_refine_2, fmt='%s', delimiter=' ')

            print('predict done!')


    def predict_p2m_middle_result(self, in_sket_root, file_list, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        face1 = np.loadtxt('./utils/sphere/face1.obj', dtype='|S32')
        face2 = np.loadtxt('./utils/sphere/face2.obj', dtype='|S32')

        with open(file_list, 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                print('processing', name, '...')

                sket_batch = []
                sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '_0_sc.png')))
                sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '_1_sc.png')))
                # sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '2_sc.png')))
                sket_batch.append(cv2.flip(sket_batch[1], 1))

                sket_batch = np.array(sket_batch)

                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                verts_1, verts_2 = self.sess.run([self.model_p2m.vert1, self.model_p2m.vert2],
                                                 feed_dict={self.model_p2m.in_sket_front: sket_front,
                                                            self.model_p2m.in_sket_side: sket_side})

                verts_1 = np.hstack((np.full([verts_1.shape[0], 1], 'v'), verts_1))
                mesh_1 = np.vstack((verts_1, face1))
                np.savetxt(os.path.join(save_path, name + '_block1.obj'), mesh_1, fmt='%s', delimiter=' ')

                verts_2 = np.hstack((np.full([verts_2.shape[0], 1], 'v'), verts_2))
                mesh_2 = np.vstack((verts_2, face2))
                np.savetxt(os.path.join(save_path, name + '_block2.obj'), mesh_2, fmt='%s', delimiter=' ')

            f.close()
            print('predict done!')


    def predict_middle_result(self, in_sket_root, file_list, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        with open(file_list, 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                print('processing', name, '...')

                sket_batch = []
                sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '_0_sc.png')))
                sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '_1_sc.png')))
                # sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '2_sc.png')))
                sket_batch.append(cv2.flip(sket_batch[1], 1))

                sket_batch = np.array(sket_batch)

                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                    [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                     self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                    feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                               self.model_vex2vdis.in_sket: sket_batch})

                # save vextex maps
                for j in range(0, 3):
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_1.png'),
                                np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_1.png'),
                                np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_2.png'),
                                np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_2.png'),
                                np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))

                verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
                mesh_init = np.vstack((verts_init, self.face))
                np.savetxt(os.path.join(save_path, name + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')

                verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
                mesh_recon_1 = np.vstack((verts_recon_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_1.obj'), mesh_recon_1, fmt='%s',
                           delimiter=' ')

                verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
                mesh_refine_1 = np.vstack((verts_refine_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_1.obj'), mesh_refine_1, fmt='%s',
                           delimiter=' ')

                verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
                mesh_recon_2 = np.vstack((verts_recon_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_2.obj'), mesh_recon_2, fmt='%s',
                           delimiter=' ')

                verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
                mesh_refine_2 = np.vstack((verts_refine_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_2.obj'), mesh_refine_2, fmt='%s',
                           delimiter=' ')

            f.close()
            print('predict done!')


    def predict_middle_result_without_end2end(self, in_sket_root, file_list, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        if self.reload():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        with open(file_list, 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                print('processing', name, '...')

                sket_batch = []
                sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '_0_sc.png')))
                sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '_1_sc.png')))
                # sket_batch.append(cv2.imread(os.path.join(in_sket_root, name + '2_sc.png')))
                sket_batch.append(cv2.flip(sket_batch[1], 1))

                sket_batch = np.array(sket_batch)

                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.

                verts_init, vexmaps_1, vdismap_pre_1, verts_recon_1, verts_refine_1, vexmaps_2, vdismap_pre_2, verts_recon_2, verts_refine_2 = self.sess.run(
                    [self.output_init, self.vexmaps_1, self.vdismaps_1, self.output_recon_1, self.output_refine_1,
                     self.vexmaps_2, self.vdismaps_2, self.output_recon_2, self.output_refine_2],
                    feed_dict={self.model_p2m.in_sket_front: sket_front, self.model_p2m.in_sket_side: sket_side,
                               self.model_vex2vdis.in_sket: sket_batch})

                # save vextex maps
                for j in range(0, 3):
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_1.png'),
                                np.array((vexmaps_1[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_1.png'),
                                np.array((vdismap_pre_1[j] + 1) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vexmap_2.png'),
                                np.array((vexmaps_2[j] + 1.) * 127.5, dtype=np.uint8))
                    cv2.imwrite(os.path.join(save_path, name + '_' + str(j) + '_vdismap_pre_2.png'),
                                np.array((vdismap_pre_2[j] + 1) * 127.5, dtype=np.uint8))

                verts_init = np.hstack((np.full([verts_init.shape[0], 1], 'v'), verts_init))
                mesh_init = np.vstack((verts_init, self.face))
                np.savetxt(os.path.join(save_path, name + '_init.obj'), mesh_init, fmt='%s', delimiter=' ')

                verts_recon_1 = np.hstack((np.full([verts_recon_1.shape[0], 1], 'v'), verts_recon_1))
                mesh_recon_1 = np.vstack((verts_recon_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_1.obj'), mesh_recon_1, fmt='%s',
                           delimiter=' ')

                verts_refine_1 = np.hstack((np.full([verts_refine_1.shape[0], 1], 'v'), verts_refine_1))
                mesh_refine_1 = np.vstack((verts_refine_1, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_1.obj'), mesh_refine_1, fmt='%s',
                           delimiter=' ')

                verts_recon_2 = np.hstack((np.full([verts_recon_2.shape[0], 1], 'v'), verts_recon_2))
                mesh_recon_2 = np.vstack((verts_recon_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_recon_2.obj'), mesh_recon_2, fmt='%s',
                           delimiter=' ')

                verts_refine_2 = np.hstack((np.full([verts_refine_2.shape[0], 1], 'v'), verts_refine_2))
                mesh_refine_2 = np.vstack((verts_refine_2, self.face))
                np.savetxt(os.path.join(save_path, name + '_refine_2.obj'), mesh_refine_2, fmt='%s',
                           delimiter=' ')

            f.close()
            print('predict done!')


    def predict_recon_result_with_normal(self, in_sket_root, file_list, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        with open(file_list, 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip('\n')
                print('processing', name, '...')
                sket_batch = []
                for j in range(0, 3):
                    sket_batch.append(cv2.imread(
                        os.path.join(in_sket_root, name + '_' + str(j) + '_sc.png')))
                sket_batch = np.array(sket_batch)

                # save front sketch
                sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
                sket_batch = np.array(sket_batch / 127.5 - 1.).reshape(3, 256, 256, 3)

                verts_recon_1, verts_recon_2 = self.sess.run([self.output_recon_norm_1, self.output_recon_norm_2],
                                                             feed_dict={self.model_p2m.in_sket_front: sket_front,
                                                                        self.model_p2m.in_sket_side: sket_side,
                                                                        self.model_vex2vdis.in_sket: sket_batch})
                save_ply(verts_recon_1, os.path.join(save_path, name + '_recon_1.ply'), with_normal=True)
                save_ply(verts_recon_2, os.path.join(save_path, name + '_recon_2.ply'), with_normal=True)
            print('predict done!')







