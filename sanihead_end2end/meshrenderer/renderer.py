from __future__ import division

import numpy as np
import tensorflow as tf
import dirt.rasterise_ops
import dirt.matrices
import dirt.lighting

class MeshRenderer(object):
    def __init__(self, img_w=256, img_h=256, n_vert=4962, n_view=3):
        self.name = 'meshrenderer'

        self.img_w = img_w
        self.img_h = img_h
        self.n_vert = n_vert
        self.n_view = n_view

        self.min_p = tf.convert_to_tensor([-0.9924, -1.0272, -1.1439])
        self.max_p = tf.convert_to_tensor([0.9993, 1.4394, 1.4995])
        self.p_scale = 1. / (self.max_p - self.min_p)
        self.max_vdis_val_gf = 0.09

        faces = np.loadtxt('./utils/sphere/face3.obj', dtype='|S32')
        faces_idx = np.array(faces[:, 1:], dtype=np.int32) - 1
        self.faces = tf.constant(faces_idx, dtype=tf.int32, shape=(9920, 3), name='faces')

        self.rot_mat_1 = tf.convert_to_tensor([[0., 0., -1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
        self.rot_mat_2 = tf.convert_to_tensor([[0., 0., 1., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.]])
        self.trans_mat = tf.constant([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., -5., 1.]],
                                     dtype=tf.float32)
        self.proj_mat = dirt.matrices.orthogonal_projection(0.1, 20.)

        self.de_rot_mat_1 = tf.convert_to_tensor(
            [[0., 0., 1., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.]])
        self.de_rot_mat_2 = tf.convert_to_tensor(
            [[0., 0., -1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])

    ### for mesh renderer
    def render_mesh_vexmap(self, verts_output):
        # information for vertex maps rendering
        verts = tf.concat([verts_output, tf.ones([self.n_vert, 1], dtype=tf.float32)], 1)
        rot_verts = tf.stack([verts, tf.matmul(verts, self.rot_mat_1), tf.matmul(verts, self.rot_mat_2)], axis=0)
        proj_verts = tf.stack([tf.matmul(tf.matmul(rot_verts[0], self.trans_mat), self.proj_mat),
                               tf.matmul(tf.matmul(rot_verts[1], self.trans_mat), self.proj_mat),
                               tf.matmul(tf.matmul(rot_verts[2], self.trans_mat), self.proj_mat)],
                              axis=0)

        # the images color channel rendered by dirt is RGB, but the vex2vdis is BGR
        vexmap_0 = dirt.rasterise_ops.rasterise(background=tf.ones([self.img_h, self.img_w, 3]),
                                                vertices=proj_verts[0],
                                                vertex_colors=tf.multiply(rot_verts[0, :, :3] - self.min_p,
                                                                          self.p_scale),
                                                faces=self.faces,
                                                height=self.img_h,
                                                width=self.img_w,
                                                channels=3)
        vexmap_1 = dirt.rasterise_ops.rasterise(background=tf.ones([self.img_h, self.img_w, 3]),
                                                vertices=proj_verts[1],
                                                vertex_colors=tf.multiply(rot_verts[1, :, :3] - self.min_p,
                                                                          self.p_scale),
                                                faces=self.faces,
                                                height=self.img_h,
                                                width=self.img_w,
                                                channels=3)
        vexmap_2 = dirt.rasterise_ops.rasterise(background=tf.ones([self.img_h, self.img_w, 3]),
                                                vertices=proj_verts[2],
                                                vertex_colors=tf.multiply(rot_verts[2, :, :3] - self.min_p,
                                                                          self.p_scale),
                                                faces=self.faces,
                                                height=self.img_h,
                                                width=self.img_w,
                                                channels=3)

        # self.vexmaps = tf.stack([vermap_0[:, :, ::-1]*2.-1., vermap_1[:, :, ::-1]*2.-1., vermap_2[:, :, ::-1]*2.-1.],
        #                         axis=0)

        # note: the values of initial vexmap rendered are belong to [0, 1], and the channel distribution is RGB
        vexmaps = tf.stack([vexmap_0[:, :, ::-1], vexmap_1[:, :, ::-1], vexmap_2[:, :, ::-1]], axis=0)
        vexmaps = vexmaps * 2. - 1.

        return rot_verts, proj_verts, vexmaps

    def recon_mesh_with_vdismap(self, rot_verts, proj_verts, vdismaps):  # for 3 views (vdismaps are tensorflow tensors)
        # information for vertex displacement maps based reconstruction
        normals_0 = dirt.lighting.vertex_normals(rot_verts[0, :, :3], self.faces)
        normals_1 = dirt.lighting.vertex_normals(rot_verts[1, :, :3], self.faces)
        normals_2 = dirt.lighting.vertex_normals(rot_verts[2, :, :3], self.faces)

        view_dir = tf.convert_to_tensor([[0.], [0.], [-1.]], name='view_direction')
        norms_dot_dirs = tf.stack([tf.reduce_sum(tf.matmul(normals_0, view_dir), axis=1),
                                   tf.reduce_sum(tf.matmul(normals_1, view_dir), axis=1),
                                   tf.reduce_sum(tf.matmul(normals_2, view_dir), axis=1)],
                                  axis=0)

        rot_verts_vdis = []
        sel_idx_vec = []
        for i in range(0, self.n_view):
            sel_idx_init = tf.squeeze(tf.where(norms_dot_dirs[i] < -0.11))
            sel_vts_init = tf.gather(proj_verts[i], sel_idx_init)
            sel_x = tf.cast((sel_vts_init[:, 0] + 1.) / 2. * self.img_w + 0.5, dtype=tf.int32)
            sel_y = tf.cast(self.img_h - (sel_vts_init[:, 1] + 1.) / 2. * self.img_h + 0.5, dtype=tf.int32)
            sel_vdis_init = tf.gather_nd(vdismaps[i], tf.stack([sel_y, sel_x], axis=1))
            sel_num_init = tf.shape(sel_vdis_init)[0]
            # sel_num_init = len(sess.run(sel_vdis_init))
            sel_idx_mid = tf.squeeze(tf.where(
                tf.reduce_sum(tf.square(sel_vdis_init - tf.constant([1., 1., 1.], dtype=tf.float32)),
                              axis=-1) > 1e-4))

            sel_vdis = tf.gather(sel_vdis_init, sel_idx_mid) * self.max_vdis_val_gf
            sel_idx = tf.squeeze(tf.where(tf.sparse_to_dense(sel_idx_init, [self.n_vert],
                                                             tf.sparse_to_dense(sel_idx_mid, [sel_num_init], 1,
                                                                                default_value=0),
                                                             default_value=0) > 0))

            sel_vdis_x = tf.reshape(tf.sparse_to_dense(sel_idx, [self.n_vert], sel_vdis[:, 2], default_value=0.),
                                    (self.n_vert, 1))
            sel_vdis_y = tf.reshape(tf.sparse_to_dense(sel_idx, [self.n_vert], sel_vdis[:, 1], default_value=0.),
                                    (self.n_vert, 1))
            sel_vdis_z = tf.reshape(tf.sparse_to_dense(sel_idx, [self.n_vert], sel_vdis[:, 0], default_value=0.),
                                    (self.n_vert, 1))
            sel_vdis = tf.concat([sel_vdis_x, sel_vdis_y, sel_vdis_z, tf.ones([self.n_vert, 1], dtype=tf.float32)],
                                 1)

            sel_idx_vec.append(tf.sparse_to_dense(sel_idx, [self.n_vert], 1, default_value=0))
            rot_verts_vdis.append(sel_vdis)

        sel_idx_sum = tf.squeeze(tf.add_n(sel_idx_vec))
        sel_idx_modify = tf.squeeze(tf.where(sel_idx_sum > tf.constant(0, dtype=tf.int32)))
        sel_wei_modify = tf.cast(1., dtype=tf.float32) / tf.cast(tf.gather(sel_idx_sum, sel_idx_modify),
                                                                 dtype=tf.float32)
        sel_wei_modify = tf.reshape(
            tf.sparse_to_dense(sel_idx_modify, [self.n_vert], sel_wei_modify, default_value=0.),
            (self.n_vert, 1))
        sel_wei_modify = tf.concat([sel_wei_modify, sel_wei_modify, sel_wei_modify], 1)

        de_rot_verts_vdis = []
        de_rot_verts_vdis.append(rot_verts_vdis[0])
        de_rot_verts_vdis.append(tf.matmul(rot_verts_vdis[1], self.de_rot_mat_1))
        de_rot_verts_vdis.append(tf.matmul(rot_verts_vdis[2], self.de_rot_mat_2))

        verts_vdis = tf.multiply(
            de_rot_verts_vdis[0][:, :3] + de_rot_verts_vdis[1][:, :3] + de_rot_verts_vdis[2][:, :3],
            sel_wei_modify)
        verts_recon = rot_verts[0][:, :3] + verts_vdis
        verts_norm = dirt.lighting.vertex_normals(verts_recon, self.faces)
        verts_recon_norms = tf.concat([verts_recon, verts_norm], axis=-1)

        return verts_recon, verts_recon_norms