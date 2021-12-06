import tflearn
from p2m.layers import *
import pickle
import os

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.name = 'sanihead'
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

        self.layers_p2m = []
        self.activations_p2m = []

        self.sess = None
        self.inputs = None  # input template vertex coordinates
        self.vert1 = None
        self.vert2 = None
        self.vert3 = None
        self.vert1_2 = None
        self.vert2_2 = None

        self.model_path = os.path.join('./utils/', 'checkpoint_dual')

        # for model define
        self.img_feat = None  # input sketch images' feature
        self.support1 = None  # graph structure in the first block
        self.support2 = None  # graph structure in the second block
        self.support3 = None  # graph structure in the third block
        self.edges = None  # helper for normal loss
        self.lape_idx = None  # helper for laplacian regularization
        self.pool_idx = None  # helper for graph unpooling

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        # with tf.device('/gpu:0'):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model
        eltwise_p2m = [3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 45]
        concat_p2m = [15, 31]
        self.activations_p2m.append(self.inputs)
        for idx, layer in enumerate(self.layers_p2m):
            hidden = layer(self.activations_p2m[-1])
            if idx in eltwise_p2m:
                hidden = tf.add(hidden, self.activations_p2m[-2]) * 0.5
            if idx in concat_p2m:
                hidden = tf.concat([hidden, self.activations_p2m[-2]], 1)
            self.activations_p2m.append(hidden)

        self.vert1 = self.activations_p2m[15]
        unpool_layer = GraphPooling(pool_idx=self.pool_idx, pool_id=1)
        self.vert1_2 = unpool_layer(self.vert1)

        self.vert2 = self.activations_p2m[31]
        unpool_layer = GraphPooling(pool_idx=self.pool_idx, pool_id=2)
        self.vert2_2 = unpool_layer(self.vert2)

        self.vert3 = self.activations_p2m[48]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def predict(self):
        pass

    def load(self):
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        # save_path = "utils/checkpoint/%s.ckpt" % self.name
        save_path = os.path.join(self.model_path, self.name + '.ckpt')
        saver.restore(self.sess, save_path)
        print("Pixel2Mesh model restored from file: %s" % save_path)

class SAniHead(Model):
    def __init__(self, sess=None, init_infor_file='utils/sphere/init_info.dat', img_size=224, hidden=192, feat_dim=1923,
                 coord_dim=3, **kwargs):
        super(SAniHead, self).__init__(**kwargs)

        self.sess = sess
        self.init_infor_file = init_infor_file
        self.img_size = img_size
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.coord_dim = coord_dim

        # Define placeholders(dict) and model
        self.in_sket_front = tf.placeholder(tf.float32, shape=(self.img_size, self.img_size, 3),
                                            name='in_sket_front')  # input front sketch image
        self.in_sket_side = tf.placeholder(tf.float32, shape=(self.img_size, self.img_size, 3),
                                           name='in_sket_side')  # input side sketch image

        self._init_placeholders()

        self.build()

    def _init_placeholders(self):
        pkl = pickle.load(open(self.init_infor_file, 'rb'), encoding='latin1')

        coord = pkl[0]
        pool_idx = pkl[4]
        edges = []
        for i in range(1, 4):
            adj = pkl[i][1]
            edges.append(adj[0])

        num_supports = 2
        self.inputs = tf.constant(coord, dtype=tf.float32, shape=(312, 3))  # input template vertex coordinates
        self.edges = [tf.constant(edges[i], dtype=tf.int32, name='edge' + str(i)) for i in range(len(edges))]
        self.pool_idx = [tf.constant(pool_idx[i], dtype=tf.int32, name='pool_idx' + str(i)) for i in
                         range(len(pool_idx))]
        self.support1 = [tf.sparse.SparseTensor(pkl[1][i][0], tf.cast(pkl[1][i][1], tf.float32), pkl[1][i][2]) for i in
                         range(num_supports)]
        self.support2 = [tf.sparse.SparseTensor(pkl[2][i][0], tf.cast(pkl[2][i][1], tf.float32), pkl[2][i][2]) for i in
                         range(num_supports)]
        self.support3 = [tf.sparse.SparseTensor(pkl[3][i][0], tf.cast(pkl[3][i][1], tf.float32), pkl[3][i][2]) for i in
                         range(num_supports)]

    def _build(self):
        self.build_cnn18()

        # first project block
        self.layers_p2m.append(GraphProjection2(img_feat=self.img_feat, name='pixel2mesh'))
        self.layers_p2m.append(GraphConvolution(input_dim=self.feat_dim,
                                                output_dim=self.hidden,
                                                # gcn_block_id=1,
                                                support=self.support1,
                                                logging=self.logging,
                                                name='pixel2mesh'))
        for _ in range(12):
            self.layers_p2m.append(GraphConvolution(input_dim=self.hidden,
                                                    output_dim=self.hidden,
                                                    # gcn_block_id=1,
                                                    support=self.support1,
                                                    logging=self.logging,
                                                    name='pixel2mesh'))
        self.layers_p2m.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=self.coord_dim,
                                                act=lambda x: x,
                                                # gcn_block_id=1,
                                                support=self.support1,
                                                logging=self.logging,
                                                name='pixel2mesh'))

        # second project block
        self.layers_p2m.append(GraphProjection2(img_feat=self.img_feat, name='pixel2mesh'))
        self.layers_p2m.append(GraphPooling(pool_idx=self.pool_idx, pool_id=1, name='pixel2mesh'))
        self.layers_p2m.append(GraphConvolution(input_dim=self.feat_dim + self.hidden,
                                                output_dim=self.hidden,
                                                # gcn_block_id=2,
                                                support=self.support2,
                                                logging=self.logging,
                                                name='pixel2mesh'))
        for _ in range(12):
            self.layers_p2m.append(GraphConvolution(input_dim=self.hidden,
                                                    output_dim=self.hidden,
                                                    # gcn_block_id=2,
                                                    support=self.support2,
                                                    logging=self.logging,
                                                    name='pixel2mesh'))
        self.layers_p2m.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=self.coord_dim,
                                                act=lambda x: x,
                                                # gcn_block_id=2,
                                                support=self.support2,
                                                logging=self.logging,
                                                name='pixel2mesh'))

        # third project block
        self.layers_p2m.append(GraphProjection2(img_feat=self.img_feat, name='pixel2mesh'))
        self.layers_p2m.append(GraphPooling(pool_idx=self.pool_idx, pool_id=2, name='pixel2mesh'))
        self.layers_p2m.append(GraphConvolution(input_dim=self.feat_dim + self.hidden,
                                                output_dim=self.hidden,
                                                # gcn_block_id=3,
                                                support=self.support3,
                                                logging=self.logging,
                                                name='pixel2mesh'))
        for _ in range(13):
            self.layers_p2m.append(GraphConvolution(input_dim=self.hidden,
                                                    output_dim=self.hidden,
                                                    # gcn_block_id=3,
                                                    support=self.support3,
                                                    logging=self.logging,
                                                    name='pixel2mesh'))
        self.layers_p2m.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=self.coord_dim,
                                                act=lambda x: x,
                                                # gcn_block_id=3,
                                                support=self.support3,
                                                logging=self.logging,
                                                name='pixel2mesh'))

    def build_cnn18(self):
        x = tf.expand_dims(self.in_sket_front, 0)
        y = tf.expand_dims(self.in_sket_side, 0)
        x = tf.concat([x, y], axis=0)

        # 224 224
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x0 = x
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 112 112
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x1 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 56 56
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 28 28
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 14 14
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        # 7 7
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5,
                                        regularizer='L2')
        x5 = x

        # updata image feature
        self.img_feat = [tf.concat([x2[0], x2[1]], axis=-1), tf.concat([x3[0], x3[1]], axis=-1),
                         tf.concat([x4[0], x4[1]], axis=-1), tf.concat([x5[0], x5[1]], axis=-1)]
