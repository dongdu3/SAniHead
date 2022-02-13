from layers import *
from losses import *
from fetcher import *
import pickle

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.name = 'meshrefine'
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}
        self.feed_dict = dict()

        self.layers_det = []
        self.activations_det = []

        self.sess = None
        self.input = None
        self.output = None

        self.total_loss = 0
        self.edge_loss = 0
        self.normal_loss = 0
        self.cd_loss = 0
        self.laplace_loss = 0
        self.move_loss = 0

        self.optimizer = None
        self.opt_op = None

        self.model_path = './meshrefine/checkpoint'

        # for model define
        self.support = None  # graph structure in the third block
        self.edges = None  # helper for normal loss
        self.lape_idx = None  # helper for laplacian regularization

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        # with tf.device('/gpu:0'):
        with tf.variable_scope(self.name) as scope:
            self._build()

        # Build DetailNet
        eltwise_det = [2, 4, 6, 8, 10]
        self.activations_det.append(self.input)
        for idx, layer in enumerate(self.layers_det):
            hidden = layer(self.activations_det[-1])
            if idx in eltwise_det:
                hidden = tf.add(hidden, self.activations_det[-2]) * 0.5
            self.activations_det.append(hidden)

        self.output = self.activations_det[-1]

        # Store model variables for easy access
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        # self.vars = {var.name: var for var in variables}
        self.vars = [var for var in tf.trainable_variables() if 'detailnet' in var.name]
        # print(self.vars)

        # Build metrics
        self._loss()

    def predict(self, in_pcn):
        # Build DetailNet
        activations_det = []
        eltwise_det = [2, 4, 6, 8, 10]
        activations_det.append(in_pcn)
        for idx, layer in enumerate(self.layers_det):
            hidden = layer(activations_det[-1])
            if idx in eltwise_det:
                hidden = tf.add(hidden, activations_det[-2]) * 0.5
            activations_det.append(hidden)

        return activations_det[-1]

    def _loss(self):
        raise NotImplementedError

    def refine_loss(self, verts_in, verts_out):
        raise NotImplementedError

    def save(self):
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(self.sess, os.path.join(self.model_path, self.name + '.ckpt'))
        print("Model saved in file: %s" % save_path)

    def load(self):
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        # save_path = "utils/checkpoint/%s.ckpt" % self.name
        save_path = os.path.join(self.model_path, self.name + '.ckpt')
        saver.restore(self.sess, save_path)
        print("Model restored from file: %s" % save_path)


class MeshRefine(Model):
    def __init__(self, sess=None, init_pcn=None, init_infor_file='utils/sphere/init_info.dat',
                 epochs=40, hidden=192, input_dim=6, learning_rate=1e-4, weight_decay=1e-5, in_root=None, gt_root=None,
                 train_list=None, test_list=None, **kwargs):
        super(MeshRefine, self).__init__(**kwargs)

        self.sess = sess
        self.input = init_pcn
        self.init_infor_file = init_infor_file
        self.epochs = epochs
        self.hidden = hidden
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.in_root = in_root
        self.gt_root = gt_root
        self.train_list = train_list
        self.test_list = test_list

        # Define placeholders(dict) and model
        # self.input = tf.placeholder(tf.float32, shape=(4962, 6), name='in_pcn')     # initial pc results with normals
        self.gt_pcn = tf.placeholder(tf.float32, shape=(None, 6), name='gt_pcn')  # ground truth pc with normals

        self._init_placeholders()
        self.build()

    def _read_template_faces(self, mesh_path='utils/sphere/face3.obj'):
        m_reader = open(mesh_path)
        m_infors = m_reader.readlines()
        m_reader.close()
        faces = []
        for line in m_infors:
            line = line.strip('\n')
            infor = line.split(' ')
            if infor[0] == 'f':
                f = [int(infor[1]) - 1, int(infor[2]) - 1, int(infor[3]) - 1]
                faces.append(f)
        faces = np.array(faces)
        return faces

    def _init_placeholders(self):
        pkl = pickle.load(open(self.init_infor_file, 'rb'))

        num_supports = 2
        lape_idx = pkl[5]
        edges = []
        for i in range(1, 4):
            adj = pkl[i][1]
            edges.append(adj[0])

        self.edges = tf.constant(edges[2], dtype=tf.int32, name='edges')
        self.lape_idx = tf.constant(lape_idx[2], dtype=tf.int32, name='lape_idx')
        self.support = [tf.sparse.SparseTensor(pkl[3][i][0], tf.cast(pkl[3][i][1], tf.float32), pkl[3][i][2]) for i in
                        range(num_supports)]

    def _build(self):
        # for geometric detail enhancement block
        self.layers_det.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden,
                                                support=self.support,
                                                logging=self.logging,
                                                name='detailnet'))
        for _ in range(11):
            self.layers_det.append(GraphConvolution(input_dim=self.hidden,
                                                    output_dim=self.hidden,
                                                    support=self.support,
                                                    logging=self.logging,
                                                    name='detailnet'))
        self.layers_det.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=3,
                                                act=lambda x: x,
                                                support=self.support,
                                                logging=self.logging,
                                                name='detailnet'))

    def _loss(self):
        '''
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
        '''
        # self.point_loss = tf.nn.l2_loss(self.output-self.placeholders['gt_pc'][:, :3])
        self.edge_loss, self.normal_loss, self.cd_loss, self.laplace_loss, self.move_loss = detail_loss(self.input,
                                                                                                        self.output,
                                                                                                        self.gt_pcn,
                                                                                                        self.edges,
                                                                                                        self.lape_idx)
        self.total_loss += self.edge_loss
        self.total_loss += self.normal_loss
        self.total_loss += self.cd_loss
        self.total_loss += self.laplace_loss
        self.total_loss += self.move_loss

    def refine_loss(self, verts_in, verts_out):
        edge_loss, norm_loss, cd_loss, lap_loss, move_loss = detail_loss(verts_in, verts_out, self.gt_pcn, self.edges,
                                                                         self.lape_idx)
        return edge_loss+norm_loss+cd_loss+lap_loss+move_loss