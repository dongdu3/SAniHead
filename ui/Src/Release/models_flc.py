from p2m.layers import *
import pickle
import os

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.name = 'flcnet'
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.layers_flc = []
        self.activations_flc = []

        self.sess = None
        self.in_pc = None
        self.out_label = None
        self.out_label_logits = None
        self.support = None

        self.label_loss = 0

        self.optimizer = None
        self.opt_op = None

        self.model_path = 'utils/checkpoint_flc/'

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        #with tf.device('/gpu:0'):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model for pixel2mesh
        eltwise_flc = [2, 4, 6, 8, 10, 12]
        self.activations_flc.append(self.in_pc)
        for idx, layer in enumerate(self.layers_flc):
            hidden = layer(self.activations_flc[-1])
            if idx in eltwise_flc:
                hidden = tf.add(hidden, self.activations_flc[-2]) * 0.5
            self.activations_flc.append(hidden)

        self.out_label = self.activations_flc[-1]
        self.out_label_logits = tf.nn.sigmoid(self.activations_flc[-1])

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def load(self):
        if not self.sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = os.path.join(self.model_path, self.name + '.ckpt')
        saver.restore(self.sess, save_path)
        print("Model restored from file: %s" % save_path)

class FLCNet(Model):
    def __init__(self, sess=None, init_infor_file='utils/sphere/init_info.dat', hidden=192, coord_dim=3, **kwargs):
        super(FLCNet, self).__init__(**kwargs)

        self.sess = sess
        self.init_infor_file = init_infor_file
        self.hidden = hidden
        self.coord_dim = coord_dim

        self.in_pc = tf.placeholder(tf.float32, shape=(None, 3), name='in_pc')

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
        num_supports = 2
        pkl = pickle.load(open(self.init_infor_file, 'rb'), encoding='latin1')
        self.support = [tf.sparse.SparseTensor(pkl[3][i][0], tf.cast(pkl[3][i][1], tf.float32), pkl[3][i][2]) for i in
                        range(num_supports)]

    def _build(self):
        self.layers_flc.append(GraphConvolution(input_dim=self.coord_dim,
                                                output_dim=self.hidden,
                                                support=self.support,
                                                logging=self.logging,
                                                name='flc'))
        for _ in range(13):
            self.layers_flc.append(GraphConvolution(input_dim=self.hidden,
                                                    output_dim=self.hidden,
                                                    support=self.support,
                                                    logging=self.logging,
                                                    name='flc'))
        self.layers_flc.append(GraphConvolution(input_dim=self.hidden,
                                                output_dim=1,
                                                support=self.support,
                                                act=lambda x: x,
                                                logging=self.logging,
                                                name='flc'))
