import pickle
import tensorflow as tf
from layers import *

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)), # initial 3D coordinates
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], # helper for normal loss
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] # helper for graph unpooling
}
feed_dict = dict()

# Construct feed dictionary
def construct_feed_dict(pkl, placeholders):
	coord = pkl[0]
	pool_idx = pkl[4]
	edges = []
	for i in range(1, 4):
		adj = pkl[i][1]
		edges.append(adj[0])

	feed_dict.update({placeholders['features']: coord})
	feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
	feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})

# Construct feed dictionary
pkl = pickle.load(open('init_info.dat', 'rb'))
construct_feed_dict(pkl, placeholders)


pool_layer1 = GraphPooling(placeholders=placeholders, pool_id=1)
pool_layer2 = GraphPooling(placeholders=placeholders, pool_id=2)

pool_out1 = pool_layer1(placeholders['features'])
pool_out2 = pool_layer2(pool_out1)

# Initialize session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

vert1 = pkl[0]
vert1 = np.hstack((np.full([vert1.shape[0], 1], 'v'), vert1))
face1 = np.loadtxt('face1.obj', dtype='|S32')
mesh1 = np.vstack((vert1, face1))
np.savetxt('mesh1.obj', mesh1, fmt='%s', delimiter=' ')

vert2 = sess.run(pool_out1, feed_dict=feed_dict)
vert2 = np.hstack((np.full([vert2.shape[0], 1], 'v'), vert2))
face2 = np.loadtxt('face2.obj', dtype='|S32')
mesh2 = np.vstack((vert2, face2))
np.savetxt('mesh2.obj', mesh2, fmt='%s', delimiter=' ')

vert3 = sess.run(pool_out2, feed_dict=feed_dict)
vert3 = np.hstack((np.full([vert3.shape[0], 1], 'v'), vert3))
face3 = np.loadtxt('face3.obj', dtype='|S32')
mesh3 = np.vstack((vert3, face3))
np.savetxt('mesh3.obj', mesh3, fmt='%s', delimiter=' ')

print('gen_template_meshes done!')