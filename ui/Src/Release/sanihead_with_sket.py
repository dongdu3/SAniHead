import os
import cv2
import numpy as np
import tensorflow as tf
import models_single
import models_dual
import models_flc
from model import SAniHeadEnd2End

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_predicted_obj(mesh_name):
	point_set = []
	with open(mesh_name) as f:
		for line in f.readlines():
			line = line.strip('\n')
			c, x, y, z = line.split(' ')
			if c == 'v':
				x = np.float32(x)
				y = np.float32(y)
				z = np.float32(z)
				point_set.append([x, y, z])
			else:
				break
		f.close()
	point_set = np.array(point_set).reshape(len(point_set), 3)
	return point_set

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

faces = np.loadtxt('./utils/sphere/face3.obj', dtype='|S32')

# model_single for single-view animal head generation/deformation
graph_single = tf.Graph()
sess_single = tf.Session(config=config, graph=graph_single)
with sess_single.as_default():
	with graph_single.as_default():
		model_single = models_single.SAniHead(sess=sess_single)
		sess_single.run(tf.global_variables_initializer())
		model_single.load()
		in_sket = cv2.resize(cv2.imread('sketch_front_0.png'), (224, 224), interpolation=cv2.INTER_CUBIC) / 255.0
		verts_single = sess_single.run(model_single.vert3, feed_dict={model_single.in_sket: in_sket})
		# verts_single = np.hstack((np.full([verts_single.shape[0], 1], 'v'), verts_single))
		# mesh_single = np.vstack((verts_single, faces))
		# np.savetxt('./predict_single.obj', mesh_single, fmt='%s', delimiter=' ')

# # model_dual for dual-view animal head generation/deformation
# graph_dual = tf.Graph()
# sess_dual = tf.Session(config=config, graph=graph_dual)
# with sess_dual.as_default():
# 	with graph_dual.as_default():
# 		model_dual = models_dual.SAniHead(sess=sess_dual)
# 		sess_dual.run(tf.global_variables_initializer())
# 		model_dual.load()
# 		in_sket_front = cv2.resize(cv2.imread('sketch_front_0.png'), (224, 224), interpolation=cv2.INTER_CUBIC) / 255.0
# 		in_sket_side = cv2.resize(cv2.imread('sketch_left_0.png'), (224, 224), interpolation=cv2.INTER_CUBIC) / 255.0
# 		verts_dual = sess_dual.run(model_dual.vert3, feed_dict={model_dual.in_sket_front: in_sket_front,
# 																model_dual.in_sket_side: in_sket_side})
# 		# verts_dual = np.hstack((np.full([verts_dual.shape[0], 1], 'v'), verts_dual))
# 		# mesh_dual = np.vstack((verts_dual, faces))
# 		# np.savetxt('./predict_dual.obj', mesh_dual, fmt='%s', delimiter=' ')

# model_whole for sketch-based animal head generation/deformation
graph_whole = tf.Graph()
sess_whole = tf.Session(config=config, graph=graph_whole)
with sess_whole.as_default():
	with graph_whole.as_default():
		model_whole = SAniHeadEnd2End(sess=sess_whole)
		sess_whole.run(tf.global_variables_initializer())
		model_whole.load()
		# initial test
		sket_batch = []
		sket_batch.append(cv2.imread('./sketch_front_0.png'))
		sket_batch.append(cv2.imread('./sketch_left_0.png'))
		sket_batch.append(cv2.imread('./sketch_right_0.png'))
		# sket_batch.append(cv2.flip(sket_batch[1], 1))
		sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC)/255.
		sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC)/255.
		verts_whole = sess_whole.run(model_whole.output_refine_2,
										 feed_dict={model_whole.model_p2m.in_sket_front: sket_front,
													model_whole.model_p2m.in_sket_side: sket_side,
													model_whole.model_vex2vdis.in_sket: sket_batch})
		# verts_whole = np.hstack((np.full([verts_whole.shape[0], 1], 'v'), verts_whole))
		# mesh_whole = np.vstack((verts_whole, faces))
		# np.savetxt('./predict_whole.obj', mesh_whole, fmt='%s', delimiter=' ')

# model_flc for feature line points inference
graph_flc = tf.Graph()
sess_flc = tf.Session(config=config, graph=graph_flc)
with sess_flc.as_default():
	with graph_flc.as_default():
		model_flc = models_flc.FLCNet(sess=sess_flc)
		sess_flc.run(tf.global_variables_initializer())
		model_flc.load()
		in_sket = cv2.resize(cv2.imread('./sketch_front_0.png'), (224, 224), interpolation=cv2.INTER_CUBIC)
		in_sket = in_sket.astype('float32') / 255.0
		in_verts = read_predicted_obj('./predict.obj')
		label_prob = sess_flc.run(model_flc.out_label_logits,
								  feed_dict={model_flc.in_pc: in_verts, model_flc.in_sket: in_sket})
		# np.savetxt('./predict_label.txt', label_prob, fmt='%f')

def predict_single(sket_name):
	with sess_single.as_default():
		with graph_single.as_default():
			in_sket = cv2.resize(cv2.imread(sket_name), (224, 224), interpolation=cv2.INTER_CUBIC) / 255.0
			verts_single = sess_single.run(model_single.vert3, feed_dict={model_single.in_sket: in_sket})
			verts_single = np.hstack((np.full([verts_single.shape[0], 1], 'v'), verts_single))
			mesh_single = np.vstack((verts_single, faces))
			np.savetxt('predict.obj', mesh_single, fmt='%s', delimiter=' ')

# def predict_dual(sket_name):
# 	with sess_dual.as_default():
# 		with graph_dual.as_default():
# 			in_sket_front = cv2.resize(cv2.imread(sket_name + '_front.png'), (224, 224),
# 									   interpolation=cv2.INTER_CUBIC) / 255.0
# 			in_sket_side = cv2.resize(cv2.imread(sket_name + '_left.png'), (224, 224),
# 									  interpolation=cv2.INTER_CUBIC) / 255.0
# 			verts_dual = sess_dual.run(model_dual.vert3, feed_dict={model_dual.in_sket_front: in_sket_front,
# 																	model_dual.in_sket_side: in_sket_side})
# 			verts_dual = np.hstack((np.full([verts_dual.shape[0], 1], 'v'), verts_dual))
# 			mesh_dual = np.vstack((verts_dual, faces))
# 			np.savetxt('predict.obj', mesh_dual, fmt='%s', delimiter=' ')

def predict_whole(sket_name):
	with sess_whole.as_default():
		with graph_whole.as_default():
			# initial test
			sket_batch = []
			sket_batch.append(cv2.resize(cv2.imread(sket_name + '_front.png'), (256, 256), interpolation=cv2.INTER_CUBIC))
			sket_batch.append(cv2.resize(cv2.imread(sket_name + '_left.png'), (256, 256), interpolation=cv2.INTER_CUBIC))
			# sket_batch.append(cv2.imread(sket_name + '_right.png'))
			sket_batch.append(cv2.flip(sket_batch[1], 1))
			sket_front = cv2.resize(sket_batch[0], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
			sket_side = cv2.resize(sket_batch[1], (224, 224), interpolation=cv2.INTER_CUBIC) / 255.
			verts_whole = sess_whole.run(model_whole.output_refine_2,
										 feed_dict={model_whole.model_p2m.in_sket_front: sket_front,
													model_whole.model_p2m.in_sket_side: sket_side,
													model_whole.model_vex2vdis.in_sket: sket_batch})
			verts_whole = np.hstack((np.full([verts_whole.shape[0], 1], 'v'), verts_whole))
			mesh_whole = np.vstack((verts_whole, faces))
			np.savetxt('predict.obj', mesh_whole, fmt='%s', delimiter=' ')

def predict_flc(mesh_path):
	with sess_flc.as_default():
		with graph_flc.as_default():
			if os.path.exists('sketch_front.png'):
				in_sket = cv2.resize(cv2.imread('sketch_front.png'), (224, 224), interpolation=cv2.INTER_CUBIC)
			else:
				in_sket = cv2.resize(cv2.imread('sketch_left.png'), (224, 224), interpolation=cv2.INTER_CUBIC)
			in_sket = in_sket.astype('float32') / 255.0
			in_verts = read_predicted_obj(mesh_path)
			label_prob = sess_flc.run(model_flc.out_label_logits,
									  feed_dict={model_flc.in_pc: in_verts, model_flc.in_sket: in_sket})
			np.savetxt('predict_label.txt', label_prob, fmt='%f')
