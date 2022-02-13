import os
import argparse
import numpy as np
import tensorflow as tf
from model import SAniHeadEnd2End

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('-mode', '--mode', default='train', help='input running mode is train or test')
args = parser.parse_args()

# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

def train_model():
	# Initialize session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	# model = SAniHeadEnd2End(sess=sess, epochs=20, lr=3e-5, data_root='/data/dudong/SAniHead/',
	model = SAniHeadEnd2End(sess=sess, epochs=30, lr=2e-5, data_root='/data/dudong/SAniHead/',
							checkpoint_dir='./checkpoint', sample_dir='./sample', log_dir='./logs', test_dir='./test')
	# model = SAniHeadEnd2End(sess=sess, epochs=50, lr=5e-5, data_root='/media/administrator/Data/SAniHead/',
	# 						checkpoint_dir='./checkpoint', sample_dir='./sample', log_dir='./logs', test_dir='./test')
	model.train()

def test_model():
	# Initialize session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# config.allow_soft_placement=True
	sess = tf.Session(config=config)
	# model = SAniHeadEnd2End(sess=sess, epochs=30, lr=3e-5, data_root='/data/dudong/SAniHead/',
	# 						checkpoint_dir='./checkpoint', sample_dir='./sample', log_dir='./logs', test_dir='./test')
	model = SAniHeadEnd2End(sess=sess, epochs=30, lr=2e-5, data_root='/data/dudong/SAniHead/',
							checkpoint_dir='./checkpoint', sample_dir='./sample', log_dir='./logs', test_dir='./test')
	model.test()

def main(_):
	if args.mode == 'train':
		print('begin to train ...')
		train_model()
	elif args.mode == 'test':
		print('begin to test ...')
		test_model()

if __name__ == '__main__':
	tf.app.run()

