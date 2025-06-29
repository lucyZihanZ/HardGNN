#!/usr/bin/env python3
"""
Main entry point for HardGNN model execution
Compatible with Google Colab Pro+ environment
"""

import os
import sys
import numpy as np
import tensorflow as tf
from time import time

# Enable TensorFlow 1.x compatibility for TF 2.x
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# Configure GPU memory growth for Colab Pro+
try:
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
	pass  # GPU configuration warnings

from Params import args
from HardGNN_model import Recommender
from DataHandler import DataHandler
import Utils.TimeLogger as logger

# Main configuration loaded

def main():
	logger.saveDefault = True
	
	# Set random seeds for reproducibility
	np.random.seed(args.seed)
	tf.compat.v1.set_random_seed(args.seed)
	
	# Initialize data handler
	handler = DataHandler()
	handler.LoadData()
	
	# Configure TensorFlow session
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.compat.v1.Session(config=config) as sess:
		# Initialize recommender
		recom = Recommender(sess, handler)
		
		# Run the experiment
		recom.run()

if __name__ == '__main__':
	main() 