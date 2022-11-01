import os
import yaml
import pickle
import logging
import argparse

import numpy as np
from sklearn.metrics import classification_report

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("evaluator")

class Evaluator:

	def __init__(self, config):
		self.config = config

	def __load_data(self, path):
		return np.load(path)

	def __load_model(self, path):
		return pickle.load(open(path, 'rb'))

	def eval(self):
		logger.info("Load data")
		X = self.__load_data(self.config['test_data_path'])
		y = self.__load_data(self.config['test_targer_path'])

		model = self.__load_model(self.config['model_path'])

		logger.info("EVAL")
		
		logger.info("Metrics on test set")
		logger.info(str(classification_report(y, model.predict(X))))

		logger.info("Eval done")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", default='config/type1.yaml', type=str, help='Path to config')
	args = parser.parse_args()

	logger.info('Read config from {}'.format(args.config_path))
	with open(args.config_path, 'r') as f:
		config = yaml.safe_load(f)
	logger.info(config['eval'])

	evaluator = Evaluator(config['eval'])
	evaluator.eval()