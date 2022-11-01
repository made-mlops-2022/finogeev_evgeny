import os
import yaml
import pickle
import logging
import argparse

import numpy as np

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("predictor")

class Predictor:

	def __init__(self, config):
		self.config = config

	def __load_data(self, path):
		return np.load(path)

	def __load_model(self, path):
		return pickle.load(open(path, 'rb'))

	def save_file(self, path: str, data: np.array) -> None:
		with open(path, 'wb') as f:
			np.save(f, data)

	def predict(self):
		logger.info("Load data")
		X = self.__load_data(self.config['data_path'])

		model = self.__load_model(self.config['model_path'])

		result = model.predict(X)
		self.save_file(self.config['result_path'], result)

		logger.info("Predict done")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", default='config/type1.yaml', type=str, help='Path to config')
	args = parser.parse_args()

	logger.info('Read config from {}'.format(args.config_path))
	with open(args.config_path, 'r') as f:
		config = yaml.safe_load(f)
	logger.info(config['eval'])

	predictor = Predictor(config['predict'])
	predictor.predict()