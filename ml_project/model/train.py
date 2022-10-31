import os
import yaml
import pickle
import logging
from typing import Dict, Union

from data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("trainer")

class Trainer:

	def __init__(self, config_path):

		logger.info('Read config from {}'.format(config_path))
		with open(config_path, 'r') as stream:
			self.config = yaml.safe_load(stream)
		logger.info(self.config)

	def train(self):
		train_data, _ = preprocess_data(self.config['data'])
		X, y = train_data
		clf = LogisticRegression()
		clf.fit(X, y)
		print(classification_report(y, clf.predict(X)))

if __name__ == "__main__":
	trainer = Trainer("config/type1.yaml")
	trainer.train()