import os
import yaml
import pickle
from typing import Dict, Union

from sklearn.linear_model import LogisticRegression

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("trainer")

class Trainer:

	def __init__(self, config_path):

		logger.info('Read config from {}'.format(config_path))
		with open(config_path, 'r') as stream:
    		self.config = yaml.safe_load(stream)

    def train(self):
