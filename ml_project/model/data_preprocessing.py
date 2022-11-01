import yaml
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("data_processor")

def save_file(path: str, data: np.array) -> None:
	with open(path, 'wb') as f:
		np.save(f, data)

def preprocess_data(data_settings: dict):

	logger.info('Read data from {}'.format(data_settings['path']))
	df = pd.read_csv(data_settings['path'])

	logger.info('Drop colums {}'.format(data_settings['skiped_columns']))
	df = df.drop(columns=data_settings['skiped_columns'])

	y = df['condition'].values
	X = df.drop(columns=['condition']).values
	X_train, X_test, y_train, y_test = train_test_split(
		X, 
		y, 
		test_size=data_settings['test_size'], 
		shuffle=data_settings['shuffle'],
		random_state=data_settings['random_state']
	)

	logger.info("Save data")
	save_file(data_settings['train_data_path'], X_train)
	save_file(data_settings['train_targer_path'], y_train)
	save_file(data_settings['test_data_path'], X_test)
	save_file(data_settings['test_targer_path'], y_test)

	logger.info("Preprocess done")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", default='config/type1.yaml', type=str, help='Path to config')
	args = parser.parse_args()

	logger.info('Read config from {}'.format(args.config_path))
	with open(args.config_path, 'r') as f:
		config = yaml.safe_load(f)
	logger.info(config['data_preprocessing'])

	preprocess_data(config['data_preprocessing'])


