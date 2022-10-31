import pickle
import logging
from typing import Dict, Union

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("data_processor")

def preprocess_data(data_settings: Dict):

	logger.info('Read data from {}'.format(data_settings['path']))
	df = pd.read_csv(data_settings['path'])

	logger.info('Drop colums {}'.format(data_settings['skiped_columns']))
	df.drop(columns=data_settings['skiped_columns'])

	y = df['condition'].values
	X = df.drop(columns=['condition']).values
	X_train, X_test, y_train, y_test = train_test_split(
		X, 
		y, 
		test_size=data_settings['test_size'], 
		shuffle=data_settings['shuffle'],
		random_state=data_settings['random_state']
	)
	logger.info("Preprocess done")
	return (X_train, y_train), (X_test, y_test)

