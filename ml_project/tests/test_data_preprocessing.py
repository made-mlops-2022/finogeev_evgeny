import os

import numpy as np
from model import data_preprocessing


def test_data_processing1():
	test_input = {
		'path': 'data/heart_cleveland_upload.csv',
		"test_size": 0.5,
		"shuffle": False,
		"random_state": 888,
		"skiped_columns": [],

		"train_data_path": "tests/test_data/X_train.npy",
		"train_targer_path": "tests/test_data/Y_train.npy",
		"test_data_path": "tests/test_data/X_test.npy",
		"test_targer_path": "tests/test_data/Y_test.npy",
	}

	if os.path.exists(test_input['train_data_path']):
		os.remove(test_input['train_data_path'])
	if os.path.exists(test_input['train_targer_path']):
		os.remove(test_input['train_targer_path'])
	if os.path.exists(test_input['test_data_path']):
		os.remove(test_input['test_data_path'])
	if os.path.exists(test_input['test_targer_path']):
		os.remove(test_input['test_targer_path'])

	data_preprocessing.preprocess_data(test_input)

	assert os.path.exists(test_input['train_data_path'])
	assert os.path.exists(test_input['train_targer_path'])
	assert os.path.exists(test_input['test_data_path'])
	assert os.path.exists(test_input['test_targer_path'])

	with open(test_input['train_data_path'], 'rb') as f:
		X = np.load(f)
	assert X.shape[1] == 13

	with open(test_input['train_targer_path'], 'rb') as f:
		y = np.load(f)
	assert X.shape[0] == y.shape[0]


def test_data_processing2():
	test_input = {
		'path': 'data/heart_cleveland_upload.csv',
		"test_size": 0.5,
		"shuffle": False,
		"random_state": 888,
		"skiped_columns": ['sex', 'cp'],

		"train_data_path": "tests/test_data/X_train.npy",
		"train_targer_path": "tests/test_data/Y_train.npy",
		"test_data_path": "tests/test_data/X_test.npy",
		"test_targer_path": "tests/test_data/Y_test.npy",
	}

	data_preprocessing.preprocess_data(test_input)

	assert os.path.exists(test_input['train_data_path'])
	assert os.path.exists(test_input['train_targer_path'])
	assert os.path.exists(test_input['test_data_path'])
	assert os.path.exists(test_input['test_targer_path'])

	with open(test_input['train_data_path'], 'rb') as f:
		X = np.load(f)
	assert X.shape[1] == 11

	with open(test_input['train_targer_path'], 'rb') as f:
		y = np.load(f)
	assert X.shape[0] == y.shape[0]


