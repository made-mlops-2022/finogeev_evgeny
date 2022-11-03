import os
import numpy as np

from model import data_preprocessing
from model.train import Trainer
from model.eval import Evaluator

if not os.path.exists("tests/test_data"):
	os.mkdir("tests/test_data")

def init_data():
	test_input = {
		'path': 'data/test_generated_data.csv',
		"test_size": 0.5,
		"shuffle": False,
		"random_state": 888,
		"skiped_columns": [],

		"train_data_path": "tests/test_data/X_train.npy",
		"train_targer_path": "tests/test_data/Y_train.npy",
		"test_data_path": "tests/test_data/X_test.npy",
		"test_targer_path": "tests/test_data/Y_test.npy",
	}

	data_preprocessing.preprocess_data(test_input)

	train_input = {
	  "train_data_path": "tests/test_data/X_train.npy",
	  "train_targer_path": "tests/test_data/Y_train.npy",
	  
	  "algorithm": "logigstic_regression", # logigstic_regression, decision_tree, random_forest
	  "logigstic_regression": {
	    "penalty": 'l2',
	    "C": 1.0,
	    "solver": 'lbfgs',
	    "max_iter": 100
	    },

	  "model_save_path": "tests/test_data/model.pickle"
	}
	if os.path.exists(train_input['model_save_path']):
		os.remove(train_input['model_save_path'])

	trainer = Trainer(train_input)
	trainer.train()


def test_eval1():
	init_data()

	train_input = {
	  	"model_path": "tests/test_data/model.pickle",
  		"test_data_path": "tests/test_data/X_test.npy",
  		"test_targer_path": "tests/test_data/Y_test.npy"
	}

	evaluator = Evaluator(train_input)
	result = evaluator.eval()
	assert result['macro avg']['f1-score'] > 0.5
