import os

from model import data_preprocessing
from model.train import Trainer

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


def test_train1():
    init_data()

    train_input = {
      "train_data_path": "tests/test_data/X_train.npy",
      "train_targer_path": "tests/test_data/Y_train.npy",

      "algorithm": "logigstic_regression",  # logigstic_regression, decision_tree, random_forest
      "drop_X_prob": 0.2,
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
    assert os.path.exists(train_input['model_save_path'])


def test_train2():
    init_data()

    train_input = {
      "train_data_path": "tests/test_data/X_train.npy",
      "train_targer_path": "tests/test_data/Y_train.npy",

      "algorithm": "decision_tree",  # logigstic_regression, decision_tree, random_forest
      "drop_X_prob": 0.2,
      "decision_tree": {
        "criterion": 'gini',
        "splitter": 'best',
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        },

      "model_save_path": "tests/test_data/model.pickle"
    }
    if os.path.exists(train_input['model_save_path']):
        os.remove(train_input['model_save_path'])

    trainer = Trainer(train_input)
    trainer.train()
    assert os.path.exists(train_input['model_save_path'])


def test_train3():
    init_data()

    train_input = {
      "train_data_path": "tests/test_data/X_train.npy",
      "train_targer_path": "tests/test_data/Y_train.npy",

      "algorithm": "random_forest",  # logigstic_regression, decision_tree, random_forest
      "drop_X_prob": 0.2,
      "random_forest": {
        "n_estimators": 100,
        "criterion": 'gini',
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        },

      "model_save_path": "tests/test_data/model.pickle"
    }
    if os.path.exists(train_input['model_save_path']):
        os.remove(train_input['model_save_path'])

    trainer = Trainer(train_input)
    trainer.train()
    assert os.path.exists(train_input['model_save_path'])
