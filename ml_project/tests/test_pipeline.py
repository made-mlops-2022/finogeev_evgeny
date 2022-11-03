import os
import yaml
import argparse

from model.train_pipeline import pipeline

if not os.path.exists("tests/test_data"):
    os.mkdir("tests/test_data")


def test_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='config/test_config.yaml', type=str, help='Path to config')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    if os.path.exists(config['data_preprocessing']['train_data_path']):
        os.remove(config['data_preprocessing']['train_data_path'])
    if os.path.exists(config['data_preprocessing']['train_targer_path']):
        os.remove(config['data_preprocessing']['train_targer_path'])
    if os.path.exists(config['data_preprocessing']['test_data_path']):
        os.remove(config['data_preprocessing']['test_data_path'])
    if os.path.exists(config['data_preprocessing']['test_targer_path']):
        os.remove(config['data_preprocessing']['test_targer_path'])
    if os.path.exists(config['train']['model_save_path']):
        os.remove(config['train']['model_save_path'])

    result = pipeline(args)

    assert os.path.exists(config['train']['model_save_path'])
    assert result['macro avg']['f1-score'] > 0.5
