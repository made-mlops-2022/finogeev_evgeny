import yaml
import logging
import argparse

from model.data_preprocessing import preprocess_data
from model.train import Trainer
from model.eval import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline")


def pipeline(args):
    logger.info('Read config from {}'.format(args.config_path))
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(config)

    trainer = Trainer(config['train'])

    evaluator = Evaluator(config['eval'])

    preprocess_data(config['data_preprocessing'])
    trainer.train()
    result = evaluator.eval()

    logger.info("Pipeline done")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='config/type1.yaml', type=str, help='Path to config')
    args = parser.parse_args()

    pipeline(args)


if __name__ == "__main__":
    main()
