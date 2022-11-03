import yaml
import pickle
import logging
import argparse

import numpy as np
from model.data_preprocessing import DropTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")


class Trainer:
    def __init__(self, config):

        self.config = config

    def __load_data(self, path):
        return np.load(path)

    def __save_model(self, model, path):
        pickle.dump(model, open(path, 'wb'))

    def train(self):
        logger.info("Load data")
        X = self.__load_data(self.config['train_data_path'])
        y = self.__load_data(self.config['train_targer_path'])

        if self.config['drop_X_prob'] > 0 and self.config['drop_X_prob'] < 1:
            logger.info("DropTransformer apply")
            logger.info("Prob = {}".format(self.config['drop_X_prob']))
            trans = DropTransformer(self.config['drop_X_prob'])
            trans.fit(X)
            X = trans.transform(X)
        elif self.config['drop_X_prob'] > 1:
            logger.info("DropTransformer has incorrect prob. Skip aplly")

        if self.config['algorithm'] == 'logigstic_regression':
            logger.info("Algorithm - LogisticRegression")
            logger.info(self.config[self.config['algorithm']])
            model = LogisticRegression(**self.config[self.config['algorithm']])

        elif self.config['algorithm'] == 'decision_tree':
            logger.info("Algorithm - decision_tree")
            logger.info(self.config[self.config['algorithm']])
            model = DecisionTreeClassifier(**self.config[self.config['algorithm']])

        elif self.config['algorithm'] == 'random_forest':
            logger.info("Algorithm - random_forest")
            logger.info(self.config[self.config['algorithm']])
            model = RandomForestClassifier(**self.config[self.config['algorithm']])

        else:
            logger.info("ERROR!!! Algorithm NOT FOUND!!! BRAKE")
            return

        logger.info("TRAIN")
        model.fit(X, y)

        logger.info("Metrics on train set")
        logger.info(str(classification_report(y, model.predict(X))))

        logger.info("Save model")
        self.__save_model(model, self.config['model_save_path'])

        logger.info("Train done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='config/type1.yaml', type=str, help='Path to config')
    args = parser.parse_args()

    logger.info('Read config from {}'.format(args.config_path))
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(config['train'])

    trainer = Trainer(config['train'])
    trainer.train()


if __name__ == "__main__":
    main()
