import pickle
import logging
import argparse

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor")


class Predictor:

    def __init__(self, path):
        self.model = pickle.load(open(path, 'rb'))

    def predict(self, X):

        logger.info("Predict")
        result = self.model.predict(X)
        logger.info("Predict done")

        return result
