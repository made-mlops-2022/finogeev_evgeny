# coding=utf-8
import os
import re
import sys
import time
import uuid
import logging
import numpy as np
from typing import Union, List
from concurrent import futures

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from model.inference import Predictor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")

app = FastAPI()

server_ready = False
predictor = None


@app.on_event("startup")
def init():
    global predictor
    global server_ready

    url = os.environ.get("MODEL_URL", "https://disk.yandex.com/d/rg5hkoJsMxAd4A")
    logger.info("Model URL {}".format(url))

    os.system("curl -L $(yadisk-direct {}) -o ./weights/model_current.pickle".format(url))

    predictor = Predictor("./weights/model_current.pickle")
    server_ready = True

@app.get("/health")
async def health():
    logger.info("Health request")
    if server_ready:
        return 
    else:
        raise HTTPException(status_code=421)

@app.post("/predict")
async def predict(file: UploadFile):
    logger.info("Predict request")
    response = {}

    name = str(uuid.uuid4())

    file_to_save = await file.read()
    file_path = os.path.join("./tmp_for_upload", name)

    with open(file_path, 'wb') as f:
        f.write(file_to_save)

    try:
        X = np.load(file_path)
    except Exception as e:
        raise HTTPException(status_code=400)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    y = predictor.predict(X)

    response['predict'] = y.tolist() 

    return response
