from fastapi import FastAPI
import uvicorn
from typing import Union
import joblib
import numpy as np
import pandas as pd
from transformer import Droper, Faisser
import faiss
from itertools import chain

app = FastAPI(
    title='Matching'
)
dim = 72

@app.on_event("startup")
def start():
    global model, naming
    model, naming = joblib.load('faiss_pipe.joblib')

@app.get("/knn")
def parse_string(string: str) -> list[float]:
    list_str = string.split(',')
    
    output = np.array([float(el) for el in list_str])
    if len(list_str) != dim:
        output = None
    arr = match(output)
    return arr

def match(item = None) -> dict:
    if item is None:
        return {"status":"fail", "message":"No inputs data"}
    
    idx = list(chain(*model.predict([item])))

    result = []
    for i in idx:
        result.append(naming[i])

    return {'status':200, "data":result}


@app.get('/')
def main():
    return {"status":"OK",
            "message":"Hello"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8031)