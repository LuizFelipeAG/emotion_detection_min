from fastapi import FastAPI
from utils.io_utils import load_config
from utils.model_utils import loadmodel, load_cascade, facecrop, predict

config = load_config()
model = loadmodel(config["paths"]["model"])
facedata = load_cascade(config["paths"]["facedata"])

app = FastAPI()

@app.get("/")
def read_root():
    return "Up and running"

@app.get("/predict/")
def predict_url(url):
    """Prediction Endpoint
    Args:
        url ([str]): image url
    Returns:
        [dict]: prediction dictionary
    """
    return predict(url,model)