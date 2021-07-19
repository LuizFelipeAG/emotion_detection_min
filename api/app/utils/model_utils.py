from os import path
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model

def loadmodel(path):
    """Load model function
    Args:
        path (string): path for the model file
    Returns:
        model: model instance
    """
    model = load_model(path)
    return model

def load_cascade(path):
    """Load haarcascades xml file
    Args:
        path (string): path for the cascade file
    Returns:
        model: cascade file
    """
    global facedata
    facedata= path
    return facedata

def facecrop(url):
    """Face capture function. Grab the url and cut a small rectangle containing only
    the relevant face information.
    Args:
        url (string): image url
        Returns:
        vector with the relevant face information
    """
    global sub_face
    resp = requests.get(url, stream=True, timeout=5).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    cascade=cv2.CascadeClassifier(facedata)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    minisize=(img.shape[1],img.shape[0])
    miniframe=cv2.resize(img,minisize) 
    faces=cascade.detectMultiScale(miniframe)
    for f in faces:
        x,y,w,h = [v for v in f]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        sub_face = img[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        sub_face = np.expand_dims(sub_face, axis=0)/255
    return sub_face

def predict(url,model):
    """Funcao de previsao
    Args:
        url (string): image url
        model (model.istance): emotion detection model
    Returns:
        dict: dict with emotion and probability detected
    """
    sub_face = facecrop(url)
    probas = model.predict(sub_face)
    return {
        "url": url,
        "anger": probas.item(0),
        "disgust": probas.item(1),
        "fear": probas.item(2),
        "happiness": probas.item(3),
        "neutral": probas.item(4),
        "sadness": probas.item(5),
        "surprise": probas.item(6),
        "sub_face": str(sub_face.flatten().tolist()),
    }
