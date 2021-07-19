import streamlit as st
import requests
from utils.io_utils import load_config
import numpy as np

config = load_config()

st.title("Facial Emotion Classifier")
url = st.text_input("Image url")

if url:
    response = requests.get(config["api"]["prediction_url"], params={"url": url})
    imageurl = response.json()['url']
    anger = response.json()['anger']
    disgust = response.json()['disgust']
    fear = response.json()['fear']
    happiness = response.json()['happiness']
    neutral = response.json()['neutral']
    sadness = response.json()['sadness']
    surprise = response.json()['surprise']
    sub_face = response.json()['sub_face']
    cutimage = np.asarray(sub_face.strip('[]').split(', ')).astype(np.float32).reshape(224, 224, 3)
    st.write(url)
    st.image([imageurl,cutimage], width=224, caption=["Original Image","Analyzed Crop"])
    st.write("Detected Emotions:")
    st.write("Anger : ","{:.8f}%".format(anger * 100),"  \n",
    "Disgust : ","{:.8f}%".format(disgust * 100),"  \n",
    "Fear : ","{:.8f}%".format(fear * 100),"  \n",
    "Happiness : ","{:.8f}%".format(happiness * 100),"  \n",
    "Neutral : ","{:.8f}%".format(neutral * 100),"  \n",
    "Sadness : ","{:.8f}%".format(sadness * 100),"  \n",
    "Surprise : ","{:.8f}%".format(surprise * 100))