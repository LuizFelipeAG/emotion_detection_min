emotion_detection_min
==============================

My WIP facial emotion recognition model project from Awari ML School.

The used dataset is a mix of the FER+ and ExpW datasets, stratified and divided in 80/20.
You can download my zip file here or follow the code on one of the notebooks.

https://drive.google.com/file/d/1toh7GhPI8aSobYyGmyzaP0jB2TFp11St/

Using Tensorflow and OpenCV

Project Organization
------------

    ├── LICENSE
    ├── README.md                       <- This file.
    ├── api                             <- Docker container setup for my demo - Using streamlit and fastapi
    ├── data
    │   ├── train                       <- Training Data.
    │   ├── test                        <- Validation Data.
    │   ├── mixed_dataset.zip           <- The dataset - url:  https://drive.google.com/file/d/1toh7GhPI8aSobYyGmyzaP0jB2TFp11St
    │
    │
    ├── models                          <- Trained and serialized models, model predictions, or model summaries
    │   ├── model_tl_full_224x224.h5    <- rensnet50 model retrained on my dataset - url: https://drive.google.com/file/d/1sNKudyAw7TpCuMVem9jX48eV6N6IwZIV
    |
    └── notebooks                       <- Jupyter notebooks

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
