# Fichier dans lequel nous allons coder l'API avec FastAPI
# pip install fastapi
# pip install uvicorn

# librairies
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris


iris = load_iris()

# chargement du modèle
loaded_model = load('logreg.joblib')

# création d'une nouvelle instance API
app = FastAPI()

# Définir un objet (une classe) pour réaliser des requêtes
# dot notation ( . )

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# Définition du chemin de l'endpoint

# route pour l'accueil
@app.get('/')
def home():
    return "Application de FastAPI"

@app.post('/predict') # local : http://localhost:8000/predict

# Définition de la fonction de prédiction
def predict(data : request_body):
    # Nouvelles données sur lesquelles on fait la prédiction
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # prediction
    class_idx = loaded_model.predict(new_data)[0]

    # retourne le nom de l'espèce d'iris
    return { 'class': iris.target_names[class_idx]}

