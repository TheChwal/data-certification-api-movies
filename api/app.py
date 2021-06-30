
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(original_title,title,release_date,duration_min,description,budget,original_language,status,number_of_awards_won,number_of_nominations,has_collection,all_genres,top_countries,number_of_top_productions, available_in_english):
    X_input = {
        "original_title": original_title,
        "title": title,
        "release_date": release_date,
        "duration_min" : [float(duration_min)],
        "description" : description,
        "budget" : [float(budget)],
        "original_language" : original_language,
        "status" : status,
        "number_of_awards_won" : [int(number_of_awards_won)],
        "number_of_nominations" : [int(number_of_nominations)],
        "has_collection" : [int(has_collection)],
        "all_genres" : all_genres,
        "top_countries" : top_countries,
        "number_of_top_productions" : [float(number_of_top_productions)],
        "available_in_english" : [bool(available_in_english)]}
    X_input_table = pd.DataFrame.from_dict(X_input)
    model = joblib.load("model.joblib")
    popularity = int(model.predict(X_input_table))
    return {"title" : title,
            "popularity" : popularity}