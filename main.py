import pandas as pd
from preprocessing.automate_agusswardanaa import preprocess_data

dataset = pd.read_csv("crop_yield.csv")
preprocess_data(dataset, "Yield_tons_per_hectare", "preprocessing/preprocessor_pipeline.joblib", "preprocessing/header.csv")