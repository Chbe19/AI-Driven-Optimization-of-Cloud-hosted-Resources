from Services.DataPreprocessor import DataPreprocessor
from Services.ModelBuilding import ModelBuilding
from functions import *
from openai import OpenAI
from google import genai
import pandas as pd
import config

def main():
    df = pd.read_csv('data/vm_cpu_data.csv', index_col=0, parse_dates=True)
    
    data = DataPreprocessor(df, 0.2)
    features = ['hour','minute']
    data.set_features(features)
    train_x, train_y = data.get_training_set("CPU")
    test_x, test_y = data.get_testing_set("CPU")

    model_xgboost = ModelBuilding(
        "xgboost",
        train_x,
        train_y,
        test_x,
        test_y,
        data.features,
        "CPU"
    )

if __name__ == "__main__":
    main()
