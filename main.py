from Services.DataPreprocessor import DataPreprocessor
from Services.ModelBuilding import ModelBuilding
from functions import *
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('data/PJMW_hourly.csv', index_col=0, parse_dates=True)

    data = DataPreprocessor(df, 0.2)
    features = ['month','day','hour', 'minute']
    data.set_features(features)
    train_x, train_y = data.get_training_set('PJMW_MW')
    test_x, test_y = data.get_testing_set('PJMW_MW')
    dataPost = data.get_data_frame()
    
    model_xgboost = ModelBuilding(
        "xgboost",
        train_x,
        train_y,
        test_x,
        test_y,
        data.features,
        'PJMW_MW',
        dataPost
    )   
    # model_arima = ModelBuilding(
    #     model_type="arima",
    #     X_train=None,  # ARIMA doesn't use features, so pass None
    #     y_train=train_y,
    #     X_test=None,   # ARIMA doesn't use features, so pass None
    #     y_test=test_y,
    #     features=None,
    #     target="CPU"
    #)

if __name__ == "__main__":
    main()
