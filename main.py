from Services.DataPreprocessor import DataPreprocessor
from Services.ModelBuilding import ModelBuilding
from functions import *
import pandas as pd
import matplotlib.pyplot as plt

#pip install numpy==1.23.5 // compatible version for tensorflow pip install tensorflow==2.10.1
def main():
    #remove_empty_rows('data/vmCloud_data_filtered.csv', 'data/vmCloud_data_cleaned.csv')
    df = pd.read_csv('data/cpu_usage_full_period.csv', index_col=0, parse_dates=True)

    data = DataPreprocessor(df, 0.2)
    features = []
    data.set_features(features)
    train_x, train_y = data.get_training_set('cpu')
    test_x, test_y = data.get_testing_set('cpu')
    dataPost = data.get_data_frame()
    
    model_xgboost = ModelBuilding(
           "xgboost",
           train_x,
           train_y,
           test_x,
           test_y,
           data.features,
           'cpu',
           dataPost
       )   

    # model_arima = ModelBuilding(
    #       model_type="arima",
    #       X_train=train_x,  # ARIMA doesn't use features, so pass None
    #       y_train=train_y,
    #       X_test=test_x,   # ARIMA doesn't use features, so pass None
    #       y_test=test_y,
    #       features=None,
    #       target="cpu",
    #       data=dataPost
    #   )
    # model_xgboost = ModelBuilding(
    #        "randomforest",
    #        train_x,
    #        train_y,
    #        test_x,
    #        test_y,
    #        data.features,
    #        'cpu',
    #        dataPost
    #    )   
    # model_xgboost = ModelBuilding(
    #        "svr",
    #        train_x,
    #        train_y,
    #        test_x,
    #        test_y,
    #        data.features,
    #        'cpu',
    #        dataPost
    #   )   
    # model_lstm = ModelBuilding(
    #     "lstm",
    #     train_x,
    #     train_y,
    #     test_x,
    #     test_y,
    #     data.features,
    #     'PJMW_MW',
    #     dataPost
    #  )

    # model_cnn = ModelBuilding(
    #     "cnn",
    #     train_x,
    #     train_y,
    #     test_x,
    #     test_y,
    #     data.features,
    #     'PJMW_MW',
    #     dataPost
    # )

    # model_cnn_lstm = ModelBuilding(
    #  "cnn-lstm",
    #  train_x,
    #  train_y,
    #  test_x,
    #  test_y,
    #  data.features,
    #  'cpu',
    #  dataPost
    #  )



if __name__ == "__main__":
    main()
