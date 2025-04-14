import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

class ModelBuilding:
    def __init__(self, model_type, X_train, y_train, X_test, y_test, features, target):
        """ Initialize the ModelBuilding class.
        """
        self.model_type = model_type
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.target = target
        self.initialize_model(model_type)

    def initialize_model(self, model_type):
        """
        Initialize the model based on the provided model type.

        Args:
            model_type (str): The type of the model (e.g., 'xgboost', 'arima').

        Returns:
            object: An instance of the corresponding model class.
        """
        if model_type.lower() == "xgboost":
            self.model = xgb.XGBRegressor(
                base_score=0.5,
                booster='gbtree',
                early_stopping_rounds=50,
                objective='reg:linear',
                n_estimators=1000,
                max_depth=3,
                learning_rate=0.01,

            )
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                verbose=100
            )
            fi = pd.DataFrame(data=self.model.feature_importances_,
                            index=self.model.feature_names_in_,
                            columns=['importance'])
            fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
            plt.show()
        elif model_type.lower() == "arima":
            
            self.model = "ARIMA"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

