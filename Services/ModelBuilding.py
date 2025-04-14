from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ModelBuilding:
    def __init__(self, model_type, X_train, y_train, X_test, y_test, features, target, data=None):
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
        self.data = data
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
            print("Fitting xgboost model...")
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
           
            # Create a DataFrame for the test set
            test = pd.DataFrame(self.X_test.copy())
            test[self.target] = self.y_test  # Add the actual target values to the test DataFrame
            print(test)
            #print(train)

            test['prediction'] = self.model.predict(self.X_test)

            self.data = self.data.merge(test[['prediction']], how='left', left_index=True, right_index=True)
            
            # Plot the results
            ax = test[[self.target]].plot(figsize=(15, 5))
            self.data['prediction'].plot(ax=ax, style='.')
            plt.legend(['Truth Data', 'Predictions'])
            ax.set_title('Raw Data and Prediction')
            plt.show()
            
            
            score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
            print(f'RMSE Score on Test set: {score:0.2f}')

        elif model_type.lower() == "arima":
            # ARIMA implementation
            print("Fitting ARIMA model...")
            self.model = ARIMA(self.y_train, order=(5, 1, 0))  # Example order (p=5, d=1, q=0)
            self.model = self.model.fit()
            
            # Print ARIMA summary
            print(self.model.summary())

            # Forecast on the test set
            #.predict() ?
            forecast = self.model.forecast(steps=len(self.y_test))
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_test.index, self.y_test, label="Actual")
            plt.plot(self.y_test.index, forecast, label="Forecast", linestyle="--")
            plt.title("ARIMA Forecast vs Actual")
            plt.legend()
            plt.show()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_model(self):
        """
        Get the trained model.

        Returns:
            object: The trained model instance.
        """
        return self.model
    