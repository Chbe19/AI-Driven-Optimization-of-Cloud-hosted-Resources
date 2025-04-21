from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
from sklearn.preprocessing import MinMaxScaler


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
                n_estimators=1500,
                max_depth=5,
                learning_rate=0.01,
                colsample_bytree=1.0,
                subsample=0.8,
                reg_alpha=0.5,
                reg_lambda=1.5,
                gamma=0.1,
                
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
            test['prediction'] = self.model.predict(self.X_test)

            self.data = self.data.merge(test[['prediction']], how='left', left_index=True, right_index=True)
            
            # Plot the results
            ax = test[[self.target]].plot(figsize=(15, 5))
            self.data['prediction'].plot(ax=ax, style='.')
            plt.legend(['Truth Data', 'Predictions'])
            ax.set_title('Raw Data and Prediction')
            plt.show()
            
            
            score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
            print(f'RMSE Score on Test set: {score:0.2f}') # 2.33% average error
            print("Mean of y_test:", self.y_test.mean())
            print("Standard Deviation of y_test:", self.y_test.std())
 
        elif model_type.lower() == "arima":
            # ARIMA implementation
            print("Fitting ARIMA model...")
            
            self.y_train = self.y_train.loc[self.y_train.index >= (self.y_train.index[-1] - pd.DateOffset(years=1))]
            self.y_test = self.y_test.loc[self.y_test.index >= (self.y_test.index[-1] - pd.DateOffset(years=1))]
            #------------------------------------------------------------------ SKA TAS BORT MED NÄR VI FÅTT RIKTIGT DATA 
            # Check for duplicate indices
            self.y_train = self.y_train.sort_index()
            self.y_test = self.y_test.sort_index()
            self.y_train = self.y_train.groupby(self.y_train.index).mean()
            self.y_test = self.y_test.groupby(self.y_test.index).mean()
            # Assign the hourly frequency
            self.y_train = self.y_train.asfreq('h', method='ffill')  # Forward-fill missing values
            self.y_test = self.y_test.asfreq('h', method='ffill')
            
            scaler = MinMaxScaler()
            self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
            self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)
            
            self.y_test = self.y_test.dropna()
            self.y_train = self.y_train.dropna()  # Drop NaN values         

            # Check for stationarity
            plot_acf(self.y_train, lags=50)
            plt.title("ACF Plot") # P-values
            plt.show()
            plot_pacf(self.y_train, lags=50, method='ywm')
            plt.title("PACF Plot") # Q-values
            plt.show()
            plot_acf(self.y_train, lags=240)  # Check for seasonal lags (e.g., multiples of 24)
            plt.title("Seasonal ACF Plot")
            plt.show()
            plot_pacf(self.y_train, lags=240, method='ywm')
            plt.title("Seasonal PACF Plot")
            plt.show()
            #------------------------------------------------------
            start_time = time.time()
            self.model = SARIMAX(self.y_train, exog=self.X_train ,order=(2, 0, 7), seasonal_order=(1, 1, 2, 24))  # Example order (p=5, d=0, q=2) season (P,D=1,Q,s)
            self.model = self.model.fit()
            end_time = time.time()
            print(f"Model fitting took {end_time - start_time:.2f} seconds.")


            # Print ARIMA summary
            print(self.model.summary())

            # Forecast on the test set
            #.predict() ?
            forecast = self.model.forecast(steps=len(self.y_test), exog=self.X_test)
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_test.index, self.y_test, label="Actual")
            plt.plot(self.y_test.index, forecast, label="Forecast", linestyle="*")
            plt.title("ARIMA Forecast vs Actual")
            plt.legend()
            plt.show()
            rmse = np.sqrt(mean_squared_error(self.y_test, forecast))
            print(f"RMSE: {rmse}")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_model(self):
        """
        Get the trained model.

        Returns:
            object: The trained model instance.
        """
        return self.model
    