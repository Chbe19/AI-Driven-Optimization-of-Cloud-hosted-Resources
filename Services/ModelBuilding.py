import shutil
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# from pmdarima import auto_arima
import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, GRU, BatchNormalization, GlobalAveragePooling1D, TimeDistributed, Input, LayerNormalization, InputLayer, RepeatVector# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
import keras_tuner as kt
from tensorflow.keras.losses import Huber # type: ignore



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
        if model_type.lower() == "xgboost":
            self.model_xgboost()
        elif model_type.lower() == "catboost":
            self.model_catboost()
        elif model_type.lower() == "gru":
            self.train_gru()
        elif model_type.lower() == "arima":
            # ARIMA implementation
            print("Fitting ARIMA model...")
            
            # Check for duplicate indices
            self.y_train = self.y_train.sort_index()
            self.y_test = self.y_test.sort_index()
            self.y_train = self.y_train.groupby(self.y_train.index).mean()
            self.y_test = self.y_test.groupby(self.y_test.index).mean()
            # Assign the hourly frequency
            self.y_train = self.y_train.asfreq('30T').ffill()  # Forward-fill missing values
            self.y_test = self.y_test.asfreq('30T').ffill()  # Forward-fill missing values
            
            scaler = MinMaxScaler()
            self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
            self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)
            
            self.y_test = self.y_test.dropna()
            self.y_train = self.y_train.dropna()  # Drop NaN values         

            # Ensure indices are aligned between y_train and X_train
            self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)

            # Ensure indices are aligned between y_test and X_test (if needed later)
            self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

            # # Check for stationarity
            # plot_acf(self.y_train, lags=50)
            # plt.title("ACF Plot") # P-values
            # plt.show()
            # plot_pacf(self.y_train, lags=50, method='ywm')
            # plt.title("PACF Plot") # Q-values
            # plt.show()
            # plot_acf(self.y_train, lags=240)  # Check for seasonal lags (e.g., multiples of 24)
            # plt.title("Seasonal ACF Plot")
            # plt.show()
            # plot_pacf(self.y_train, lags=240, method='ywm')
            # plt.title("Seasonal PACF Plot")
            # plt.show()
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
            forecast = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()

            #print(self.y_test.index.equals(forecast.index))
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_test.index, y_test_original, label="Actual")
            plt.plot(self.y_test.index, forecast, label="Forecast", linestyle="--")
            plt.title("ARIMA Forecast vs Actual")
            plt.legend()
            plt.show()
            # Calculate MAE
            mae = mean_absolute_error(self.y_test, forecast)
            print(f'MAE Score on Test set: {mae:0.2f}')
            rmse = np.sqrt(mean_squared_error(self.y_test, forecast))
            print(f"RMSE: {rmse}")
        
        elif model_type.lower() == "randomforest":
            self.model_randomforest()
        elif model_type.lower() == "svr":
            self.model_svr()

        elif model_type.lower() == "autoencoder":
            self.train_autoencoder()


        elif model_type.lower() == "lstm":

            self.train_lstm()




            # # Hur många steg som används för att förutsäga nästa, antal träningsomgångar, inlärningshastighet
            # TIME_STEPS = 24
            # EPOCHS = 100
            # LR = 0.01

            # # === 1. Förbered data ===
            # # Skapar dataframe med endast cpu som variabel
            # cpu_train = self.y_train.to_frame(name='cpu')
            # cpu_test = self.y_test.to_frame(name='cpu')

            # # Standadiserar datan för att snabba upp träning
            # scaler = StandardScaler()
            # all_scaled = pd.DataFrame(scaler.fit_transform(pd.concat([cpu_train, cpu_test])),
            #                         index=pd.concat([cpu_train, cpu_test]).index,
            #                         columns=['cpu'])

            # # Delar upp datan i träning / test
            # train_scaled = all_scaled.loc[cpu_train.index]
            # test_scaled = all_scaled.loc[cpu_test.index]

            # # Funkti0onm som skapar sekvenser för lstm med X historiska värden och y nästa värde
            # def create_sequences(data, time_steps=TIME_STEPS):
            #     X, y = [], []
            #     for i in range(len(data) - time_steps):
            #         X.append(data[i:i+time_steps])
            #         y.append(data[i+time_steps])
            #     return np.array(X), np.array(y)

            # # Använder funktionen
            # X_train_seq, y_train_seq = create_sequences(train_scaled.values)
            # X_test_seq, y_test_seq = create_sequences(test_scaled.values)

            # # Ändrar data till PyTorch-tensorer
            # X_train_tensor = torch.from_numpy(X_train_seq).float()
            # y_train_tensor = torch.from_numpy(y_train_seq).float().view(-1, 1)
            # X_test_tensor = torch.from_numpy(X_test_seq).float()
            # y_test_tensor = torch.from_numpy(y_test_seq).float().view(-1, 1)

            # # === 2. Modell ===
            # # Bygger LSTM modellen
            # class LSTMModel(nn.Module):
            #     def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
            #         super(LSTMModel, self).__init__()
            #         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            #         self.fc = nn.Linear(hidden_dim, output_dim)

            #     def forward(self, x):
            #         out, _ = self.lstm(x)
            #         return self.fc(out[:, -1, :])

            # model = LSTMModel()
            # criterion = nn.MSELoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # train_losses, val_losses = [], []

            # # === 3. Träning ===
            # # Träningsloop
            # for epoch in range(1, EPOCHS+1):
            #     model.train()
            #     optimizer.zero_grad()
            #     output = model(X_train_tensor)
            #     loss = criterion(output, y_train_tensor)
            #     loss.backward()
            #     optimizer.step()

            #     model.eval()
            #     with torch.no_grad():
            #         val_output = model(X_test_tensor)
            #         val_loss = criterion(val_output, y_test_tensor)

            #     train_losses.append(loss.item())
            #     val_losses.append(val_loss.item())

            # # === Prediktion och visualisering ===
            # with torch.no_grad():
            #     y_pred_scaled = model(X_test_tensor).cpu().numpy()
            #     y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            # pred_index = self.y_test.index[TIME_STEPS:]
            # y_true = self.y_test.loc[pred_index]

            # # === Result ===
            # result_df = pd.DataFrame({
            #     'Actual': y_true,
            #     'Prediction': y_pred
            # }, index=pred_index)

            # ax = result_df['Actual'].plot(figsize=(15, 5), alpha=0.3, color='skyblue')
            # result_df['Prediction'].plot(ax=ax, style='--', color='orange', linewidth=2)
            # plt.legend(['Actual', 'Predicted'])
            # ax.set_title('LSTM Forecast')
            # plt.grid(True, linestyle='--', alpha=0.3)
            # plt.tight_layout()
            # plt.show()

            # # === Training vs Validation Loss ===
            # plt.figure(figsize=(12, 4))
            # plt.plot(train_losses, label='Training Loss')
            # plt.plot(val_losses, label='Validation Loss')
            # plt.title("Training vs Validation Loss (LSTM)")
            # plt.xlabel("Epoch")
            # plt.ylabel("Loss (MSE)")
            # plt.legend()
            # plt.grid(True, linestyle='--', alpha=0.3)
            # plt.tight_layout()
            # plt.show()

            # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            # mean_actual = result_df['Actual'].mean()
            # std_actual = result_df['Actual'].std()
            # relative_error = (rmse / mean_actual) * 100

            # print(f'RMSE: {rmse:.2f}')
            # print(f'Relative Error: {relative_error:.2f}%')

            # # === Statistik ===
            # test_df = pd.DataFrame({
            #     'actual': y_true,
            #     'prediction': y_pred
            # }, index=pred_index)

            # mean_val = test_df['actual'].mean()
            # std_val = test_df['actual'].std()
            # mae = mean_absolute_error(test_df['actual'], test_df['prediction'])
            # rmse = np.sqrt(mean_squared_error(test_df['actual'], test_df['prediction']))
            # relative_error = (rmse / mean_val) * 100
            # relative_mae = (mae / mean_val) * 100

            # print(f'RMSE Score on Test set: {rmse:.2f}')
            # print(f"Mean of y_test: {mean_val:.2f}")
            # print(f"Standard Deviation of y_test: {std_val:.2f}")
            # print(f"Relative Error (RMSE): {relative_error:.2f}%")
            # print(f"MAE Score on Test set: {mae:.2f}")
            # print(f"Relative Error (MAE): {relative_mae:.2f}%")
    
        elif model_type.lower() == "cnn":
            self.train_cnn()

        elif model_type.lower() == "cnn-lstm":
            self.train_cnn_lstm()

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_model(self):
        """
        Get the trained model.

        Returns:
            object: The trained model instance.
        """
        return self.model

    def model_xgboost(self):
        print("Fitting xgboost model...")

        # Scale y_train
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # Add lagged features and rolling statistics to both X_train and X_test
        for lag in range(1, 3):  # Add lagged features (e.g., lag_1, lag_2)
            self.X_train[f'cpu_lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'cpu_lag_{lag}'] = self.y_test.shift(lag)

        self.X_train['cpu_rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['cpu_rolling_mean_3'] = self.y_test.rolling(window=3).mean()

        self.X_train['cpu_rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['cpu_rolling_std_3'] = self.y_test.rolling(window=3).std()

        # Drop rows with NaN values caused by lagging and rolling operations
        self.X_train = self.X_train.dropna()
        self.X_test = self.X_test.dropna()

        # Align y_train and y_test with X_train and X_test after dropping NaN rows
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        # Tune and train the model
        best_params, self.model = self.tune_xgboost(self.X_train, self.y_train)
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.model.set_params(eval_metric='rmse')
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set= eval_set,
            verbose=100,
        )

        # Try to retrieve evaluation results
        if hasattr(self.model, 'evals_result'):
            evals_result = self.model.evals_result()
            train_loss = evals_result['validation_0']['rmse']
            val_loss = evals_result['validation_1']['rmse']

            # Plot training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss, label='Training RMSE')
            plt.plot(val_loss, label='Validation RMSE')
            plt.xlabel('Boosting Round')
            plt.ylabel('RMSE')
            plt.title('XGBoost Training and Validation Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("evals_result is not available in this XGBoost version.")
        
        # Feature importance
        fi = pd.DataFrame(data=self.model.feature_importances_,
                        index=self.model.feature_names_in_,
                        columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')

        # Create a DataFrame for the test set
        test = pd.DataFrame(self.X_test.copy())
        test[self.target] = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()  # Inverse transform y_test
        test['prediction'] = scaler.inverse_transform(self.model.predict(self.X_test).reshape(-1, 1)).flatten()  # Inverse transform predictions
        self.data = self.data.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)  # Drop rows with NaN values
        
        # Plot the results
        ax = test[[self.target]].plot(figsize=(15, 5))
        test['prediction'].plot(ax=ax, style='--')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('XGBOOST: Raw Data and Prediction')
        plt.show()

        # Calculate MAE
        mae = mean_absolute_error(test[self.target], test['prediction'])
        print(f'MAE Score on Test set: {mae:0.2f}')
        # Calculate RMSE
        score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')
        # Calculate R^2 Score
        r_square = r2_score(test[self.target], test['prediction'])
        print(f'R^2 Score on Test set: {r_square:0.2f}')
        print("Mean of y_test:", test[self.target].mean())  # Use inverse-transformed y_test
        print("Standard Deviation of y_test:", test[self.target].std())  # Use inverse-transformed y_test
        print("Mean of y_train:", self.y_train.mean())  # Use inverse-transformed y_test
        print("Standard Deviation of y_train:", self.y_train.std())  # Use inverse-transformed y_test
        relative_error = (score / test[self.target].mean()) * 100
        print(f"Relative Error(RMSE): {relative_error:.2f}%")
        relative_error_mae = (mae / test[self.target].mean()) * 100
        print(f"Relative Error(MAE): {relative_error_mae:.2f}%")        

    def tune_xgboost(self, X_train, y_train):
        """
        Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            dict: Best parameters from the search.
            xgb.XGBRegressor: Best XGBoost model.
        """
        # Define the parameter grid
        param_grid = {
            'n_estimators': [500, 1000, 1500, 2000],
            'max_depth': [3, 5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.6, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [1.0, 1.5, 2.0, 3.0],
        }

        kf = KFold(n_splits=4, shuffle=True, random_state=0)

        # Initialize the XGBoost regressor
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, booster='gbtree',)

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,  # Number of random combinations to try
            scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
            cv=kf,  # 3-fold cross-validation
            verbose=1,  # Print progress
            n_jobs=-1,  # Use all available CPU cores
            random_state=42
        )

        # Fit the random search to the data
        random_search.fit(X_train, y_train)

        # Get the best parameters and the best model
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        print("Best Parameters:", best_params)
        print("Best RMSE (negative MSE):", np.sqrt(-random_search.best_score_))

        return best_params, best_model

    def model_catboost(self):
        print("Fitting CatBoost model...")

        # Scale y_train
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # Feature engineering (reuse your lag/rolling code if needed)
        for lag in range(1, 3):
            self.X_train[f'cpu_lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'cpu_lag_{lag}'] = self.y_test.shift(lag)
        self.X_train['cpu_rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['cpu_rolling_mean_3'] = self.y_test.rolling(window=3).mean()
        self.X_train['cpu_rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['cpu_rolling_std_3'] = self.y_test.rolling(window=3).std()

        self.X_train = self.X_train.dropna()
        self.X_test = self.X_test.dropna()
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        best_params, self.model = self.tune_catboost(self.X_train, self.y_train)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_test, self.y_test),
            use_best_model=True
        )

        # Plot training and validation loss
        if hasattr(self.model, 'get_evals_result'):
            evals_result = self.model.get_evals_result()
            train_loss = evals_result['learn']['RMSE']
            val_loss = evals_result['validation']['RMSE']
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss, label='Training RMSE')
            plt.plot(val_loss, label='Validation RMSE')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('CatBoost Training and Validation Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Create a DataFrame for the test set
        test = pd.DataFrame(self.X_test.copy())
        test[self.target] = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()  # Inverse transform y_test
        test['prediction'] = scaler.inverse_transform(self.model.predict(self.X_test).reshape(-1, 1)).flatten()  # Inverse transform predictions
        self.data = self.data.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)  # Drop rows with NaN values
        
        # Plot the results
        ax = test[[self.target]].plot(figsize=(15, 5))
        test['prediction'].plot(ax=ax, style='--')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('XGBOOST: Raw Data and Prediction')
        plt.show()

        # Calculate MAE
        mae = mean_absolute_error(test[self.target], test['prediction'])
        print(f'MAE Score on Test set: {mae:0.2f}')
        # Calculate RMSE
        score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')
        # Calculate R^2 Score
        r_square = r2_score(test[self.target], test['prediction'])
        print(f'R^2 Score on Test set: {r_square:0.2f}')
        print("Mean of y_test:", test[self.target].mean())  # Use inverse-transformed y_test
        print("Standard Deviation of y_test:", test[self.target].std())  # Use inverse-transformed y_test
        print("Mean of y_train:", self.y_train.mean())  # Use inverse-transformed y_test
        print("Standard Deviation of y_train:", self.y_train.std())  # Use inverse-transformed y_test
        relative_error = (score / test[self.target].mean()) * 100
        print(f"Relative Error(RMSE): {relative_error:.2f}%")
        relative_error_mae = (mae / test[self.target].mean()) * 100
        print(f"Relative Error(MAE): {relative_error_mae:.2f}%")  

    def tune_catboost(self, X_train, y_train):
        """
        Perform hyperparameter tuning for CatBoostRegressor using RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            dict: Best parameters from the search.
            CatBoostRegressor: Best CatBoost model.
        """
        param_grid = {
            'iterations': [200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'bagging_temperature': [0, 1, 2, 5],
            'border_count': [32, 64, 128]
        }

        cat_model = CatBoostRegressor(loss_function='RMSE', verbose=0, random_state=42)
        kf = KFold(n_splits=4, shuffle=True, random_state=0)

        random_search = RandomizedSearchCV(
            estimator=cat_model,
            param_distributions=param_grid,
            n_iter=20,
            scoring='neg_mean_squared_error',
            cv=kf,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        print("Best CatBoost Parameters:", best_params)
        print("Best CatBoost RMSE (negative MSE):", np.sqrt(-random_search.best_score_))

        return best_params, best_model

    def model_randomforest(self):
        print("Fitting Random Forest model...")

        # Scale y_train
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # Add lagged features and rolling statistics to both X_train and X_test
        for lag in range(1, 3):  # Add lagged features (e.g., lag_1, lag_2)
            self.X_train[f'cpu_lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'cpu_lag_{lag}'] = self.y_test.shift(lag)

        self.X_train['cpu_rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['cpu_rolling_mean_3'] = self.y_test.rolling(window=3).mean()

        self.X_train['cpu_rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['cpu_rolling_std_3'] = self.y_test.rolling(window=3).std()

        # Drop rows with NaN values caused by lagging and rolling operations
        self.X_train = self.X_train.dropna()
        self.X_test = self.X_test.dropna()

        # Align y_train and y_test with X_train and X_test after dropping NaN rows
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        # Tune and train the model
        best_params, self.model = self.tune_randomforest(self.X_train, self.y_train)
        self.model.fit(self.X_train, self.y_train)

         # Calculate training and validation RMSE
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        train_mse = mean_squared_error(scaler.inverse_transform(self.y_train.values.reshape(-1, 1)), scaler.inverse_transform(y_train_pred.reshape(-1, 1)))
        val_mse = mean_squared_error(scaler.inverse_transform(self.y_test.values.reshape(-1, 1)), scaler.inverse_transform(y_test_pred.reshape(-1, 1)))

        # Bar plot for RMSE
        plt.figure(figsize=(6, 4))
        plt.bar(['Train RMSE', 'Validation RMSE'], [train_mse, val_mse], color=['blue', 'orange'])
        plt.title('Random Forest RMSE')
        plt.ylabel('RMSE')
        plt.show()


        # Feature Importance
        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        importance_df.plot(kind='barh', x='Feature', y='Importance', title='Feature Importance (Random Forest)', figsize=(10, 6))
        plt.show()

        # Predict
        y_pred_scaled = self.model.predict(self.X_test)

        # Ensure predictions are reshaped correctly for inverse transformation
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

        # Create a DataFrame for the test set
        test = pd.DataFrame(self.X_test.copy())
        test[self.target] = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()  # Inverse transform y_test
        test['prediction'] = y_pred
        self.data = self.data.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)  # Drop rows with NaN values

        # Plot the results
        ax = test[[self.target]].plot(figsize=(15, 5))
        test['prediction'].plot(ax=ax, style='--')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('Random Forest: Raw Data and Prediction')
        plt.show()

        # Calculate MAE
        mae = mean_absolute_error(test[self.target], test['prediction'])
        print(f'MAE Score on Test set: {mae:0.2f}')
        # Calculate RMSE
        score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')
        # Calculate R^2 Score
        r_square = r2_score(test[self.target], test['prediction'])
        print(f'R^2 Score on Test set: {r_square:0.2f}')
        print("Mean of y_test:", test[self.target].mean())
        print("Standard Deviation of y_test:", test[self.target].std())
        relative_error = (score / test[self.target].mean()) * 100
        print(f"Relative Error (RMSE): {relative_error:.2f}%")
        relative_error_mae = (mae / test[self.target].mean()) * 100
        print(f"Relative Error(MAE): {relative_error_mae:.2f}%")

    def tune_randomforest(self, X_train, y_train):
        """
        Perform hyperparameter tuning for RandomForestRegressor using RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            dict: Best parameters from the search.
            RandomForestRegressor: Best Random Forest model.
        """
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30, 50],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
            'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
        }

        # Initialize the Random Forest regressor
        rf_model = RandomForestRegressor(random_state=42)

        kf = KFold(n_splits=4, shuffle=True, random_state=0)

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=50,  # Number of random combinations to try
            scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
            cv=kf,  # 3-fold cross-validation
            verbose=1,  # Print progress
            n_jobs=-1,  # Use all available CPU cores
            random_state=42
        )

        # Fit the random search to the data
        random_search.fit(X_train, y_train)

        # Get the best parameters and the best model
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        print("Best Parameters:", best_params)
        print("Best RMSE (negative MSE):", np.sqrt(-random_search.best_score_))

        return best_params, best_model

    def model_svr(self):
        print("Fitting Support Vector Regression (SVR) model...")

        # Scale y_train
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # Add lagged features and rolling statistics to both X_train and X_test
        for lag in range(1, 3):  # Add lagged features (e.g., lag_1, lag_2)
            self.X_train[f'cpu_lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'cpu_lag_{lag}'] = self.y_test.shift(lag)

        self.X_train['cpu_rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['cpu_rolling_mean_3'] = self.y_test.rolling(window=3).mean()

        self.X_train['cpu_rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['cpu_rolling_std_3'] = self.y_test.rolling(window=3).std()

        # Drop rows with NaN values caused by lagging and rolling operations
        self.X_train = self.X_train.dropna()
        self.X_test = self.X_test.dropna()

        # Align y_train and y_test with X_train and X_test after dropping NaN rows
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        # Tune and train the model
        best_params, self.model = self.tune_svr(self.X_train, self.y_train)
        self.model.fit(self.X_train, self.y_train)

        # Calculate training and validation RMSE
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        train_mse = mean_squared_error(scaler.inverse_transform(self.y_train.values.reshape(-1, 1)), scaler.inverse_transform(y_train_pred.reshape(-1, 1)))
        val_mse = mean_squared_error(scaler.inverse_transform(self.y_test.values.reshape(-1, 1)), scaler.inverse_transform(y_test_pred.reshape(-1, 1)))

        # Bar plot for RMSE
        plt.figure(figsize=(6, 4))
        plt.bar(['Train RMSE', 'Validation RMSE'], [train_mse, val_mse], color=['blue', 'orange'])
        plt.title('Random Forest RMSE')
        plt.ylabel('RMSE')
        plt.show()


        # Predict
        y_pred_scaled = self.model.predict(self.X_test)

        # Ensure predictions are reshaped correctly for inverse transformation
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

        # Create a DataFrame for the test set
        test = pd.DataFrame(self.X_test.copy())
        test[self.target] = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()  # Inverse transform y_test
        test['prediction'] = y_pred
        self.data = self.data.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)  # Drop rows with NaN values

        # Plot the results
        ax = test[[self.target]].plot(figsize=(15, 5))
        test['prediction'].plot(ax=ax, style='--')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('SVR: Raw Data and Prediction')
        plt.show()

         # === 9. Träning vs Validerings-loss ===
        plt.figure(figsize=(12, 5))
        plt.plot(['loss'], label='Training Loss')
        plt.plot(['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (CNN-LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Calculate MAE
        mae = mean_absolute_error(test[self.target], test['prediction'])
        print(f'MAE Score on Test set: {mae:0.2f}')
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {rmse:0.2f}')
        # Calculate R^2 Score
        r_square = r2_score(test[self.target], test['prediction'])
        print(f'R^2 Score on Test set: {r_square:0.2f}')
        print("Mean of y_test:", test[self.target].mean())
        print("Standard Deviation of y_test:", test[self.target].std())
        relative_error = (rmse / test[self.target].mean()) * 100
        print(f"Relative Error (RMSE): {relative_error:.2f}%")
        relative_error_mae = (mae / test[self.target].mean()) * 100
        print(f"Relative Error(MAE): {relative_error_mae:.2f}%")

    def tune_svr(self, X_train, y_train):
        """
        Perform hyperparameter tuning for SVR using RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            dict: Best parameters from the search.
            SVR: Best SVR model.
        """
        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'epsilon': [0.01, 0.1, 0.2],  # Epsilon in the epsilon-SVR model
            'kernel': ['linear'],  # Kernel type ['linear', 'poly', 'rbf', 'sigmoid']
            'gamma': ['scale', 'auto'],  # Kernel coefficient
        }

        kf = KFold(n_splits=4, shuffle=True, random_state=0)

        # Initialize the SVR model
        svr_model = SVR()

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=svr_model,
            param_distributions=param_grid,
            n_iter=50,  # Number of random combinations to try
            scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
            cv=kf,  # 4-fold cross-validation
            verbose=1,  # Print progress
            n_jobs=-1,  # Use all available CPU cores
            random_state=42
        )

        # Fit the random search to the data
        random_search.fit(X_train, y_train)

        # Get the best parameters and the best model
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        print("Best Parameters:", best_params)
        print("Best RMSE (negative MSE):", np.sqrt(-random_search.best_score_))

        return best_params, best_model
    
    def train_cnn(self):
        # === 1. Förbered data ===
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # === 2. Laggade features + rullande statistik ===
        for lag in range(1, 4):
            self.X_train[f'lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'lag_{lag}'] = self.y_test.shift(lag)

        self.X_train['rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['rolling_mean_3'] = self.y_test.rolling(window=3).mean()
        self.X_train['rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['rolling_std_3'] = self.y_test.rolling(window=3).std()

        # === 3. Ta bort NaN och align ===
        self.X_train = self.X_train.dropna()
        self.X_test = self.X_test.dropna()
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        # === 4. Forma om till CNN-ingång ===
        X_train_cnn = self.X_train.values.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test_cnn = self.X_test.values.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        # === 5. Modell ===
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_cnn,
            self.y_train.values,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        # === 6. Förutsäg och skala tillbaka ===
        y_pred_scaled = model.predict(X_test_cnn).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()

        # === 7. Lägg till prediction i DataFrame ===
        test_df = pd.DataFrame(self.X_test.copy())
        test_df[self.target] = y_actual
        test_df['prediction'] = y_pred
        self.data = self.data.merge(test_df[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)

        # === 8. Plot ===
        ax = test_df[[self.target]].plot(figsize=(16, 6), alpha=0.3, label="Actual")
        test_df['prediction'].plot(ax=ax, color="orange", label="Predicted")
        plt.legend()
        plt.title("CNN")
        plt.show()

        # === 9. Loss-plot ===
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === 10. Histogram för fördelning ===
        plt.figure(figsize=(6, 5))
        plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted')
        plt.hist(y_actual, bins=50, alpha=0.7, label='Actual')
        plt.title("Distribution of predictions")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === 11. Metrics ===
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r_score = r2_score(y_actual, y_pred)
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R^2 Score: {r_score:.2f}")
        print(f"Relative Error (RMSE): {rmse / np.mean(y_actual) * 100:.2f}%")
        print(f"Relative Error (MAE): {mae / np.mean(y_actual) * 100:.2f}%")
        print(f"Mean: {np.mean(y_actual):.2f}")
        print(f"Std: {np.std(y_actual):.2f}")



    def train_cnn_lstm(self):
        # === 1. Förbered data ===
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # === 2. Laggade features + rullande statistik ===
        for lag in range(1, 4):
            self.X_train[f'lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'lag_{lag}'] = self.y_test.shift(lag)

        self.X_train['rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['rolling_mean_3'] = self.y_test.rolling(window=3).mean()
        self.X_train['rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['rolling_std_3'] = self.y_test.rolling(window=3).std()

        # === 3. Ta bort NaN och align ===
        self.X_train.dropna(inplace=True)
        self.X_test.dropna(inplace=True)
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        # === 4. CNN-LSTM kräver 4D: (samples, subseq, timesteps, features) ===
        total_features = self.X_train.shape[1]
        for subseq in range(2, total_features + 1):
            if total_features % subseq == 0:
                timesteps = total_features // subseq
                break
        else:
            raise ValueError("Ingen giltig subsequence/timestep-kombination hittades.")

        print(f"Using subsequences={subseq}, timesteps={timesteps}")

        X_train_seq = self.X_train.values.reshape((self.X_train.shape[0], subseq, timesteps, 1))
        X_test_seq = self.X_test.values.reshape((self.X_test.shape[0], subseq, timesteps, 1))

        # === 5. Modell ===
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(subseq, timesteps, 1)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_seq,
            self.y_train.values,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        # === 6. Prediktion och omvänd skalning ===
        y_pred_scaled = model.predict(X_test_seq).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()

        # === 7. Lägg till prediction i DataFrame ===
        test_df = pd.DataFrame(self.X_test.copy())
        test_df[self.target] = y_actual
        test_df['prediction'] = y_pred
        self.data = self.data.merge(test_df[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)

        # === 8. Plot ===
        ax = test_df[[self.target]].plot(figsize=(16, 6), alpha=0.3, label="Actual")
        test_df['prediction'].plot(ax=ax, color="orange", label="Predicted")
        plt.legend()
        plt.title("CNN-LSTM")
        plt.show()

        # === 9. Loss-plot ===
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (CNN-LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === 10. Histogram ===
        plt.figure(figsize=(6, 5))
        plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted')
        plt.hist(y_actual, bins=50, alpha=0.7, label='Actual')
        plt.title("Distribution of predictions (CNN-LSTM)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === 11. Utvärdering ===
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r_score = r2_score(y_actual, y_pred)
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R^2 Score: {r_score:.2f}")
        print(f"Relative Error (RMSE): {rmse / np.mean(y_actual) * 100:.2f}%")
        print(f"Relative Error (MAE): {mae / np.mean(y_actual) * 100:.2f}%")
        print(f"Mean: {np.mean(y_actual):.2f}")
        print(f"Std: {np.std(y_actual):.2f}")


    def train_lstm(self):
        print("Training improved LSTM model...")

       # === 1. Skala y ===
        scaler = MinMaxScaler()
        self.y_train = pd.Series(scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(), index=self.y_train.index)
        self.y_test = pd.Series(scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(), index=self.y_test.index)

        # === 2. Laggade features + rullande statistik ===
        for lag in range(1, 4):
            self.X_train[f'lag_{lag}'] = self.y_train.shift(lag)
            self.X_test[f'lag_{lag}'] = self.y_test.shift(lag)

        self.X_train['rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        self.X_test['rolling_mean_3'] = self.y_test.rolling(window=3).mean()
        self.X_train['rolling_std_3'] = self.y_train.rolling(window=3).std()
        self.X_test['rolling_std_3'] = self.y_test.rolling(window=3).std()

        # === 3. Ta bort NaN och align ===
        self.X_train.dropna(inplace=True)
        self.X_test.dropna(inplace=True)
        self.X_train, self.y_train = self.X_train.align(self.y_train, join='inner', axis=0)
        self.X_test, self.y_test = self.X_test.align(self.y_test, join='inner', axis=0)

        # === 4. LSTM kräver 3D: (samples, timesteps, features) ===
        X_train_seq = self.X_train.values.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        X_test_seq = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

        # === 5. Modell ===
        model = Sequential()
        model.add(LSTM(64, activation='tanh', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_seq,
            self.y_train.values,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        # === 6. Prediktion och omvänd skalning ===
        y_pred_scaled = model.predict(X_test_seq).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()

        # === 7. Lägg till prediction i DataFrame ===
        test_df = pd.DataFrame(self.X_test.copy())
        test_df[self.target] = y_actual
        test_df['prediction'] = y_pred
        self.data = self.data.merge(test_df[['prediction']], how='left', left_index=True, right_index=True)
        self.data.dropna(inplace=True)

        # === 8. Plot ===
        ax = test_df[[self.target]].plot(figsize=(16, 6), alpha=0.3, label="Actual")
        test_df['prediction'].plot(ax=ax, color="orange", label="Predicted")
        plt.legend()
        plt.title("Improved LSTM Forecast")
        plt.show()

        # === 9. Loss-plot ===
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (Improved LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === 10. Histogram ===
        plt.figure(figsize=(6, 5))
        plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted')
        plt.hist(y_actual, bins=50, alpha=0.7, label='Actual')
        plt.title("Distribution of predictions (LSTM)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === 11. Utvärdering ===
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r_score = r2_score(y_actual, y_pred)
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R^2 Score: {r_score:.2f}")
        print(f"Relative Error (RMSE): {rmse / np.mean(y_actual) * 100:.2f}%")
        print(f"Relative Error (MAE): {mae / np.mean(y_actual) * 100:.2f}%")
        print(f"Mean: {np.mean(y_actual):.2f}")
        print(f"Std: {np.std(y_actual):.2f}")


    def train_autoencoder(self):
        print("Training Autoencoder model...")

        # === 1. Skapa och skala data ===
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(self.data[[self.target]].values)
        self.y_train = pd.Series(y_scaled[:int(len(y_scaled) * 0.8)].flatten(), index=self.data.index[:int(len(y_scaled) * 0.8)])
        self.y_test = pd.Series(y_scaled[int(len(y_scaled) * 0.8):].flatten(), index=self.data.index[int(len(y_scaled) * 0.8):])

        # === 2. Skapa sekvenser ===
        def create_sequences(series, window_size):
            X = []
            for i in range(len(series) - window_size):
                seq = series[i:i + window_size]
                X.append(seq)
            return np.array(X)

        SEQ_LEN = 10
        X_train_seq = create_sequences(self.y_train.values, SEQ_LEN)
        X_test_seq = create_sequences(self.y_test.values, SEQ_LEN)

        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], SEQ_LEN, 1))
        X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], SEQ_LEN, 1))

        # === 3. Modell (Autoencoder) ===
        input_dim = X_train_seq.shape[1:]
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_dim, return_sequences=False),
            RepeatVector(SEQ_LEN),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1))
        ])

        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # === 4. Träning ===
        history = model.fit(
            X_train_seq,
            X_train_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        # === 5. Prediktion ===
        X_test_pred = model.predict(X_test_seq)
        X_test_pred_flat = X_test_pred[:, -1, 0]  # ta sista steget som prediktion
        X_test_true_flat = X_test_seq[:, -1, 0]

        # === 6. Omvandla tillbaka
        y_pred = scaler.inverse_transform(X_test_pred_flat.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(X_test_true_flat.reshape(-1, 1)).flatten()

        # === 7. Visualisering av resultat
        index_start = self.y_test.index[-len(y_actual):]
        test_df = pd.DataFrame({'Actual': y_actual, 'Prediction': y_pred}, index=index_start)

        ax = test_df['Actual'].plot(figsize=(16, 6), alpha=0.3, label="Actual")
        test_df['Prediction'].plot(ax=ax, color="orange", label="Predicted")
        plt.legend()
        plt.title("Autoencoder Forecast")
        plt.tight_layout()
        plt.show()

        # === 8. Förlustkurva
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (Autoencoder)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === 9. Histogram
        plt.figure(figsize=(6, 5))
        plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted')
        plt.hist(y_actual, bins=50, alpha=0.7, label='Actual')
        plt.title("Distribution of predictions (Autoencoder)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === 10. Utvärdering
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r_score = r2_score(y_actual, y_pred)
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R^2 Score: {r_score:.2f}")
        print(f"Relative Error (RMSE): {rmse / np.mean(y_actual) * 100:.2f}%")
        print(f"Relative Error (MAE): {mae / np.mean(y_actual) * 100:.2f}%")
        print(f"Mean: {np.mean(y_actual):.2f}")
        print(f"Std: {np.std(y_actual):.2f}")

    def train_gru(self):
        print("Tuning GRU model with KerasTuner...")
        TIME_STEPS = 48  # 1 day for 30-min data
        EPOCHS = 30
        BATCH_SIZE = 32
        tuner_dir = 'gru_tuning'
        if os.path.exists(tuner_dir):
            shutil.rmtree(tuner_dir)
        # Feature engineering (same as before)
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()

        for lag in range(1, 3):
            X_train[f'cpu_lag_{lag}'] = y_train.shift(lag)
            X_test[f'cpu_lag_{lag}'] = y_test.shift(lag)
        X_train['cpu_rolling_mean_3'] = y_train.rolling(window=3).mean()
        X_test['cpu_rolling_mean_3'] = y_test.rolling(window=3).mean()
        X_train['cpu_rolling_std_3'] = y_train.rolling(window=3).std()
        X_test['cpu_rolling_std_3'] = y_test.rolling(window=3).std()

        # Drop NaNs from X and y together
        train_df = X_train.copy()
        train_df['y'] = y_train
        train_df = train_df.dropna()
        X_train = train_df.drop(columns=['y'])
        y_train = train_df['y']

        test_df = X_test.copy()
        test_df['y'] = y_test
        test_df = test_df.dropna()
        X_test = test_df.drop(columns=['y'])
        y_test = test_df['y']

        # Scale target
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # Create sequences
        def create_sequences(X, y, time_steps):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X.iloc[i:i + time_steps].values)
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, pd.Series(y_train_scaled, index=y_train.index), TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test, pd.Series(y_test_scaled, index=y_test.index), TIME_STEPS)

        # Time-based validation split
        split = int(len(X_train_seq) * 0.8)
        X_train_sub, X_val = X_train_seq[:split], X_train_seq[split:]
        y_train_sub, y_val = y_train_seq[:split], y_train_seq[split:]

        model = Sequential([
            GRU(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                input_shape=(TIME_STEPS, X_train_seq.shape[2]), return_sequences=True),
            GRU(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(
            X_train_sub, y_train_sub,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )

        self.model = model
        # Predict
        y_pred_scaled = self.model.predict(X_test_seq)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        test_index = y_test.index[TIME_STEPS:]

        # Plot
        plt.figure(figsize=(15, 5))
        plt.plot(test_index, y_test.iloc[TIME_STEPS:], label='Actual')
        plt.plot(test_index, y_pred.flatten(), label='Predicted')
        plt.legend()
        plt.title('Tuned GRU Prediction vs Actual')
        plt.show()

        # Plot training and validation loss
        
        if hasattr(history, 'history'):
            plt.figure(figsize=(12, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title("Training vs Validation Loss (Tuned GRU)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test.iloc[TIME_STEPS:], y_pred.flatten()))
        mae = mean_absolute_error(y_test.iloc[TIME_STEPS:], y_pred.flatten())
        r_score = r2_score(y_test.iloc[TIME_STEPS:], y_pred.flatten())
        mean_actual = y_test.iloc[TIME_STEPS:].mean()
        relative_error_rmse = (rmse / mean_actual) * 100
        relative_error_mae = (mae / mean_actual) * 100
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R^2 Score: {r_score:.2f}")
        print(f"Relative Error (RMSE): {relative_error_rmse:.2f}%")
        print(f"Relative Error (MAE): {relative_error_mae:.2f}%")
        