from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping



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

            # Check for stationarity
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
            self.model = auto_arima(
                self.y_train,
                seasonal=True,
                m=48,  # Seasonal period (e.g., 48 for half-hourly data with daily seasonality)
                trace=True,  # Print the model selection process
                error_action='ignore',  # Ignore errors and continue
                suppress_warnings=True,  # Suppress warnings
                stepwise=True,  # Use stepwise search to reduce computation time
            )
            end_time = time.time()
            print(f"Model fitting took {end_time - start_time:.2f} seconds.")


            # Print ARIMA summary
            print(self.model.summary())

            # Forecast on the test set
            #.predict() ?
            forecast = self.model.predict(n_periods=len(self.y_test))

            forecast = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).flatten()

            #print(self.y_test.index.equals(forecast.index))
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_test.index, y_test_original, label="Actual")
            plt.plot(self.y_test.index, forecast, label="Forecast", linestyle="--")
            plt.title("ARIMA Forecast vs Actual")
            plt.legend()
            plt.show()
            rmse = np.sqrt(mean_squared_error(self.y_test, forecast))
            print(f"RMSE: {rmse}")
        
        elif model_type.lower() == "randomforest":
            self.model_randomforest()
        elif model_type.lower() == "svr":
            self.model_svr()


        elif model_type.lower() == "lstm":
            print("LSTM...")

            # Hur många steg som används för att förutsäga nästa, antal träningsomgångar, inlärningshastighet
            TIME_STEPS = 24
            EPOCHS = 100
            LR = 0.01

            # === 1. Förbered data ===
            # Skapar dataframe med endast cpu som variabel
            cpu_train = self.y_train.to_frame(name='cpu')
            cpu_test = self.y_test.to_frame(name='cpu')

            # Standadiserar datan för att snabba upp träning
            scaler = StandardScaler()
            all_scaled = pd.DataFrame(scaler.fit_transform(pd.concat([cpu_train, cpu_test])),
                                    index=pd.concat([cpu_train, cpu_test]).index,
                                    columns=['cpu'])

            # Delar upp datan i träning / test
            train_scaled = all_scaled.loc[cpu_train.index]
            test_scaled = all_scaled.loc[cpu_test.index]

            # Funkti0onm som skapar sekvenser för lstm med X historiska värden och y nästa värde
            def create_sequences(data, time_steps=TIME_STEPS):
                X, y = [], []
                for i in range(len(data) - time_steps):
                    X.append(data[i:i+time_steps])
                    y.append(data[i+time_steps])
                return np.array(X), np.array(y)

            # Använder funktionen
            X_train_seq, y_train_seq = create_sequences(train_scaled.values)
            X_test_seq, y_test_seq = create_sequences(test_scaled.values)

            # Ändrar data till PyTorch-tensorer
            X_train_tensor = torch.from_numpy(X_train_seq).float()
            y_train_tensor = torch.from_numpy(y_train_seq).float().view(-1, 1)
            X_test_tensor = torch.from_numpy(X_test_seq).float()
            y_test_tensor = torch.from_numpy(y_test_seq).float().view(-1, 1)

            # === 2. Modell ===
            # Bygger LSTM modellen
            class LSTMModel(nn.Module):
                def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, output_dim)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            train_losses, val_losses = [], []

            # === 3. Träning ===
            # Träningsloop
            for epoch in range(1, EPOCHS+1):
                model.train()
                optimizer.zero_grad()
                output = model(X_train_tensor)
                loss = criterion(output, y_train_tensor)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_output = model(X_test_tensor)
                    val_loss = criterion(val_output, y_test_tensor)

                train_losses.append(loss.item())
                val_losses.append(val_loss.item())

            # === Prediktion och visualisering ===
            with torch.no_grad():
                y_pred_scaled = model(X_test_tensor).cpu().numpy()
                y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            pred_index = self.y_test.index[TIME_STEPS:]
            y_true = self.y_test.loc[pred_index]

            # === Result ===
            result_df = pd.DataFrame({
                'Actual': y_true,
                'Prediction': y_pred
            }, index=pred_index)

            ax = result_df['Actual'].plot(figsize=(15, 5), alpha=0.3, color='skyblue')
            result_df['Prediction'].plot(ax=ax, style='--', color='orange', linewidth=2)
            plt.legend(['Actual', 'Predicted'])
            ax.set_title('LSTM Forecast')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

            # === Training vs Validation Loss ===
            plt.figure(figsize=(12, 4))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title("Training vs Validation Loss (LSTM)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mean_actual = result_df['Actual'].mean()
            std_actual = result_df['Actual'].std()
            relative_error = (rmse / mean_actual) * 100

            print(f'RMSE: {rmse:.2f}')
            print(f'Relative Error: {relative_error:.2f}%')

            # === Statistik ===
            test_df = pd.DataFrame({
                'actual': y_true,
                'prediction': y_pred
            }, index=pred_index)

            mean_val = test_df['actual'].mean()
            std_val = test_df['actual'].std()
            mae = mean_absolute_error(test_df['actual'], test_df['prediction'])
            rmse = np.sqrt(mean_squared_error(test_df['actual'], test_df['prediction']))
            relative_error = (rmse / mean_val) * 100
            relative_mae = (mae / mean_val) * 100

            print(f'RMSE Score on Test set: {rmse:.2f}')
            print(f"Mean of y_test: {mean_val:.2f}")
            print(f"Standard Deviation of y_test: {std_val:.2f}")
            print(f"Relative Error (RMSE): {relative_error:.2f}%")
            print(f"MAE Score on Test set: {mae:.2f}")
            print(f"Relative Error (MAE): {relative_mae:.2f}%")




        # elif model_type.lower() == "lstm":
        #     print("Fitting LSTM model...")
        #     start_time = time.time()

        #     # Scale target
        #     scaler = MinMaxScaler()
        #     y_train_scaled = scaler.fit_transform(self.y_train.values.reshape(-1, 1))
        #     y_test_scaled = scaler.transform(self.y_test.values.reshape(-1, 1))

        #     # LSTM behöver sekvenser (X timsteg bakåt för att förutsäga nästa steg)
        #     def create_sequences(X, y, time_steps=24):
        #         Xs, ys = [], []
        #         for i in range(len(X) - time_steps):
        #             v = X.iloc[i:(i + time_steps)].values
        #             Xs.append(v)
        #             ys.append(y[i + time_steps])
        #         return np.array(Xs), np.array(ys)

        #     TIME_STEPS = 24  # T.ex., använd 24 timmar bakåt för att förutsäga nästa

        #     X_train_seq, y_train_seq = create_sequences(self.X_train, y_train_scaled, TIME_STEPS)
        #     X_test_seq, y_test_seq = create_sequences(self.X_test, y_test_scaled, TIME_STEPS)

        #     # Skapa LSTM-modellen
        #     model = Sequential()
        #     model.add(tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))) 
        #     model.add(LSTM(64, activation='relu', return_sequences=True))  # Lägg till en LSTM som lämnar sekvenser
        #     model.add(LSTM(32, activation='relu'))
        #     model.add(Dense(1))
        #     model.compile(optimizer='adam', loss='mse')

        #     # Träna modellen
        #     history = model.fit(
        #         X_train_seq, 
        #         y_train_seq, 
        #         epochs=20, 
        #         batch_size=64, 
        #         validation_split=0.1, 
        #         verbose=1
        #         )

        #     self.model = model  # Spara modellen

        #     # Prediktera
        #     y_pred_scaled = model.predict(X_test_seq)
        #     y_pred = scaler.inverse_transform(y_pred_scaled)

        #     # Justera test-indexet för att matcha sekvensen
        #     test_index = self.y_test.index[TIME_STEPS:]

        #     plt.figure(figsize=(15, 5))
        #     plt.plot(test_index, self.y_test.iloc[TIME_STEPS:], label='Actual')
        #     plt.plot(test_index, y_pred.flatten(), label='Predicted')
        #     plt.legend()
        #     plt.title('LSTM Prediction vs Actual')
        #     plt.show()

        #     rmse = np.sqrt(mean_squared_error(self.y_test.iloc[TIME_STEPS:], y_pred.flatten()))
        #     print(f"RMSE: {rmse}")
        #     end_time = time.time()
        #     print(f"Process LSTM took {end_time - start_time:.2f} seconds.")
    
        elif model_type.lower() == "cnn":
            self.train_cnn()
            # print("Fitting CNN model...")
            # start_time = time.time()
            # TIME_STEPS = 24
            # output_steps = 1

            # #Förberedelser
            # # Scale target
            # scaler = MinMaxScaler()
            # y_train_scaled = scaler.fit_transform(self.y_train.values.reshape(-1, 1))
            # y_test_scaled = scaler.transform(self.y_test.values.reshape(-1, 1))

            # def create_sequences(X, y, time_steps=24):
            #     Xs, ys = [], []
            #     for i in range(len(X) - time_steps):
            #         v = X.iloc[i:(i + time_steps)].values
            #         Xs.append(v)
            #         ys.append(y[i + time_steps])
            #     return np.array(Xs), np.array(ys)
            
            # X_train_seq, y_train_seq = create_sequences(self.X_train, y_train_scaled, TIME_STEPS)
            # X_test_seq, y_test_seq = create_sequences(self.X_test, y_test_scaled, TIME_STEPS)

            # input_timesteps = X_train_seq.shape[1]  # 24
            # features = X_train_seq.shape[2]   

            # # Bygger modellen
            # model = Sequential([
            #     Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_timesteps, features)),
            #     MaxPooling1D(pool_size=2),
            #     Conv1D(filters=128, kernel_size=3, activation='relu'),
            #     Flatten(),
            #     Dense(64, activation='relu'),
            #     Dense(output_steps)  # No activation -> regression output
            # ])
            
            # model.compile(optimizer='adam',
            #   loss='mse',      # Mean Squared Error is typical for forecasting
            #   metrics=['mae']) # Mean Absolute Error also useful to track

            # model.summary()

            # # Träna
            # model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

            # self.model = model

            # # Prediktera
            # y_pred_scaled = model.predict(X_test_seq)
            # y_pred = scaler.inverse_transform(y_pred_scaled)

            # test_index = self.y_test.index[TIME_STEPS:]

            # plt.figure(figsize=(15, 5))
            # plt.plot(test_index, self.y_test.iloc[TIME_STEPS:], label='Actual')
            # plt.plot(test_index, y_pred.flatten(), label='Predicted')
            # plt.legend()
            # plt.title('CNN Prediction vs Actual')
            # plt.show()

            # rmse = np.sqrt(mean_squared_error(self.y_test.iloc[TIME_STEPS:], y_pred.flatten()))
            # print(f"RMSE: {rmse}")
            # end_time = time.time()
            # print(f"Process CNN took {end_time - start_time:.2f} seconds.")

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

        # Calculate RMSE
        score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')
        print("Mean of y_test:", test[self.target].mean())  # Use inverse-transformed y_test
        print("Standard Deviation of y_test:", test[self.target].std())  # Use inverse-transformed y_test
        relative_error = (score / test[self.target].mean()) * 100
        print(f"Relative Error: {relative_error:.2f}%")        

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

        # Calculate RMSE
        score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')
        print("Mean of y_test:", test[self.target].mean())
        print("Standard Deviation of y_test:", test[self.target].std())
        relative_error = (score / test[self.target].mean()) * 100
        print(f"Relative Error: {relative_error:.2f}%")

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

        # Calculate RMSE
        score = np.sqrt(mean_squared_error(test[self.target], test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')
        print("Mean of y_test:", test[self.target].mean())
        print("Standard Deviation of y_test:", test[self.target].std())
        relative_error = (score / test[self.target].mean()) * 100
        print(f"Relative Error: {relative_error:.2f}%")

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
            'C': [0.1, 1, 10, 100],  # Regularization parameter
            'epsilon': [0.01, 0.1, 0.2, 0.5],  # Epsilon in the epsilon-SVR model
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
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
        print("Fitting CNN model...")
        start_time = time.time()

        # === Hyperparametrar ===
        TIME_STEPS = 24  # Antal tidssteg som används för att skapa en sekvens
        OUTPUT_STEPS = 1  # Antal steg framåt modellen ska förutsäga

        # === 1. Skala målvariabeln (CPU) ===
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(self.y_test.values.reshape(-1, 1)).flatten()

        # === 2. Feature engineering: Skapa fördröjningar och rullande statistik ===
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        for lag in range(1, 3):  # CPU-värden från 1 och 2 steg tillbaka
            X_train[f'cpu_lag_{lag}'] = self.y_train.shift(lag)
            X_test[f'cpu_lag_{lag}'] = self.y_test.shift(lag)

        X_train['cpu_rolling_mean_3'] = self.y_train.rolling(window=3).mean()
        X_test['cpu_rolling_mean_3'] = self.y_test.rolling(window=3).mean()

        X_train['cpu_rolling_std_3'] = self.y_train.rolling(window=3).std()
        X_test['cpu_rolling_std_3'] = self.y_test.rolling(window=3).std()

        # === 3. Ta bort rader med NaN från feature engineering ===
        X_train.dropna(inplace=True)
        X_test.dropna(inplace=True)

        # Justera y så att det matchar feature-matriserna
        y_train_aligned = pd.Series(y_train_scaled, index=self.y_train.index).loc[X_train.index]
        y_test_aligned = pd.Series(y_test_scaled, index=self.y_test.index).loc[X_test.index]

        # === 4. Skapa sekvenser för CNN-inmatning ===
        def create_sequences(X, y, time_steps):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X.iloc[i:i + time_steps].values)
                ys.append(y.iloc[i + time_steps])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train_aligned, TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test_aligned, TIME_STEPS)

        # === 5. Definiera CNN-arkitekturen ===
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(TIME_STEPS, X_train_seq.shape[2])),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(OUTPUT_STEPS)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # === 6. Träna modellen ===
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100, batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        self.model = model

        # === 7. Prediktion på testdatan ===
        y_pred_scaled = model.predict(X_test_seq).flatten()
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # === 8. Skapa korrekt index för y_actual ===
        y_actual = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        test_index = self.y_test.loc[X_test.index[TIME_STEPS:]].index[:len(y_pred)]

        # === 9. Visualisering av prediktion ===
        plt.figure(figsize=(15, 5))
        plt.plot(test_index, y_actual, label='Actual', alpha=0.3, color='skyblue')
        plt.plot(test_index, y_pred, label='Predicted', color='orange', linewidth=2)
        plt.title("CNN Forecast (Preprocessed Input)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === 10. Visualisering av träningskurvor ===
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (CNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === 11. Utvärdering ===
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        print(f"RMSE: {rmse:.2f}")
        print(f"Process CNN took {time.time() - start_time:.2f} seconds.")

        # === Statistik ===
        test_df = pd.DataFrame({
            'actual': y_actual,
            'prediction': y_pred
        }, index=test_index)

        mean_val = test_df['actual'].mean()
        std_val = test_df['actual'].std()
        mae = mean_absolute_error(test_df['actual'], test_df['prediction'])
        rmse = np.sqrt(mean_squared_error(test_df['actual'], test_df['prediction']))
        relative_error = (rmse / mean_val) * 100
        relative_mae = (mae / mean_val) * 100

        print(f'RMSE Score on Test set: {rmse:.2f}')
        print(f"Mean of y_test: {mean_val:.2f}")
        print(f"Standard Deviation of y_test: {std_val:.2f}")
        print(f"Relative Error (RMSE): {relative_error:.2f}%")
        print(f"MAE Score on Test set: {mae:.2f}")
        print(f"Relative Error (MAE): {relative_mae:.2f}%")



    def train_cnn_lstm(self):
        print("Fitting CNN-LSTM...")
        start_time = time.time()

        TIME_STEPS = 96 # 4 dygn 
        OUTPUT_STEPS = 1

        # 1. Feature engineering
        df = self.X_train.copy()
        df[self.target] = self.y_train

        df['trend'] = df[self.target].rolling(window=3, center=True).mean()
        df['residual'] = df[self.target] - df['trend']

        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(list(range(0, 7)) + list(range(22, 24))).astype(int)


        # Extra residualbaserade features
        df['residual_rolling_mean_3'] = df['residual'].rolling(window=3).mean()
        df['residual_rolling_mean_12'] = df['residual'].rolling(window=12).mean()
        df['residual_rolling_std_3'] = df['residual'].rolling(window=3).std()
        df['residual_rolling_std_6'] = df['residual'].rolling(window=6).std()
        df['residual_diff'] = df['residual'].diff()
        df['residual_lag3'] = df['residual'].shift(3)
        df['residual_lag6'] = df['residual'].shift(6)

        df.dropna(inplace=True)

        y = df['residual']
        X = df.drop(columns=[self.target, 'trend', 'residual'])

        # 2. Skalning
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), index=X.index, columns=X.columns)
        y_scaled = pd.Series(target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)

        # 3. Skapa sekvenser
        def create_sequences(X, y, time_steps, output_steps):
            Xs, ys = [], []
            for i in range(len(X) - time_steps - output_steps + 1):
                v = X.iloc[i:i + time_steps].values
                o = y.iloc[i + time_steps:i + time_steps + output_steps].values
                Xs.append(v)
                ys.append(o)
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS, OUTPUT_STEPS)
        y_seq = y_seq.reshape((y_seq.shape[0], OUTPUT_STEPS))

        # 4. Split
        split_index = int(len(X_seq) * 0.8)
        X_train_seq, X_test_seq = X_seq[:split_index], X_seq[split_index:]
        y_train_seq, y_test_seq = y_seq[:split_index], y_seq[split_index:]

        # 5. Modell (tweaked)
        model = Sequential([
            Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            LSTM(256, return_sequences=True, activation='relu', recurrent_dropout=0.2),
            Dropout(0.2),
            LSTM(32, activation='relu', recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dense(OUTPUT_STEPS)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        self.model = model

        # 6. Prediktion
        y_pred_scaled = model.predict(X_test_seq)
        y_pred_residual = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # 7. Återskapa test_index och trend
        test_index = y.index[split_index + TIME_STEPS: split_index + TIME_STEPS + len(y_pred_residual)]
        trend = df.loc[test_index, 'trend'].values[:len(y_pred_residual)]
        final_pred = y_pred_residual.flatten() + trend

        # 8. Smoothing
        final_pred_smooth = pd.Series(final_pred, index=test_index).rolling(window=6, center=True).mean()

        # 9. Visualisering
        plt.figure(figsize=(15, 5))
        plt.plot(df.loc[test_index, self.target], label='Actual', alpha=0.2, color='skyblue')
        plt.plot(final_pred_smooth, label='Predicted', color='orange', linewidth=3)
        plt.legend()
        plt.title('CNN-LSTM')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("cnn_lstm_forecast.png", dpi=300, bbox_inches='tight')
        plt.show()

        # === 9. Träning vs Validerings-loss ===
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training vs Validation Loss (CNN-LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cnn_lstm_loss_plot.png", dpi=300)
        plt.show()

        # === 10. RMSE och summering ===
        target_series = df.loc[test_index, self.target]
        common_index = final_pred_smooth.dropna().index
        rmse = np.sqrt(mean_squared_error(target_series.loc[common_index], final_pred_smooth.loc[common_index]))
        print(f"RMSE: {rmse:0.2f}")
        print(f"Process CNN-LSTM took {time.time() - start_time:.2f} seconds.")
    
        # === Statistik ===
        test_df = pd.DataFrame({
            'actual': target_series.loc[common_index],
            'prediction': final_pred_smooth.loc[common_index]
        })

        mean_val = test_df['actual'].mean()
        std_val = test_df['actual'].std()
        mae = mean_absolute_error(test_df['actual'], test_df['prediction'])
        rmse = np.sqrt(mean_squared_error(test_df['actual'], test_df['prediction']))
        relative_error = (rmse / mean_val) * 100
        relative_mae = (mae / mean_val) * 100

        print(f'RMSE Score on Test set: {rmse:.2f}')
        print(f"Mean of y_test: {mean_val:.2f}")
        print(f"Standard Deviation of y_test: {std_val:.2f}")
        print(f"Relative Error (RMSE): {relative_error:.2f}%")
        print(f"MAE Score on Test set: {mae:.2f}")
        print(f"Relative Error (MAE): {relative_mae:.2f}%")
