from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# from pmdarima import auto_arima 

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout


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
           # self.model = SARIMAX(self.y_train, order=(3, 0, 3), seasonal_order=(1, 0, 1, 48),enforce_stationarity=False, enforce_invertibility=False)  # Example order (p=5, d=0, q=2) season (P,D=1,Q,s)
            self.model = ARIMA(self.y_train, order=(5, 0, 3))  # Example order (p=3, d=0, q=3)
            self.model = self.model.fit()
            end_time = time.time()
            print(f"Model fitting took {end_time - start_time:.2f} seconds.")


            # Print ARIMA summary
            print(self.model.summary())

            # Forecast on the test set
            #.predict() ?
            forecast = self.model.forecast(len(self.y_test))
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

        elif model_type.lower() == "lstm":
            print("Fitting LSTM model...")
            start_time = time.time()

            # Scale target
            scaler = MinMaxScaler()
            y_train_scaled = scaler.fit_transform(self.y_train.values.reshape(-1, 1))
            y_test_scaled = scaler.transform(self.y_test.values.reshape(-1, 1))

            # LSTM behöver sekvenser (X timsteg bakåt för att förutsäga nästa steg)
            def create_sequences(X, y, time_steps=24):
                Xs, ys = [], []
                for i in range(len(X) - time_steps):
                    v = X.iloc[i:(i + time_steps)].values
                    Xs.append(v)
                    ys.append(y[i + time_steps])
                return np.array(Xs), np.array(ys)

            TIME_STEPS = 24  # T.ex., använd 24 timmar bakåt för att förutsäga nästa

            X_train_seq, y_train_seq = create_sequences(self.X_train, y_train_scaled, TIME_STEPS)
            X_test_seq, y_test_seq = create_sequences(self.X_test, y_test_scaled, TIME_STEPS)

            # Skapa LSTM-modellen
            model = Sequential()
            model.add(tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))) 
            model.add(LSTM(64, activation='relu', return_sequences=True))  # Lägg till en LSTM som lämnar sekvenser
            model.add(LSTM(32, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            # Träna modellen
            history = model.fit(
                X_train_seq, 
                y_train_seq, 
                epochs=20, 
                batch_size=64, 
                validation_split=0.1, 
                verbose=1
                )

            self.model = model  # Spara modellen

            # Prediktera
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            # Justera test-indexet för att matcha sekvensen
            test_index = self.y_test.index[TIME_STEPS:]

            plt.figure(figsize=(15, 5))
            plt.plot(test_index, self.y_test.iloc[TIME_STEPS:], label='Actual')
            plt.plot(test_index, y_pred.flatten(), label='Predicted')
            plt.legend()
            plt.title('LSTM Prediction vs Actual')
            plt.show()

            rmse = np.sqrt(mean_squared_error(self.y_test.iloc[TIME_STEPS:], y_pred.flatten()))
            print(f"RMSE: {rmse}")
            end_time = time.time()
            print(f"Process LSTM took {end_time - start_time:.2f} seconds.")
    
        elif model_type.lower() == "cnn":
            print("Fitting CNN model...")
            start_time = time.time()
            TIME_STEPS = 24
            output_steps = 1

            #Förberedelser
            # Scale target
            scaler = MinMaxScaler()
            y_train_scaled = scaler.fit_transform(self.y_train.values.reshape(-1, 1))
            y_test_scaled = scaler.transform(self.y_test.values.reshape(-1, 1))

            def create_sequences(X, y, time_steps=24):
                Xs, ys = [], []
                for i in range(len(X) - time_steps):
                    v = X.iloc[i:(i + time_steps)].values
                    Xs.append(v)
                    ys.append(y[i + time_steps])
                return np.array(Xs), np.array(ys)
            
            X_train_seq, y_train_seq = create_sequences(self.X_train, y_train_scaled, TIME_STEPS)
            X_test_seq, y_test_seq = create_sequences(self.X_test, y_test_scaled, TIME_STEPS)

            input_timesteps = X_train_seq.shape[1]  # 24
            features = X_train_seq.shape[2]   

            # Bygger modellen
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_timesteps, features)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=3, activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(output_steps)  # No activation -> regression output
            ])
            
            model.compile(optimizer='adam',
              loss='mse',      # Mean Squared Error is typical for forecasting
              metrics=['mae']) # Mean Absolute Error also useful to track

            model.summary()

            # Träna
            model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

            self.model = model

            # Prediktera
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            test_index = self.y_test.index[TIME_STEPS:]

            plt.figure(figsize=(15, 5))
            plt.plot(test_index, self.y_test.iloc[TIME_STEPS:], label='Actual')
            plt.plot(test_index, y_pred.flatten(), label='Predicted')
            plt.legend()
            plt.title('CNN Prediction vs Actual')
            plt.show()

            rmse = np.sqrt(mean_squared_error(self.y_test.iloc[TIME_STEPS:], y_pred.flatten()))
            print(f"RMSE: {rmse}")
            end_time = time.time()
            print(f"Process CNN took {end_time - start_time:.2f} seconds.")

        elif model_type.lower() == "cnn-lstm":
            print("Fitting CNN-LSTM model...")
            start_time = time.time()
            TIME_STEPS = 24
            output_steps = 1

            #Förberedelser
            # Scale target
            scaler = MinMaxScaler()
            y_train_scaled = scaler.fit_transform(self.y_train.values.reshape(-1, 1))
            y_test_scaled = scaler.transform(self.y_test.values.reshape(-1, 1))

            def create_sequences(X, y, time_steps=24):
                Xs, ys = [], []
                for i in range(len(X) - time_steps):
                    v = X.iloc[i:(i + time_steps)].values
                    Xs.append(v)
                    ys.append(y[i + time_steps])
                return np.array(Xs), np.array(ys)
            
            X_train_seq, y_train_seq = create_sequences(self.X_train, y_train_scaled, TIME_STEPS)
            X_test_seq, y_test_seq = create_sequences(self.X_test, y_test_scaled, TIME_STEPS)

            input_timesteps = X_train_seq.shape[1]  # 24
            features = X_train_seq.shape[2]   

            # Bygger modellen
            model = Sequential([
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_timesteps, features)),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(50, activation='relu', return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(output_steps)
            ])
            
            model.compile(optimizer='adam',
              loss='mse',      # Mean Squared Error is typical for forecasting
              metrics=['mae']) # Mean Absolute Error also useful to track

            model.summary()

            # Träna
            model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

            self.model = model

            # Prediktera
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            test_index = self.y_test.index[TIME_STEPS:]

            plt.figure(figsize=(15, 5))
            plt.plot(test_index, self.y_test.iloc[TIME_STEPS:], label='Actual')
            plt.plot(test_index, y_pred.flatten(), label='Predicted')
            plt.legend()
            plt.title('CNN-LSTM Prediction vs Actual')
            plt.show()
            
            rmse = np.sqrt(mean_squared_error(self.y_test.iloc[TIME_STEPS:], y_pred.flatten()))
            print(f"RMSE: {rmse}")
            end_time = time.time()
            print(f"Process CNN-LSTM took {end_time - start_time:.2f} seconds.")

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
        evals_result = {}
        best_params, self.model = self.tune_xgboost(self.X_train, self.y_train)
        self.model.set_params(eval_metric="rmse",evals_result = evals_result)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=100,
            
        )
        fi = pd.DataFrame(data=self.model.feature_importances_,
                        index=self.model.feature_names_in_,
                        columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')

        
        # Plot training and validation loss
        train_rmse = evals_result['validation_0']['rmse']
        val_rmse = evals_result['validation_1']['rmse']
        plt.figure(figsize=(8, 4))
        plt.plot(train_rmse, label='Train RMSE')
        plt.plot(val_rmse, label='Validation RMSE')
        plt.xlabel('Boosting Round')
        plt.ylabel('RMSE')
        plt.title('XGBoost Training and Validation RMSE')
        plt.legend()
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
    