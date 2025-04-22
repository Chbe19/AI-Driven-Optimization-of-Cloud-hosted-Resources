import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataPreprocessor:
    """
    A utility class for pre-processing time series data, including splitting
    the data into training and testing sets.
    """
    def __init__(self, df, split_percentage=0.2):
        self.split_percentage = split_percentage
        self.data_frame = df
        self.training_set = None
        self.testing_set = None
        self.features = None
        self.clean()
        self.split()

    def clean(self):
        """
        Clean the DataFrame by removing '%' characters and converting them to floats
        for columns that are not part of the index (e.g., 'Time').
        """
        for col in self.data_frame.columns:
            if col != 'Time':  # Exclude the 'Time' column or any index column
                if self.data_frame[col].dtype == 'object':  # Check if column contains strings
                    self.data_frame[col] = self.data_frame[col].str.replace('%', '').astype(float)
        self.data_frame = self.data_frame.dropna()

    def split(self):
        """
        Split the DataFrame into training and testing sets using train_test_split.

        Returns:
            tuple: A tuple containing the training and testing DataFrames.
        """
        if self.data_frame.index.duplicated().any():
            print("Duplicate index labels detected. Aggregating duplicates by taking the mean.")
            self.data_frame = self.data_frame.groupby(self.data_frame.index).mean()
        full_index = pd.date_range(start=self.data_frame.index.min(), end=self.data_frame.index.max(), freq='H')  # Adjust frequency as needed
        missing_dates = full_index.difference(self.data_frame.index)

        print(f"Missing dates: {len(missing_dates)}")
        print(missing_dates)
        # Reindex the DataFrame to include all dates and fill missing values
        self.data_frame = self.data_frame.reindex(full_index)
        self.data_frame = self.data_frame.fillna(method='ffill')  # Forward-fill missing values
        
        train, test = train_test_split(
            self.data_frame,
            test_size=self.split_percentage,
            random_state=100,
            shuffle=False  # Ensure the time series order is preserved
        )

       
        self.training_set = train
        self.testing_set = test
    
    def set_features(self, features):
        """
        Set the features for feature engineering by extracting parts of the string index.

        Args:
            features (list): A list of strings representing the features to extract
                             (e.g., 'hour', 'minute').
        """
        self.features = features
        df_copy = self.data_frame.copy()
        train_copy = self.training_set.copy()
        test_copy = self.testing_set.copy()

        for feature in features:
            if feature == 'hour':
                df_copy[feature] = self.data_frame.index.hour
                train_copy[feature] = self.training_set.index.hour
                test_copy[feature] = self.testing_set.index.hour
            elif feature == 'minute':  
                df_copy[feature] = self.data_frame.index.minute
                train_copy[feature] = self.training_set.index.minute
                test_copy[feature] = self.testing_set.index.minute
            elif feature == 'day':     
                df_copy[feature] = self.data_frame.index.day
                train_copy[feature] = self.training_set.index.day
                test_copy[feature] = self.testing_set.index.day
            elif feature == 'month':
                df_copy[feature] = self.data_frame.index.month
                train_copy[feature] = self.training_set.index.month
                test_copy[feature] = self.testing_set.index.month        
            else:
                raise ValueError(f"Unsupported feature: {feature}")

        self.data_frame = df_copy
        self.training_set = train_copy
        self.testing_set = test_copy

    def get_data_frame(self):
        return self.data_frame
    
    def get_training_set(self, target):
        """
        Get the training sets x and y.
        """
        return self.training_set[self.features], self.training_set[target]
    
    def get_testing_set(self, target):
        """
        Get the testing sets x and y.
        """
        return self.testing_set[self.features], self.testing_set[target]
    