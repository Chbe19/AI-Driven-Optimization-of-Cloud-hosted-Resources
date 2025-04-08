import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('functions/vm_cpu_data.csv', index_col=0, parse_dates=True)
print("Kolumner i CSV-filen:", df.columns.tolist())
df['CPU'] = df['CPU'].str.replace('%', '').astype(float)

print(df.head())

df.plot(style='.',
    figsize=(15, 5),
    color=color_pal[0],
    title='CPU Usage Over Time')
plt.show()

train = df.loc[df.index < '2025-03-11 03:27:00']
test = df.loc[df.index >= '2025-03-11 03:27:00']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('2025-03-11 03:27:00', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

df.loc[(df.index > '2025-03-10 00:00:00') & (df.index < '2025-03-11 03:27:00')] \
    .plot(figsize=(15, 5), title='Day Of Data')
plt.show()

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    return df

df = create_features(df)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='CPU')
ax.set_title('CPU by Hour')
plt.show()

train = create_features(train)
test = create_features(test)

FEATURES = ['minute', 'hour']
TARGET = 'CPU'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['CPU']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

score = np.sqrt(mean_squared_error(test['CPU'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')