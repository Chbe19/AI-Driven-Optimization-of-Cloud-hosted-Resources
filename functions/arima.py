import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA 

df = pd.read_csv('functions/vm_cpu_data.csv', index_col=0, parse_dates=True)
df['CPU'] = df['CPU'].str.replace('%', '').astype(float)
df.info()
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df.plot(style='.',
    figsize=(15, 5),
    color=color_pal[0],
    title='CPU Usage Over Time')
plt.show()

df = np.log(df) 

df.plot()

split_time = pd.Timestamp("2025-03-11")
df_train = df[df.index < split_time].copy()
df_test = df[df.index >= split_time].copy()


acf_original = plot_acf(df_train)
pacf_original = plot_pacf(df_train)

adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')


df_train_diff = df_train.diff().dropna()
df_train_diff.plot()
plt.show()

acf_diff = plot_acf(df_train_diff)
pacf_diff = plot_pacf(df_train_diff)
plt.show()

adf_test = adfuller(df_train_diff)
print(f'p-value: {adf_test[1]}')

## ARIMA ## 

model = ARIMA(df_train, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())

residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()

acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)

forecast_test = model_fit.forecast(len(df_test))
df['forecast_210'] = [None]*len(df_train) + list(forecast_test)
df.plot()
plt.show()