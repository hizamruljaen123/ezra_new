# Global Rice Price Prediction using SARIMAX Algorithm

## Overview
This project predicts **global rice (gabah) prices** using the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)** algorithm.  
SARIMAX accounts for **seasonality, trends**, and **external factors** (like oil prices, weather indices, or global demand).

## Features
- Predict **future rice prices** using historical data.  
- Incorporate **exogenous variables** that affect rice prices globally.  
- Evaluate predictions using metrics like **RMSE** and **MAPE**.  

## Steps
1. **Data Collection**: Historical rice price data + exogenous features.  
2. **Data Preprocessing**: Handle missing values, seasonal decomposition.  
3. **SARIMAX Modeling**: Fit SARIMAX model to historical data.  
4. **Prediction**: Forecast rice prices for future periods.  
5. **Evaluation**: Calculate RMSE, MAPE, and visualize results.  

## Python Example

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load historical rice price data
data = pd.read_csv("rice_price_global.csv", parse_dates=['date'], index_col='date')
prices = data['price']

# Example exogenous variable: global oil price
exog = data[['oil_price']]

# Fit SARIMAX model (order=(1,1,1), seasonal_order=(1,1,1,12) for monthly seasonality)
model = SARIMAX(prices, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Forecast next 12 months
forecast = model_fit.predict(start=len(prices), end=len(prices)+11, exog=exog[-12:])

# Plot
plt.plot(prices, label='Historical Price')
plt.plot(forecast, label='Forecast', color='red')
plt.title("Global Rice Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
