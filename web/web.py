from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
import plotly.graph_objs as go
import json
import plotly


app = Flask(__name__)

# Fungsi untuk memuat dan membersihkan data dari file data.csv secara default
def load_and_clean_data():
    file_path = 'data.csv'  # Path ke file data.csv
    data = pd.read_csv(file_path, delimiter=';')
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
    return data[['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']]

# Fungsi untuk uji ADF
def uji_adf(series):
    hasil = adfuller(series, autolag='AIC')
    return hasil[1] <= 0.05

# Fungsi untuk differencing
def differencing(series, d):
    diff_series = series.copy()
    for _ in range(d):
        diff_series = np.diff(diff_series, n=1)
    return np.concatenate(([np.nan] * d, diff_series))

# Fungsi untuk autoregressive component
def autoregressive_component(series, p):
    ar_component = np.zeros_like(series)
    for t in range(p, len(series)):
        ar_component[t] = np.dot(series[t-p:t][::-1], np.ones(p))
    return ar_component

# Fungsi untuk moving average component
def moving_average_component(series, q):
    ma_component = np.zeros_like(series)
    for t in range(q, len(series)):
        ma_component[t] = np.mean(series[t-q:t])
    return ma_component

# Fungsi untuk SARIMA manual
def manual_sarima(series, p, d, q, P, D, Q, m):
    diff_series = differencing(series, d)
    seasonal_diff_series = differencing(diff_series, D * m)
    ar_series = autoregressive_component(seasonal_diff_series, p)
    ma_series = moving_average_component(seasonal_diff_series, q)
    seasonal_ar_series = autoregressive_component(seasonal_diff_series, P)
    seasonal_ma_series = moving_average_component(seasonal_diff_series, Q)
    sarima_series = ar_series + ma_series + seasonal_ar_series + seasonal_ma_series
    return sarima_series

# Fungsi untuk SARIMAX manual dengan eksogen
def manual_sarimax(series, exog, p, d, q, P, D, Q, m):
    diff_series = differencing(series, d)
    seasonal_diff_series = differencing(diff_series, D * m)
    ar_series = autoregressive_component(seasonal_diff_series, p)
    ma_series = moving_average_component(seasonal_diff_series, q)
    seasonal_ar_series = autoregressive_component(seasonal_diff_series, P)
    seasonal_ma_series = moving_average_component(seasonal_diff_series, Q)
    exog_series = np.dot(exog, np.ones(exog.shape[1]))
    sarimax_series = ar_series + ma_series + seasonal_ar_series + seasonal_ma_series + exog_series
    return sarimax_series

# Fungsi untuk menghitung AIC
def calculate_aic(y, y_pred, k):
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    aic = len(y) * np.log(sse/len(y)) + 2 * k
    return aic

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - predicted))
    mape = np.mean(np.abs((original - predicted) / original)) * 100
    return mse, rmse, mae, mape


def find_best_arima_params(data, p_values, d_values, q_values):
    best_aic = np.inf
    best_params = None
    best_model = None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    y_pred = manual_arima(data, p, d, q)
                    aic = calculate_aic(data[d:], y_pred[d:], p + q)
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = y_pred
                except:
                    continue
    return best_params, best_model

# Fungsi untuk ARIMA manual
# Fungsi untuk ARIMA manual
def manual_arima(series, p, d, q):
    diff_series = differencing(series, d)
    ar_series = autoregressive_component(diff_series, p)
    ma_series = moving_average_component(diff_series, q)
    arima_series = ar_series + ma_series
    return arima_series
# Memperbaiki find_best_arima_params
def find_best_arima_params(data, p_values, d_values, q_values):
    best_aic = np.inf
    best_params = None
    best_model = None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    y_pred = manual_arima(data, p, d, q)
                    aic = calculate_aic(data[d:], y_pred[d:], p + q)
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = y_pred
                except:
                    continue
    return best_params, best_model

# Memperbaiki find_best_sarima_params
def find_best_sarima_params(data, p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
    best_aic = np.inf
    best_params = None
    best_model = None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for m in m_values:
                                try:
                                    # Generate SARIMA model
                                    y_pred = manual_sarima(data, p, d, q, P, D, Q, m)
                                    
                                    # Calculate AIC
                                    aic = calculate_aic(data[max(d, D * m):], y_pred[max(d, D * m):], p + q + P + Q)
                                    
                                    # Update best model if current model is better
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q, P, D, Q, m)
                                        best_model = y_pred
                                except Exception as e:
                                    continue
    
    return best_params, best_model

# Memperbaiki find_best_sarimax_params
def find_best_sarimax_params(data, exog, p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
    best_aic = np.inf
    best_params = None
    best_model = None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for m in m_values:
                                try:
                                    y_pred = manual_sarimax(data, exog, p, d, q, P, D, Q, m)
                                    y_pred = np.nan_to_num(y_pred, nan=1)
                                    data_cleaned = np.nan_to_num(data, nan=1)
                                    aic = calculate_aic(data_cleaned[max(d, D * m):], y_pred[max(d, D * m):], p + q + P + Q + exog.shape[1])
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q, P, D, Q, m)
                                        best_model = y_pred
                                except:
                                    continue
    return best_params, best_model



@app.route('/')
def index():
    data = load_and_clean_data()

    # Model predictions
    data['MA'] = data['Terakhir'].rolling(window=5).mean()
    p_values, d_values, q_values = range(0, 3), range(0, 3), range(0, 3)
    P_values, D_values, Q_values, m_values = range(0, 3), range(0, 3), range(0, 3), [6]

    # ARIMA
    _, data['ARIMA'] = find_best_arima_params(data['Terakhir'].values, p_values, d_values, q_values)

    # SARIMA
    _, data['SARIMA'] = find_best_sarima_params(data['Terakhir'].values, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)

    # SARIMAX (example with dummy exog data)
    data_exog = pd.DataFrame({
        'Curah Hujan': np.random.uniform(100, 130, size=len(data)),
        'Jumlah Produksi': np.random.uniform(1100, 1500, size=len(data))
    })
    data = pd.concat([data.reset_index(drop=True), data_exog], axis=1)
    exog = data[['Curah Hujan', 'Jumlah Produksi']].values
    _, data['SARIMAX'] = find_best_sarimax_params(data['Terakhir'].values, exog, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)


    # Bulatkan nilai SARIMAX menjadi dua angka desimal
    data['SARIMAX'] = data['SARIMAX'].round(1)


    # Prepare plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['Terakhir'], mode='lines', name='Harga Asli'))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['MA'], mode='lines', name='MA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['ARIMA'], mode='lines', name='ARIMA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMA'], mode='lines', name='SARIMA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMAX'], mode='lines', name='SARIMAX', line=dict(dash='dash')))

    fig.update_layout(title='Harga Gabah dan Prediksi', xaxis_title='Tanggal', yaxis_title='Harga')

    # Ensure that the graph is correctly encoded as JSON
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', tables=data.to_dict(orient='records'), graph_json=graph_json)

if __name__ == '__main__':
    app.run(debug=True)

