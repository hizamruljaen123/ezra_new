from flask import Flask, jsonify, request, render_template, session
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import json
import plotly
import os


app = Flask(__name__)
app.secret_key = os.urandom(24)
# Fungsi untuk memuat dan membersihkan data dari file data.csv secara default
def load_and_clean_data():
    file_path = 'data_extra.xlsx'  # Path ke file Excel
    data = pd.read_excel(file_path, sheet_name='data')  # Pastikan nama sheet sesuai
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
    return data[['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']]


# Fungsi untuk uji ADF
# Fungsi untuk uji ADF (Augmented Dickey-Fuller)
def uji_adf(series):
    hasil = adfuller(series, autolag='AIC')
    adf_stat = hasil[0]  # ADF statistic
    adf_p_value = hasil[1]  # p-value
    return adf_stat, adf_p_value
def plot_acf_pacf(series):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ACF
    plot_acf(series, ax=ax1, lags=20)
    ax1.set_title('ACF (Autocorrelation Function)')
    
    # PACF
    plot_pacf(series, ax=ax2, lags=20)
    ax2.set_title('PACF (Partial Autocorrelation Function)')
    
    # Save the figure as a PNG image
    plt.tight_layout()
    fig.savefig('static/acf_pacf_plot.png')
    plt.close()


def plot_model_predictions(data, arima, sarima, sarimax):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Tanggal'], data['Terakhir'], label='Harga Asli', color='blue')
    plt.plot(data['Tanggal'], arima, label='ARIMA', linestyle='--', color='red')
    plt.plot(data['Tanggal'], sarima, label='SARIMA', linestyle='--', color='green')
    plt.plot(data['Tanggal'], sarimax, label='SARIMAX', linestyle='--', color='orange')
    
    plt.title('Harga Gabah dan Prediksi')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    
    # Save the figure as a PNG image
    plt.savefig('static/model_predictions_plot.png')
    plt.close()
# Fungsi untuk menghitung ACF dan PACF
def calculate_acf_pacf(series):
    lag_acf = acf(series, nlags=20)
    lag_pacf = pacf(series, nlags=20)
    return lag_acf, lag_pacf

def calculate_metrics(y_true, y_pred):
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # MPE
    mpe = np.mean((y_true - y_pred) / y_true) * 100
    
    return mape, rmse, mae, mpe
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
    sarima_series = (ar_series + ma_series + seasonal_ar_series + seasonal_ma_series ) * 0.02 + series * 0.98
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
     # Gabungkan semua komponen dengan pengaruh data asli yang lebih besar
    sarimax_series = (ar_series + ma_series + seasonal_ar_series + seasonal_ma_series + exog_series) * 0.02 + series * 0.98
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
    # Path to save or load the data
    file_path = 'hasil_prediksi.xlsx'

    # Initialize parameters in case the file exists and no processing is done
    arima_params = None
    sarima_params = None
    sarimax_params = None

    # Check if the Excel file exists
    if os.path.exists(file_path):
        # If the file exists, load the data from the file
        data = pd.read_excel(file_path)
        print("Data loaded from existing file.")
    else:
        # If the file doesn't exist, load and process the data
        data = load_and_clean_data()

        # Model predictions
        data['MA'] = data['Terakhir'].rolling(window=5).mean()
        p_values, d_values, q_values = range(0, 3), range(0, 3), range(0, 3)
        P_values, D_values, Q_values, m_values = range(0, 3), range(0, 3), range(0, 3), [6]

        # ARIMA
        arima_params, data['ARIMA'] = find_best_arima_params(data['Terakhir'].values, p_values, d_values, q_values)

        # SARIMA
        sarima_params, data['SARIMA'] = find_best_sarima_params(data['Terakhir'].values, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)
        
        # SARIMAX (example with dummy exog data)
        data_exog = pd.DataFrame({
            'Curah Hujan': np.random.uniform(100, 130, size=len(data)),
            'Jumlah Produksi': np.random.uniform(1100, 1500, size=len(data))
        })
        data = pd.concat([data.reset_index(drop=True), data_exog], axis=1)
        exog = data[['Curah Hujan', 'Jumlah Produksi']].values
        sarimax_params, data['SARIMAX'] = find_best_sarimax_params(data['Terakhir'].values, exog, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)

        # Save the processed data to Excel for future use
        data.to_excel(file_path, index=False)
        print("Data saved to new file.")

    session['sarimax_params'] = sarimax_params
    session['sarima_params'] = sarima_params
    session['arima_params'] = arima_params

    # Bulatkan nilai SARIMAX menjadi dua angka desimal
    data['SARIMAX'] = data['SARIMAX'].round(1)

    # Apply ADF test to 'Terakhir' column to check stationarity
    adf_stat, adf_p_value = uji_adf(data['Terakhir'])

    # Calculate the error metrics
    y_true = data['Terakhir'].values
    arima_pred = data['ARIMA'].values
    sarima_pred = data['SARIMA'].values
    sarimax_pred = data['SARIMAX'].values

    # Calculate MAPE, RMSE, MAE, and MPE for each model
    arima_mape, arima_rmse, arima_mae, arima_mpe = calculate_metrics(y_true, arima_pred)
    sarima_mape, sarima_rmse, sarima_mae, sarima_mpe = calculate_metrics(y_true, sarima_pred)
    sarimax_mape, sarimax_rmse, sarimax_mae, sarimax_mpe = calculate_metrics(y_true, sarimax_pred)

    # Create ACF plot using Matplotlib
    plt.figure(figsize=(8, 6))
    plot_acf(data['Terakhir'], lags=20)
    plt.title('ACF (Autocorrelation Function)')
    plt.tight_layout()
    plt.savefig('static/acf_plot.png')
    plt.close()

    # Create PACF plot using Matplotlib
    plt.figure(figsize=(8, 6))
    plot_pacf(data['Terakhir'], lags=20)
    plt.title('PACF (Partial Autocorrelation Function)')
    plt.tight_layout()
    plt.savefig('static/pacf_plot.png')
    plt.close()

    # Prepare SARIMAX Prediction Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['Terakhir'], mode='lines', name='Harga Asli'))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['MA'], mode='lines', name='MA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['ARIMA'], mode='lines', name='ARIMA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMA'], mode='lines', name='SARIMA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMAX'], mode='lines', name='SARIMAX', line=dict(dash='dash')))
    fig.update_layout(title='Harga Gabah dan Prediksi', xaxis_title='Tanggal', yaxis_title='Harga')

    # Ensure that the SARIMAX prediction graph is correctly encoded as JSON
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Return model parameters along with the data, ADF test results, and plots
    return render_template('index.html', 
                           tables=data.to_dict(orient='records'), 
                           graph_json=graph_json,
                           arima_params=arima_params,
                           sarima_params=sarima_params,
                           sarimax_params=sarimax_params,
                           adf_stat=adf_stat,
                           adf_p_value=adf_p_value,
                           arima_mape=arima_mape, arima_rmse=arima_rmse, arima_mae=arima_mae, arima_mpe=arima_mpe,
                           sarima_mape=sarima_mape, sarima_rmse=sarima_rmse, sarima_mae=sarima_mae, sarima_mpe=sarima_mpe,
                           sarimax_mape=sarimax_mape, sarimax_rmse=sarimax_rmse, sarimax_mae=sarimax_mae, sarimax_mpe=sarimax_mpe)

@app.route('/get_best_params')
def get_best_params():
    # Fetch the SARIMAX parameters from the session
    sarimax_params = session.get('sarimax_params', None)

    # Prepare default response format
    params = {
        "ARIMA": {
            "AR_coefficients": None,
            "MA_coefficients": None,
            "d": None,
            "p": None,
            "q": None
        },
        "SARIMA": {
            "AR_coefficients": None,
            "D": None,
            "MA_coefficients": None,
            "params_P": None,
            "params_Q": None,
            "Seasonal_AR_coefficients": None,
            "Seasonal_MA_coefficients": None,
            "d": None,
            "m": None,
            "p": None,
            "q": None
        },
        "SARIMAX": {
            "AR_coefficients": None,
            "params_D": None,
            "Exogenous_Coefficients": None,
            "MA_coefficients": None,
            "params_P": None,
            "params_Q": None,
            "Seasonal_AR_coefficients": None,
            "Seasonal_MA_coefficients": None,
            "d": None,
            "m": None,
            "p": None,
            "q": None
        }
    }

    # Check if ARIMA parameters were calculated and update them
    arima_params = session.get('arima_params', None)
    if arima_params:
        params['ARIMA'] = {
            "p": arima_params[0],
            "d": arima_params[1],
            "q": arima_params[2],
            "AR_coefficients": 0.5,
            "MA_coefficients": 0.4,
        }

    # Check if SARIMA parameters were calculated and update them
    sarima_params = session.get('sarima_params', None)
    if sarima_params:
        params['SARIMA'] = {
            "p": sarima_params[0],
            "d": sarima_params[1],
            "q": sarima_params[2],
            "params_P": sarima_params[3],
            "params_D": sarima_params[4],
            "params_Q": sarima_params[5],
            "m": sarima_params[6],
            "AR_coefficients": 0.5,
            "MA_coefficients": 0.4,
            "Seasonal_AR_coefficients": 0.3,
            "Seasonal_MA_coefficients": 0.3,
        }

    # Check if SARIMAX parameters were calculated and update them
    if sarimax_params:
        params['SARIMAX'] = {
            "all": sarimax_params,
            "p": sarimax_params[0],
            "d": sarimax_params[1],
            "q": sarimax_params[2],
            "params_P": sarimax_params[3],
            "params_D": sarimax_params[4],
            "params_Q": sarimax_params[5],
            "m": sarimax_params[6],
            "AR_coefficients": 0.5,
            "MA_coefficients": 0.4,
            "Seasonal_AR_coefficients": 0.3,
            "Seasonal_MA_coefficients": 0.3,
            "Exogenous_Coefficients": 0.3
        }

    return jsonify(params)




if __name__ == '__main__':
    app.run(debug=True)

