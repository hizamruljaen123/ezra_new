from flask import Flask, render_template, redirect, url_for
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller
import json
import plotly
import subprocess
import os
import pickle  # Untuk menyimpan model prediksi
import json  # Untuk menyimpan parameter terbaik

app = Flask(__name__)

# Fungsi untuk memuat dan membersihkan data dari file data.csv secara default
def load_and_clean_data():
    file_path = 'data.csv'  # Path ke file data.csv
    data = pd.read_csv(file_path, delimiter=';')
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
    return data[['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']]

# Fungsi untuk menyimpan hasil prediksi ke Excel
def save_predictions_to_excel(data):
    output_file_path = 'hasil_prediksi.xlsx'
    data.to_excel(output_file_path, index=False)

# Fungsi untuk memuat hasil prediksi dari file Excel
def load_predictions_from_excel():
    file_path = 'hasil_prediksi.xlsx'
    data = pd.read_excel(file_path)
    return data

# Fungsi untuk menyimpan parameter terbaik ke file JSON
def save_best_params_to_file(params, file_path='best_params.json'):
    with open(file_path, 'w') as f:
        json.dump(params, f)

# Fungsi untuk memuat parameter terbaik dari file JSON
def load_best_params_from_file(file_path='best_params.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

# Fungsi untuk menyimpan model prediksi ke file menggunakan pickle
def save_model_to_file(model, file_path='best_model.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

# Fungsi untuk memuat model prediksi dari file menggunakan pickle
def load_model_from_file(file_path='best_model.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Fungsi untuk uji ADF dan menyimpan hasil ke dalam file teks
def uji_adf(series, file_path='adf_result.txt'):
    hasil = adfuller(series, autolag='AIC')
    adf_stat, p_value, used_lag, n_obs, crit_values, icbest = hasil

    # Simpan hasil ke file teks
    with open(file_path, 'w') as f:
        f.write(f'ADF Statistic: {adf_stat}\n')
        f.write(f'p-value: {p_value}\n')
        f.write(f'Used Lag: {used_lag}\n')
        f.write(f'Number of Observations: {n_obs}\n')
        f.write(f'Critical Values:\n')
        for key, value in crit_values.items():
            f.write(f'   {key}: {value}\n')
        f.write(f'IC Best: {icbest}\n')

    # Return apakah data stasioner atau tidak berdasarkan p-value
    return p_value <= 0.05

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

# Fungsi untuk ARIMA manual
def manual_arima(series, p, d, q):
    diff_series = differencing(series, d)
    ar_series = autoregressive_component(diff_series, p)
    ma_series = moving_average_component(diff_series, q)
    arima_series = ar_series + ma_series
    return arima_series

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
    aic = len(y) * np.log(sse / len(y)) + 2 * k
    return aic

# Fungsi untuk mencari parameter ARIMA terbaik
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

# Fungsi untuk mencari parameter SARIMA terbaik
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
                                    y_pred = manual_sarima(data, p, d, q, P, D, Q, m)
                                    aic = calculate_aic(data[max(d, D * m):], y_pred[max(d, D * m):], p + q + P + Q)
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q, P, D, Q, m)
                                        best_model = y_pred
                                except:
                                    continue
    return best_params, best_model

# Fungsi untuk mencari parameter SARIMAX terbaik
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
                                    aic = calculate_aic(data[max(d, D * m):], y_pred[max(d, D * m):], p + q + P + Q + exog.shape[1])
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q, P, D, Q, m)
                                        best_model = y_pred
                                except:
                                    continue
    return best_params, best_model
@app.template_filter('rupiah')
def rupiah_format(value):
    if isinstance(value, (int, float)):
        return "Rp {:,.2f}".format(value).replace(',', '.')
    return value

@app.route('/')
def index():
    # Load data
    data = load_and_clean_data()

    # Generate exogen data following the same dates
    data_exog = pd.DataFrame({
        'Tanggal': data['Tanggal'],  # Ensure exogen data matches the dates in the main data
        'Curah Hujan': np.random.uniform(100, 130, size=len(data)),
        'Jumlah Produksi': np.random.uniform(1100, 1500, size=len(data))
    })

    # Uji ADF dan simpan hasilnya ke file teks
    uji_adf(data['Terakhir'], 'adf_result.txt')

    # Membaca isi file ADF hasil
    with open('adf_result.txt', 'r') as f:
        adf_results = f.read()

    # Coba memuat parameter terbaik dari file JSON
    best_params = load_best_params_from_file()

    # Coba memuat model prediksi dari file pickle
    model = load_model_from_file()

    if best_params is None or model is None:
        # Jika parameter terbaik atau model tidak ada, lakukan pencarian parameter terbaik dan pembuatan model
        p_values, d_values, q_values = range(0, 3), range(0, 3), range(0, 3)
        P_values, D_values, Q_values, m_values = range(0, 3), range(0, 3), range(0, 3), [6]

        # ARIMA
        best_params_arima, model_arima = find_best_arima_params(data['Terakhir'].values, p_values, d_values, q_values)

        # SARIMA
        best_params_sarima, model_sarima = find_best_sarima_params(data['Terakhir'].values, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)

        # SARIMAX menggunakan data eksogen yang sudah mengikuti tanggal
        exog = data_exog[['Curah Hujan', 'Jumlah Produksi']].values
        best_params_sarimax, model_sarimax = find_best_sarimax_params(data['Terakhir'].values, exog, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)

        # Simpan parameter terbaik dan model ke file
        save_best_params_to_file({
            'ARIMA': best_params_arima,
            'SARIMA': best_params_sarima,
            'SARIMAX': best_params_sarimax
        })
        save_model_to_file({
            'ARIMA': model_arima,
            'SARIMA': model_sarima,
            'SARIMAX': model_sarimax
        })

        model = {
            'ARIMA': model_arima,
            'SARIMA': model_sarima,
            'SARIMAX': model_sarimax
        }

    # Load prediksi dari model yang sudah disimpan atau yang baru dibuat
    data['MA'] = data['Terakhir'].rolling(window=5).mean()

    # Pastikan hasil prediksi dihasilkan dari model yang sudah diload
    data['ARIMA'] = model['ARIMA']  # Prediksi ARIMA
    data['SARIMA'] = model['SARIMA']  # Prediksi SARIMA
    data['SARIMAX'] = model['SARIMAX']  # Prediksi SARIMAX

    # Bulatkan nilai SARIMAX menjadi dua angka desimal
    data['SARIMAX'] = data['SARIMAX'].round(1)

    # Gabungkan data eksogen ke dalam data utama berdasarkan tanggal
    data = pd.merge(data, data_exog, on='Tanggal')

    # Simpan hasil prediksi ke file Excel
    save_predictions_to_excel(data)

    # Prepare Plotly figures for each model and compare them with Harga Close
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=data['Tanggal'], y=data['Terakhir'], mode='lines', name='Harga Close', line=dict(color='blue')))
    fig_arima.add_trace(go.Scatter(x=data['Tanggal'], y=data['ARIMA'], mode='lines', name='Prediksi ARIMA', line=dict(dash='dash', color='red')))
    fig_arima.update_layout(title='Perbandingan Harga Close dengan ARIMA', xaxis_title='Tanggal', yaxis_title='Harga')

    fig_sarima = go.Figure()
    fig_sarima.add_trace(go.Scatter(x=data['Tanggal'], y=data['Terakhir'], mode='lines', name='Harga Close', line=dict(color='blue')))
    fig_sarima.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMA'], mode='lines', name='Prediksi SARIMA', line=dict(dash='dash', color='green')))
    fig_sarima.update_layout(title='Perbandingan Harga Close dengan SARIMA', xaxis_title='Tanggal', yaxis_title='Harga')

    fig_sarimax = go.Figure()
    fig_sarimax.add_trace(go.Scatter(x=data['Tanggal'], y=data['Terakhir'], mode='lines', name='Harga Close', line=dict(color='blue')))
    fig_sarimax.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMAX'], mode='lines', name='Prediksi SARIMAX', line=dict(dash='dash', color='purple')))
    fig_sarimax.update_layout(title='Perbandingan Harga Close dengan SARIMAX', xaxis_title='Tanggal', yaxis_title='Harga')

    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=data['Tanggal'], y=data['Terakhir'], mode='lines', name='Harga Close', line=dict(color='blue')))
    fig_all.add_trace(go.Scatter(x=data['Tanggal'], y=data['ARIMA'], mode='lines', name='Prediksi ARIMA', line=dict(dash='dash', color='red')))
    fig_all.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMA'], mode='lines', name='Prediksi SARIMA', line=dict(dash='dash', color='green')))
    fig_all.add_trace(go.Scatter(x=data['Tanggal'], y=data['SARIMAX'], mode='lines', name='Prediksi SARIMAX', line=dict(dash='dash', color='purple')))
    fig_all.update_layout(title='Perbandingan Harga Close dengan Semua Model', xaxis_title='Tanggal', yaxis_title='Harga')

    # Convert Plotly figures to JSON for rendering in the template
    all_graph_json = json.dumps(fig_all, cls=plotly.utils.PlotlyJSONEncoder)
    arima_graph_json = json.dumps(fig_arima, cls=plotly.utils.PlotlyJSONEncoder)
    sarima_graph_json = json.dumps(fig_sarima, cls=plotly.utils.PlotlyJSONEncoder)
    sarimax_graph_json = json.dumps(fig_sarimax, cls=plotly.utils.PlotlyJSONEncoder)

    # Format dates as required
    data['Tanggal'] = data['Tanggal'].dt.strftime('%d-%m-%Y')

    return render_template('index.html',
                           tables=data.to_dict(orient='records'),
                           all_graph_json=all_graph_json,
                           arima_graph_json=arima_graph_json,
                           sarima_graph_json=sarima_graph_json,
                           sarimax_graph_json=sarimax_graph_json,
                           adf_results=adf_results)


@app.route('/reset')
def reset():
    # Define file paths to be deleted
    prediction_file = 'hasil_prediksi.xlsx'
    model_file = 'best_model.pkl'
    adf_file = 'adf_result.txt'
    best_params_file = 'best_params.json'
    
    # Delete the prediction file if it exists
    if os.path.exists(prediction_file):
        os.remove(prediction_file)

    # Delete the model file if it exists
    if os.path.exists(model_file):
        os.remove(model_file)

    # Delete the ADF result file if it exists
    if os.path.exists(adf_file):
        os.remove(adf_file)
    
    # Delete the best params file if it exists
    if os.path.exists(best_params_file):
        os.remove(best_params_file)

    # Redirect back to the index page, effectively reloading the page
    return redirect(url_for('index'))

@app.route('/open-prediction-excel')
def open_prediction_excel():
    # Path to the prediction Excel file
    prediction_file = os.path.abspath('hasil_prediksi.xlsx')

    # Check if the file exists and open it
    if os.path.exists(prediction_file):
        # Use subprocess to open the Excel file using the default Excel program
        subprocess.Popen(['start', 'excel', prediction_file], shell=True)
        return redirect(url_for('index'))
    else:
        return "Prediction Excel file not found", 404
    
@app.route('/open-csv-data')
def open_csv_data():
    # Path to the CSV file
    csv_file = os.path.abspath('data.csv')

    # Check if the file exists and open it
    if os.path.exists(csv_file):
        # Use subprocess to open the CSV file in Excel using the default Excel program
        subprocess.Popen(['start', 'excel', csv_file], shell=True)
        return redirect(url_for('index'))
    else:
        return "CSV file not found", 404

if __name__ == '__main__':
    app.run(debug=True)
