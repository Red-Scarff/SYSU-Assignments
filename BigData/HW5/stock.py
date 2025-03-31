import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# 参数设置
TICKER = "AAPL"  # 股票代码
START_DATE = "2018-01-01"
END_DATE = "2023-01-01"
FORECAST_DAYS = 7  # 预测天数
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# 修正后的数据下载函数
def download_data():
    file_path = "stock_data.csv"
    if not os.path.exists(file_path):
        data = yf.download(TICKER, start=START_DATE, end=END_DATE)
        data.to_csv(file_path)

    # 跳过无关行并正确指定列名
    df = pd.read_csv(file_path, skiprows=0, usecols=["Date", "Close"], index_col="Date", parse_dates=True)
    return df["Close"].dropna()


# 数据预处理
def prepare_data(data):
    train = data[:-FORECAST_DAYS]
    test = data[-FORECAST_DAYS:]
    print(f"测试集长度: {len(test)}, 预测步长: {FORECAST_DAYS}")
    return train, test


# 参数设置（新增ARIMA参数）
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2023-01-01"
FORECAST_DAYS = 7
ARIMA_ORDER = (2, 1, 2)  # 手动设置的ARIMA参数(p,d,q)
DIFF_ORDER = 1  # 差分阶数


def manual_arima_forecast(train, test):
    """手动实现ARIMA预测流程"""
    print("\nTraining Manual ARIMA Model...")

    # 强制平稳性处理（二阶差分）
    train_diff = train.diff(DIFF_ORDER).dropna()

    # 模型训练（使用statsmodels的SARIMAX实现）
    model = SARIMAX(train_diff, order=ARIMA_ORDER, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=0)

    # 生成预测（需处理差分还原）
    forecast_diff = model_fit.forecast(steps=FORECAST_DAYS)

    # 差分逆变换重建预测值
    last_value = train.iloc[-1]
    forecast = np.r_[last_value, forecast_diff].cumsum()[1:]
    print(forecast)

    return pd.Series(forecast, index=test.index)


# 修正后的LSTM模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=FORECAST_DAYS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        # 取最后一个时间步的输出
        return self.linear(x[:, -1, :])  # 输出维度 (batch_size, output_size)


# 修正后的数据序列生成
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - FORECAST_DAYS + 1):  # 修正索引
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + FORECAST_DAYS])
    return np.array(X), np.array(y)


# 修正后的LSTM训练流程
def lstm_forecast(train, test, window_size=60):
    print("\nTraining LSTM model...")

    # 标准化数据
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    # 生成序列数据
    X, y = create_sequences(train_scaled, window_size)

    # 调整维度
    X = X.reshape(-1, window_size, 1)  # (samples, window_size, features)
    y = y.reshape(-1, FORECAST_DAYS)  # (samples, forecast_days)

    # 转换为张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型配置
    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 100
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # 输入维度 (32,60,1)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # 预测
    last_seq = train_scaled[-window_size:].reshape(1, window_size, 1)
    with torch.no_grad():
        forecast_scaled = model(torch.FloatTensor(last_seq)).numpy()

    # 逆标准化
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    print(f"LSTM预测结果NaN数量: {np.isnan(forecast).sum()}")
    print("Forecast:\n", forecast)
    return pd.Series(forecast, index=test.index)


# 结果可视化
def plot_results(train, test, arima_pred, lstm_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(train[-60:], label="Training Data")
    plt.plot(test, label="Actual Price")
    plt.plot(test.index, arima_pred, label=f"ARIMA Forecast (MAE: {mean_absolute_error(test, arima_pred):.2f})")
    plt.plot(test.index, lstm_pred, label=f"LSTM Forecast (MAE: {mean_absolute_error(test, lstm_pred):.2f})")
    plt.title(f"{TICKER} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"{TICKER}_stock_price_prediction.png")


# 主程序
if __name__ == "__main__":
    # 数据准备
    data = download_data()
    train, test = prepare_data(data)
    result = adfuller(train)
    print(f"ADF p-value: {result[1]}")

    # ARIMA预测
    arima_pred = manual_arima_forecast(train, test)

    # LSTM预测
    lstm_pred = lstm_forecast(train, test)

    # 评估指标
    print("\nModel Comparison:")
    print(f"ARIMA MAE: {mean_absolute_error(test, arima_pred):.4f}")
    print(f"ARIMA RMSE: {np.sqrt(mean_squared_error(test, arima_pred)):.4f}")
    print(f"LSTM MAE: {mean_absolute_error(test, lstm_pred):.4f}")
    print(f"LSTM RMSE: {np.sqrt(mean_squared_error(test, lstm_pred)):.4f}")

    # 可视化结果
    plot_results(train, test, arima_pred, lstm_pred)
