#%%
# The FinanceDataReader is financial data reader(crawler) for finance.
# The main functions are as follows.

# Stock Symbol listings: 'KRX' ('KOSPI', 'KODAQ', 'KONEX'), 'NASDAQ', 'NYSE', 'AMEX' and 'S&P500'
# KRX delistings: 'KRX-DELISTING'
# ETF Symbol listings: Support for ETF lists for multiple countries ('KR', 'US', 'JP')
# Stock price(KRX): '005930'(Samsung), '091990'(Celltrion Healthcare) ...
# Stock price(Word wide): 'AAPL', 'AMZN', 'GOOG' ... (you can specify exchange(market) and symbol)
# Indexes: 'KOSPI', 'KOSDAQ', 'DJI', 'IXIC', 'US500'(S&P 500) ...
# Exchanges: 'USD/KRX', 'USD/EUR', 'CNY/KRW' ...
# Cryptocurrency price data: 'BTC/USD' (Bitfinex), 'BTC/KRW' (Bithumb)

import FinanceDataReader as fdr
fdr.__version__

import matplotlib.pyplot as plt

# Technical Analysis Library
# TA-Lib is widely used by trading software developers requiring to perform technical analysis of financial market data.
import talib as ta
import pandas as pd
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.get_device_name(0)

# Default CUDA device
cuda = torch.device('cuda')
"""
# allocates a tensor on default GPU
a = torch.tensor([1., 2.], device=cuda)
 
# transfers a tensor from 'C'PU to 'G'PU
b = torch.tensor([1., 2.]).cuda()
 
# Same with .cuda()
b2 = torch.tensor([1., 2.]).to(device=cuda)
"""


df_spx = fdr.StockListing('S&P500')
df_spx.columns
# Index(['Symbol', 'Name', 'Sector', 'Industry'], dtype='object')
# 회사심볼, 회사이름, 산업군, 산업
len(df_spx)
# 505개의 회사

# 암당(AMD), 2018-01-01 ~ 2018-03-30
# reader = InvestingDailyReader
df_amd = fdr.DataReader('AMD', '1980-03-17', '2020-06-30')
df_amd.head()
# Close  Open  High   Low    Volume  Change
df_amd['Close'].plot()

# Hyperparameters
timeperiod = 120
# 하이퍼파라미터 몇가지를 잘 모르겠다.

# Turnover
# Stock Turnover = Cost of Goods Sold / Average Inventory, # 거래량 / 총 발행 주식수

# Bias ???

# Bollingerbands
df_amd_bb_upper, df_amd_bb_middle, df_amd_bb_lower = ta.BBANDS(df_amd['Close'], timeperiod=timeperiod)

# Directionalmovementindex
df_amd_dx = ta.DX(df_amd['Close'],df_amd['Open'],df_amd['High'], timeperiod=timeperiod)

# Exponentialmovingaverages
df_amd_ema = ta.EMA(df_amd['Close'], timeperiod=timeperiod)

# Stochasticindex
df_amd_slowk, df_amd_slowd = ta.STOCH(df_amd['High'], df_amd['Low'], df_amd['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# Movingaverages
df_amd_ma5 = ta.MA(df_amd['Close'], timeperiod=timeperiod)

# MACD
df_amd_macd, df_amd_macdsignal, df_amd_macdhist = ta.MACD(df_amd['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Relativestrengthindex
df_amd_rsi = ta.RSI(df_amd['Close'], timeperiod=timeperiod)

#%%
df_amd_total = df_amd.copy()

df_amd_total = pd.concat([df_amd_total,
                          df_amd_bb_upper, df_amd_bb_middle, df_amd_bb_lower,   # Bollingerbands
                          df_amd_dx,                                            # Directionalmovementindex
                          df_amd_ema,                                           # Exponentialmovingaverages
                          df_amd_slowk, df_amd_slowd,                           # Stochasticindex
                          df_amd_ma5,                                           # Movingaverages
                          df_amd_macd, df_amd_macdsignal, df_amd_macdhist,      # MACD
                          df_amd_rsi], axis = 1)                                # Relativestrengthindex                

df_amd_total.columns = ['Close','Open','High','Low','Volume','Change','bb_upper','bb_middle','bb_lower','dx','ema','slowk','slowd','ma5','macd','macdsignal','macdhist','rsi']

df_amd_total.shape
# (10160, 18)
# indicator data가 없는 행 삭제
df_amd_total = df_amd_total.dropna(axis = 0, how = 'any')

df_amd_total.shape
# (9795, 18)

# 정규화
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))

df_amd_total_scaled = min_max_scaler.fit_transform(df_amd_total)

df_amd_total_scaled.shape
# (9795, 18)

pd.DataFrame(df_amd_total_scaled[:,0]).plot()

# scaling test
# x_input = df_amd_total['Close'].values.reshape(0, 1)
# x_test = min_max_scaler.fit_transform(x_input)

#%%

# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

look_back = 120 # choose sequence length

x_train, y_train, x_test, y_test = load_data(df_amd_total_scaled, look_back)

print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

"""
**input** of shape `(seq_len, batch, input_size)`: tensor containing the features of the input sequence.
"""

# (배치, 시퀀스 랭스, 피쳐갯수)
# 배치 : 데이터의 총 갯수
# 시퀀스 랭스 : 데이터 한개에 속해있는 날짜의 갯수
# 피쳐갯수 : 데이터의 피쳐갯수
# input_dim, hidden_dim, num_layers, output_dim

# make training and test sets in torch

x_train = torch.from_numpy(x_train).type(torch.Tensor).cuda()
y_train = torch.from_numpy(y_train).type(torch.Tensor).cuda()
x_test = torch.from_numpy(x_test).type(torch.Tensor).cuda()
y_test = torch.from_numpy(y_test).type(torch.Tensor).cuda()

x_train.size()
y_train.size()

# torch.Size([7740, 364, 18])
# torch.Size([7740, 18])

#%%
# Rolling Segmentation Implementation
# 나중에

#%%

batch_size = 7936

train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)



num_epochs = 1000 #n_iters / (len(train_X) / batch_size)
#num_epochs = int(num_epochs)

# Build model
#####################
input_dim = 18
hidden_dim = 32
num_layers = 2
output_dim = 18

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).cuda()

loss_fn = torch.nn.MSELoss(size_average=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


#%%
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()
    
    # Forward pass
    y_train_pred = model(x_train).cuda()

    loss = loss_fn(y_train_pred, y_train).cuda()
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()
#%%

y_train_pred = y_train_pred.cpu()
y_train = y_train.cpu()
plt.plot(y_train_pred[0].detach().numpy(), label="Preds")
plt.plot(y_train[0].detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()
#%%

y_train_pred_invers = min_max_scaler.inverse_transform(y_train_pred.detach().numpy())
y_train_invers = min_max_scaler.inverse_transform(y_train.detach().numpy())

plt.plot(y_train_pred_invers[:,0], label="Preds")
plt.plot(y_train_invers[:,0], label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

#%%
# make predictions
y_test_pred = model(x_test).cuda()
y_test_pred = y_test_pred.cpu()
x_test = x_test.cpu()
y_test = y_test.cpu()
len(y_test)
# invert predictions
y_test_pred_invers = min_max_scaler.inverse_transform(y_test_pred.detach().numpy())
y_test_invers = min_max_scaler.inverse_transform(y_test.detach().numpy())


for i in range(18):
    
    plt.plot(y_test_pred_invers[:,i], label="Preds")
    plt.plot(y_test_invers[:,i], label="Data")
    plt.legend()
    plt.show()



from sklearn.metrics import mean_squared_error
import math
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#%%
# shift train predictions for plotting
trainPredictPlot = np.empty_like(df_ibm)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(y_train_pred)+look_back, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(df_ibm)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+look_back-1:len(df_ibm)-1, :] = y_test_pred

# plot baseline and predictions
plt.figure(figsize=(15,8))
plt.plot(scaler.inverse_transform(df_ibm))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
