#%%
###################################################################################################

# gan model below

####################################################################################
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

df_amd_total.head()

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
    
    x_test = data[train_set_size:,:-1,:]
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


x_test.size()
y_test.size()

#%%
# Hyperparameters

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

"""
From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02.
The weights_init function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to meet this criteria.
This function is applied to the models immediately after initialization.
"""
# 논문에 초기 가중치를 어떻게 하라는 구체적인 설명이 없어서 DCGAN의 초기 가중치를 가져옴
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#%%

# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        
        return out



input_dim = 18
hidden_dim = 32
num_layers = 2
output_dim = 18

# Create the generator
generator = Generator(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
generator.apply(weights_init)

# Print the model
print(generator)

#%%
# 데이터의형태
# input_size = (N,C(in),L)
# N = batch_size
# C(in) = a number of channel
# L = length of single sequence

"""
# batch, channel in, 높이, 너비
m = nn.Conv1d(18, 32, kernel_size = 2, stride=1)
input = torch.randn(1, 18, 7936)
input.size()
tput = m(input)
output.size()
"""

# Descriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.conv1d_1 = nn.Conv1d(in_channels=18, out_channels=256, kernel_size=5, stride=2)
        self.bn_1 = nn.BatchNorm1d(256)
        self.relu_1 = nn.ReLU()
        
        self.conv1d_2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
        self.bn_2 = nn.BatchNorm1d(512)
        self.relu_2 = nn.ReLU()

        self.conv1d_3 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, stride=2)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.relu_3 = nn.ReLU()

        self.linear_1 = nn.Linear(12288, 512)
        self.dr_1 = nn.Dropout(p=0.1)
        self.linear_2 = nn.Linear(512, 1)

        self.ac_1 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv1d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv1d_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        x = x.view(x.shape[0], -1)
        
        x = self.linear_1(x)
        x = self.dr_1(x)
        x = self.linear_2(x)
        
        x = self.ac_1(x)
        
        return x

in_channels = 1
out_channels = 18
kernel_size = 120

# Create the Discriminator
discriminator = Discriminator(in_channels, out_channels, kernel_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
discriminator.apply(weights_init)

# Print the model
print(discriminator)
#%%
# Initialize BCELoss function
adversarial_loss  = nn.BCELoss()

# Establish convention for real and fake labels during training
lr = 0.001
beta1 = 0.9

# Setup Adam optimizers for both G and D
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


G_losses = []
D_losses = []
D_losses_sum = []

G_data = []


#%%
num_epochs = 1

lamda_adv = 0.01
lamda_p = 0.8
lamda_dpl = 0.8

fake_label = torch.tensor([0], dtype=torch.float).cuda()
real_label = torch.tensor([1], dtype=torch.float).cuda()

for epoch in range(num_epochs):
    for i in range(len(x_train)):
        
    # train G
        # gradient 0
        generator.zero_grad()
        
        # product fake_data for d
        generated_data = generator(x_train[i].reshape(1,119,18))
        fake_data = torch.cat((x_train[i], generated_data), axis=0).reshape(1,18,120)
        
        # train d with fake_data
        output_fake_d = discriminator(fake_data)
        
        loss_g_adv = adversarial_loss(output_fake_d, real_label)
        
        L_p = torch.norm(generated_data.reshape(18) - y_train[i])
                
        L_dpl = (torch.sign(generated_data - x_train[i].reshape(1,119,18)[0][-1])
                 - torch.sign(y_train[i] - x_train[i].reshape(1,119,18)[0][-1])).mean().abs()
        
        # Loss G
        loss_g = lamda_adv*loss_g_adv + lamda_p*L_p + lamda_dpl*L_dpl
        
        loss_g.backward()
        
        loss_g_average = loss_g.mean().item()
        # update weight
        optimizer_g.step()
        
    # train D
        # gradient 0
        discriminator.zero_grad()

        # train d with real data
        real_data = torch.cat((x_train[i],y_train[i].reshape(1,18)),
                              axis=0).reshape(1,18,120)

        output_real_d = discriminator(real_data)  

        # Loss D with real_data
        # save gradient d with real_data
        loss_real_d = adversarial_loss(output_real_d, real_label)
        
        loss_real_d.backward()
        
        # train d with fake data
        # product real fake_data
        generated_data = generator(x_train[i].reshape(1,119,18))
        fake_data = torch.cat((x_train[i], generated_data),
                              axis=0).reshape(1,18,120)
        
        # Loss D with fake_data
        output_fake_d = discriminator(fake_data)
        loss_fake_d = adversarial_loss(output_fake_d, fake_label)
        
        # save gradient d with fake_data
        loss_fake_d.backward()
        
        loss_fake_d_average = loss_fake_d.mean().item()
        
        # sum of Loss D
        loss_sum = loss_real_d + loss_fake_d
        
        loss_d_sum = loss_sum.mean().item()
        
        # gradient update
        optimizer_d.step()
        
        # Output training stats
        print('%d %d, %f, %f'
              % (epoch, num_epochs, loss_fake_d_average, loss_g))

        # Save Losses for plotting later
        D_losses.append(loss_fake_d_average)
        G_losses.append(loss_g_average)

#%%

import pandas as pd
D_losses = pd.DataFrame(D_losses)
G_losses = pd.DataFrame(G_losses)


D_losses.plot()
G_losses.plot()

#%%

test_g = []

for i in range(len(x_test)):
    generated_data = generator(x_test[i].reshape(1,119,18))
    test_g.append(generated_data)

len(test_g)


x = np.zeros(shape=(3969,18))

for i in range(len(test_g)):
    a = test_g[i].cpu().detach().numpy()[0]

    x[i] = a


dds = pd.DataFrame(x[:,0])
dds.plot()


