import os
import re

import requests
import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from bs4 import BeautifulSoup

def cal_num_stock(url) :
    #상장 주식수 크롤링
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    items = soup.find_all('table', {"summary" : "시가총액 정보"})
    items = items[0].find_all("td")
    nums = re.findall("\d+", str(items[2]))
    num_stock = 0
    digits = len(nums) - 1
    for num in nums :
            num_stock += int(num) * 1000 ** digits
            digits -= 1

    return num_stock

def cal_bb(stock, w=20, k=2) :
    x = pd.Series(stock)
    mbb = x.rolling(w, min_periods=1).mean()
    ubb = mbb + k * x.rolling(w, min_periods=1).std()
    lbb = mbb - k * x.rolling(w, min_periods=1).std()

    return mbb, ubb, lbb

def cal_dmi(data, n=14, n_ADX=14) :
    #https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py : ADX
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= data.index[-1] :
        UpMove = data.loc[i + 1, "High"] - data.loc[i, "High"]
        DoMove = data.loc[i, "Low"] - data.loc[i+1, "Low"]
        if UpMove > DoMove and UpMove > 0 :
            UpD = UpMove
        else :
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0 :
            DoD = DoMove
        else :
            DoD = 0
        DoI.append(DoD)
        i = i + 1

    i = 0
    TR_l = [0]
    while i < data.index[-1]:
        TR = max(data.loc[i + 1, 'High'], data.loc[i, 'Close']) - min(data.loc[i + 1, 'Low'], data.loc[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=1).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=1).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=1).mean() / ATR)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=1).mean(),
                    name='ADX_' + str(n) + '_' + str(n_ADX))

    return PosDI, NegDI, ADX

def cal_ema_macd(data, n_fast=12, n_slow=26, n_signal=9) : 
    #https://wikidocs.net/3397
    data["EMAFast"] = data["Close"].ewm(span=n_fast, min_periods=1).mean()
    data["EMASlow"] = data["Close"].ewm(span=n_slow, min_periods=1).mean()
    data["MACD"] = data["EMAFast"] - data["EMASlow"]
    data["MACDSignal"] = data["MACD"].ewm(span=n_signal, min_periods=1).mean()
    data["MACDDiff"] = data["MACD"] - data["MACDSignal"]
    
    return data

def cal_rsi(data, N=14) :
    #https://wikidocs.net/3399
    U = np.where(data.diff(1) > 0, data.diff(1), 0)
    D = np.where(data.diff(1) < 0, data.diff(1) * (-1), 0)

    AU = pd.DataFrame(U).rolling(window=N, min_periods=1).mean()
    AD = pd.DataFrame(D).rolling(window=N, min_periods=1).mean()
    RSI = AU.div(AD+AU) * 100

    num_nan = np.sum(np.isnan(np.array(RSI[2:])))
    #if fd > 0 :
        #import pdb; pdb.set_trace()



    return RSI, num_nan

def cal_mv(data, N) :
    mv_n = data.rolling(window=N, min_periods=1).mean()

    return mv_n

def cal_kdjsi(data, N=12, M=5, T=5) :
    L = data["Low"].rolling(window=N, min_periods=1).min()
    H = data["High"].rolling(window=N, min_periods=1).max()

    k = ((data["Close"] - L) / (H - L)) * 100
    d = k.ewm(span=M).mean()
    j = d.ewm(span=T).mean()

    return k,d,j


path_dir = './data'
file_list = os.listdir(path_dir)

if not os.path.exists("D:\\AI\\Stock prediction with GAN\\pre_data") :
    os.makedirs("D:\\AI\\Stock prediction with GAN\\pre_data")

for item in file_list :

    if item.find("from") is not -1 :
        data = np.loadtxt(path_dir + '/' + item, delimiter = ',')
        data = pd.DataFrame(data)
        data.columns = ['Open', 'High', "Low", "Close", "Volumn", "Adj"]
        data = data[["Close", "Open", "High", "Low", "Volumn"]]
        
        #Turnover
        code = item[:6]
        url = "https://finance.naver.com/item/main.nhn?code={}".format(code)
        num_stock = cal_num_stock(url)
        data["Turnover"] = data["Volumn"] / num_stock

        #Bias --- ??
        #Bolinger bands  
        data["MBB"], data["HBB"], data["LBB"] = cal_bb(data["Close"])

        #moving averages
        data["MV_5"] = cal_mv(data["Close"], 5)
        data["MV_15"] = cal_mv(data["Close"], 15)
        data["MV_60"] = cal_mv(data["Close"], 60)

        #Exponential moving averages, MACD
        data = cal_ema_macd(data)

        #Stochastic index (stochastic oscillator)
        data["KDJ_K"], data["KDJ_D"], data["KDJ_J"] = cal_kdjsi(data)

        #Directional movement index
        data["PDI"],data["MDI"],data["ADX"] = cal_dmi(data)

        #Relative strength index
        data["RSI"], num_nan = cal_rsi(data["Close"])

        if num_nan > 1 :
            print("{} has nan".format(item))
            continue

        np.save("D:\\AI\\Stock prediction with GAN\\pre_data\\{}.npy".format(item[:-4]), data.to_numpy())