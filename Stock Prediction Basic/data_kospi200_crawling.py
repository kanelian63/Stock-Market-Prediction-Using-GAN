import os
import re
import requests

from bs4 import BeautifulSoup
import FinanceDataReader as fdr


BaseUrl = "http://finance.naver.com/sise/entryJongmok.nhn?&page="

date_from = '2010'
date_to = '2019'

ksp200_codes = []


for i in range(1, 22, 1) :
    try :
        url = BaseUrl + str(i)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        items = soup.find_all('td', {'class': 'ctg'})
        
        for item in items :
            txt = item.a.get('href')
            k = re.search('[\d]+', txt)
            if k :
                code = k.group()
                ksp200_codes.append(code)
    
    except :
        pass

if not os.path.exists("D:\\AI\\Stock prediction with GAN\\data") :
    os.makedirs("D:\\AI\\Stock prediction with GAN\\data")

n_saved_code = 0
for i, symbol in enumerate(ksp200_codes) :
    df = fdr.DataReader(symbol, date_from, date_to)
    if len(df) == 2220 :
        if df["Close"][0] > 4000 : 
            if df["Volume"].mean() > 40000 :
                if (len(df.iloc[:1349,5][df.iloc[:1349,5]>0.135]) +
                    len(df.iloc[:1349,5][df.iloc[:1349,5]<-0.135]) + 
                    len(df.iloc[1349:,5][df.iloc[1349:,5]>0.27]) + 
                    len(df.iloc[1349:,5][df.iloc[1349:,5]<-0.27])) < 7 : # 1349 : 2015-06-15 이후로 상한제 30%
                    df.to_csv(path_or_buf="D:\\AI\\Stock prediction with GAN\\data\\{}_from_{}.csv".format(symbol, date_from),index=False, header=False)
                    n_saved_code += 1
                    

    else :
        pass

print("{} stock data saved.".format(n_saved_code))
    #if i % 100 == 0 :
    #    print ("{}th stock data done.".format(i+1))
