# Stock-Market-Prediction-Using-GAN
Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets

# Architecture
![Architecture](https://user-images.githubusercontent.com/59387983/86938770-be88dc80-c17b-11ea-816b-17bd62675660.PNG)

# Models
총 3가지의 모델을 활용하였으며, Sequential Data를 다루기 위한 LSTM과 1D CNN, 그리고 LSTM을 학습시키기 위해 GAN을 활용하였다.

각 모델의 사용법 및 특징은 Pytorch Tutorial로 대신 하겠다.

# 1D CNN
https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
# GAN
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# LSTM
https://pytorch.org/docs/master/generated/torch.nn.LSTM.html

# Data
Advanced Micro Devices 사의 1980-03-17부터 2020-06-30까지의 총 10160일 동안의 주식 일봉 데이터를 활용하였다.

논문에서 활용하는 Channel은 총 13개이고, 다음과 같다.

![indicators](https://user-images.githubusercontent.com/59387983/86938775-c052a000-c17b-11ea-9acf-c77a624abdd2.PNG)

논문에서 활용한 지표 중, 몇개의 경우 구체적인 언급이 없어 따로 하이퍼파라미터를 찾아서 입력해주었다.

따라서 구글링을 통해서 주로 사용되는 신뢰성 있는 지표들을 찾아 FinanceDataReader 패키지를 활용하여 출력하였다.

기본적인 5개의 지표를 제외하고 활용한 지표는 다음과 같다.

1. Bollingerbands
2. Directionalmovementindex
3. Exponentialmovingaverages
4. Stochasticindex
5. Movingaverages
6. MACD
7. Relativestrengthindex

다수의 데이터를 출력하는 지표가 있기에 출력하여 활용한 Channel은 총 18개이며 다음과 같다.
['Close','Open','High','Low','Volume','Change','bb_upper','bb_middle','bb_lower','dx','ema','slowk','slowd','ma5','macd','macdsignal','macdhist','rsi']






Sequence Length(time step)는 120일로 하였고, train_dataset의 batch는 7740이고, test_dataset의 batch는 1984이다.



# Install Ta_lib
https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

ex) pip install TA_Lib‑0.4.18‑cp37‑cp37m‑win_amd64.whl

https://mrjbq7.github.io/ta-lib/

# P.s
개인적으로 주식 데이터를 다루는 마지막 Mini-Project일듯하다. 논문에서 설명하고 있는 모델, 로스, 하이퍼파라미터를 그대로 구현했음에도 제대로 된 학습이 되질 않는다. 그리고 모델의 학습에 필요한 몇가지 요소를 자세히 적어두지 않았다. Pytorch로 GAN과 LSTM을 처음으로 구현해보아서 인풋데이터와 하이퍼파라미터들의 차원에 대해서 많이 헤멨다. 고생했지만 눈문의 결과를 그대로 구현해본다는 목적으로 생각하자면 시간 낭비했다. 다른 사람이 구현시도를 해보지 않았을까 구글링을 해보았는데, 다른 사람들도 제대로 되지 않았다는 정보 밖에 없다.
