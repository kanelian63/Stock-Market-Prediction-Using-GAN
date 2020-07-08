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
Advanced Micro Devices 사의 1980-03-17부터 2020-06-30까지의 총 10160일 동안의 주식 일봉 데이터를 FinanceDataReader을 이용하여 활용하였다.

논문에서 활용하는 Channel은 총 13개이고, 다음과 같다.

![indicators](https://user-images.githubusercontent.com/59387983/86939182-37883400-c17c-11ea-9c68-9a86d20fd9ee.PNG)

논문에서 활용한 지표 중, 몇개의 경우 구체적인 언급이 없었기 때문에 활용할 수 없었다.

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

Columns = ['Close','Open','High','Low','Volume','Change','bb_upper','bb_middle','bb_lower','dx','ema','slowk','slowd','ma5','macd','macdsignal','macdhist','rsi']

Sequence Length(time step)는 120일로 하였고 119일이 train_data, 120일째가 label이다. train_dataset의 batch는 7740이고, test_dataset의 batch는 1984이다.

# Training Point
1. 119일치의 데이터를 
논문에서 특이한 점은 일반적인 GAN의 Loss와 다르게 Generator Loss의 경우, Stock Data의 학습에 맞게 변경하였다.
1. 일반적인 Generator의 Loss이다.
![L_adv](https://user-images.githubusercontent.com/59387983/86940309-9ac69600-c17d-11ea-81d9-0270df012a6a.PNG)
2. Discriminator가 학습하는데 혼란을 주기 위한 Loss이다.
![Lp_loss](https://user-images.githubusercontent.com/59387983/86940306-9a2dff80-c17d-11ea-8d5b-03e7e666b33c.PNG)
3. Stock 가격의 방향성을 변수로 주기 위한 Loss이다.
![L_dpl](https://user-images.githubusercontent.com/59387983/86940301-99956900-c17d-11ea-97d3-ae257f337ec6.PNG)
4. Generator의 Loss
![L_G](https://user-images.githubusercontent.com/59387983/86940626-03157780-c17e-11ea-8a97-020a217f4a6b.PNG)
Lamda_adv, Lamda_p, and Lamda_dpl는 이전의 기울기 파라미터들로 정의된다고 논문에서는 언급되어있다. 하지만 구체적인 언급은 없어서 일단은 세가지 Loss의 크기를 균일하게 하는 방향으로 값을 설정하였다. 이 값들도 가중치 갱신으로 자동으로 모델에서 학습되게 해놓았다고 이해하고 있지만, 다음에 시간이 되면 하겠다.
5. 실제 하드코딩으로 구현한 Losses
![losses for generator](https://user-images.githubusercontent.com/59387983/86941199-a2d30580-c17e-11ea-86c8-68a2f572ee4d.PNG)


# Install Ta_lib
https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

ex) pip install TA_Lib‑0.4.18‑cp37‑cp37m‑win_amd64.whl

https://mrjbq7.github.io/ta-lib/

# P.s
개인적으로 주식 데이터를 다루는 마지막 Mini-Project일듯하다. 논문에서 설명하고 있는 모델, 로스, 하이퍼파라미터를 그대로 구현했음에도 제대로 된 학습이 되질 않는다. 그리고 모델의 학습에 필요한 몇가지 요소를 자세히 적어두지 않았다. Pytorch로 GAN과 LSTM을 처음으로 구현해보아서 인풋데이터와 하이퍼파라미터들의 차원에 대해서 많이 헤멨다. 고생했지만 눈문의 결과를 그대로 구현해본다는 목적으로 생각하자면 시간 낭비했다. 다른 사람이 구현시도를 해보지 않았을까 구글링을 해보았는데, 다른 사람들도 제대로 되지 않았다는 정보 밖에 없다.
