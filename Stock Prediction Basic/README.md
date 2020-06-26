# Stock_Prediction_Basic

# 필요한 Library
  - Tensorflow 2.0 이상
  - Numpy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - tqdm
  데이터 수집과 Preprocess 용
    - Requests
    - FinanceDataReader
    - BeautifulSoup
    
    
# 사용 방법

## 데이터 수집

python data_kospi200_crawling.py

## 전처리 - 기술적 지표 생성

python preprocess.py

## 훈련

python train.py


# HyperParameter
- Batch size, epoch, learning_rate, 데이터 셋 크기 등등
- size_stocks : 훈련시에 사용하는 주식 종목의 갯수(단일 or 복수) - 아래 데이터 처리 방식 참고
- loss plot : epoch 진행에 따른 loss 확인
- task : 분류 or 회귀 
- savehistory : 모델 훈련이 끝난 뒤, confusion matrix 확인 및 저장 (분류 모델에서만 작동)
- gpu : 사용유무에 따라서 설정해주기.




# 데이터 처리 방식
## Data Rolling Segmentation (https://www.hindawi.com/journals/mpe/2018/4907423/) 참고
- 모델을 훈련시킬 때, 단순히 train/(val)/test set으로 split 하는 것이 아니라, 전체 데이터셋을 시간순에 따라서 이동시키면서 훈련시키는 방식.
- 짧은 시간의 거래일수록 주식 시장의 트렌드에 영향을 많이 받을 것임. 전문적인 트레이더들도 수익을 내기 위해서 그때 그때마다의 트렌드를 포착해야한다. 이런 트렌드를 모델링 하기 위해서 도입한 방식. 또 생각해보면 train / test split 방식으로 훈련을 진행하게 되면 test 셋의 기간이 길어질 수록 실제 예측하는 현재와 train set의 기간 괴리가 커짐. 이러한 기간의 괴리를 줄일 수 있는 방법이기도 함.

## 한개의 주식 vs 복수의 주식 (NSE Stock Market Prediction Using Deep-Learning Models(논문) 참고)
- A라는 주식을 예측하기 위해서 모델을 만든다면, A의 데이터를 통해 훈련한 모델을 사용하는 것이 일반적. B를 예측하기 위해서는 B 데이터를 또 훈련시켜야함. 참고 논문에서는 A라는 주식을 아주 잘 예측하는 모델을 만들면, 그 모델을 통해 B, C 등 다른 주식도 예측할 수 있다고 함. 서로 다른 주식 종목들이 Inner Dynamics를 공유하기 때문. 그래서 애초에 훈련 시킬때 모든 데이터를 사용하는것도 괜찮을까? 라는 생각으로 훈련때부터 모든 서로 다른 주식 종목을 이용하도록 했음. 
- 일단 위 코드 상으로는 결과가 안좋음. 하나의 주식을 훈련시킬때랑, 복수의 주식을 훈련시킬 때 고려해야 할 사항이 훨씬 늘어나기 떄문이라고 생각함. 정규화 방식 이라던가, batch 안에 하나의 주식을 넣을지 서로 다른 주식을 넣어도 되는지, 그러면 그 다음 batch는 어떻게 해야할지...
