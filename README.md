# KUDataForecasting
- 원본 시계열 데이터를 입력으로 받아 미래 데이터 값을 예측하는 time series forecasting 대한 설명
- 입력 데이터 형태 : 단변량 시계열 데이터(univariate time-series data)
<br>

**Time series forecasting 사용 시, 설정해야하는 값**
* **model** : [‘lstm’, ‘gru’, ‘informer’, ‘scinet’] 중 선택
* **training** : 모델 학습 여부, [True, False] 중 선택, 학습 완료된 모델이 저장되어 있다면 False 선택
* **best_model_path** : 학습 완료된 모델을 저장할 경로

* **시계열 representation 모델 hyperparameter :** 아래에 자세히 설명.
  * LSTM hyperparameter 
  * GRU hyperparameter 
  * Informer hyperparameter

<br>

#### 시계열 forecasting 모델 hyperparameter <br>

#### 1. LSTM
- **input_size** : 데이터 변수 개수, int(default : 1, 범위: 1 이상)
- **window_size** : input sequence의 길이, int(default : 48, 범위: 1 이상)
- **forecast_step** : 예측할 미래 시점의 길이, int(default: 24, 범위: 1 이상)
- **num_layers** : recurrent layers의 수, int(default: 2, 범위: 1 이상)
- **hidden_size** : hidden state의 차원, int(default: 64, 범위: 1 이상)
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **bidirectional** : 모델의 양방향성 여부, bool(default: True)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)

<br>

#### 2. GRU
- **input_size** : 데이터 변수 개수, int(default : 1, 범위: 1 이상)
- **window_size** : input sequence의 길이, int(default : 48, 범위: 1 이상)
- **forecast_step** : 예측할 미래 시점의 길이, int(default: 24, 범위: 1 이상)
- **num_layers** : recurrent layers의 수, int(default: 2, 범위: 1 이상)
- **hidden_size** : hidden state의 차원, int(default: 64, 범위: 1 이상)
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **bidirectional** : 모델의 양방향성 여부, bool(default: True)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

#### 2. Informer
- **input_size** : 데이터 변수 개수, int(default : 1, 범위: 1 이상)
- **window_size** : input sequence의 길이, int(default : 48, 범위: 1 이상)
- **forecast_step** : 예측할 미래 시점의 길이, int(default: 24, 범위: 1 이상)
- **label_len** : Decoder의 start token 길이, int(default: 12, 범위: 1 이상)
- **d_model** : 모델의 hidden dimension, int(default: 512, 범위: 1 이상)
- **e_layers** : encoder layer 수, int(default: 2, 범위: 1 이상)
- **d_layers** : decoder layer 수, int(default: 1, 범위: 1 이상)
- **d_ff** : fully connected layer의 hidden dimension, int(default: 2048, 범위: 1 이상)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
- **factor** : 모델의 ProbSparse Attention factor, int(default: 5)
- **dropout** : dropout ratio, int(default: 0.05)
- **attn** : 모델의 attention 계산 방식, (default: 'prob', ['prob', 'full'] 중 선택)
- **n_heads** : multi-head attention head 수, int(default: 8)
- **embed** : time features encoding 방식, (default: 'timeF', ['timeF', 'fixed', 'learned'] 중 선택)
- **lradj** : learning rate 조정 방식, (default: 'type1', ['type1', 'type2'] 중 선택)

<br>



