# KUDataForecasting
- 원본 시계열 데이터를 입력으로 받아 미래 데이터 값을 예측하는 time series forecasting 대한 설명
- 입력 데이터 형태 : 단변량 시계열 데이터(univariate time-series data)
<br>

**Time series forecasting 사용 시, 설정해야하는 값**
* **model** : [‘lstm’, ‘gru’, ‘informer’, ‘scinet’] 중 'informer’ 선택
* **training** : 모델 학습 여부, [True, False] 중 선택, 학습 완료된 모델이 저장되어 있다면 False 선택
* **best_model_path** : 학습 완료된 모델을 저장할 경로

* **시계열 representation 모델 hyperparameter :** 아래에 자세히 설명.
  * LSTM hyperparameter 
  * GRU hyperparameter 

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

#### 4. SCINet
- **input_size** : 데이터 변수 개수, int(default : 1, 범위: 1 이상)
- **window_size** : input sequence의 길이, int(default : 48, 범위: 1 이상)
- **forecast_step** : 예측할 미래 시점의 길이, int(default: 24, 범위: 1 이상)
- **levels** : Tree의 depth, int(default: 2, 범위: input sequence의 로그 값 이하, 2~4 설정 권장)
- **stacks** : SCINet 구조를 중첩하는 횟수 (default : 1, stack 값은 1을 고정적으로 사용)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
- **SCI-Block hyper-parameter** : 해당 하는 parameter 는 고정된 값을 사용함 
- **Hidden-size** : 개별 1D-CNN Hidden channel 수 (default :1)
- **Kernel** : 1D-CNN 의 kernel size (default :3)
- **Dilation** : 1D-CNN 의 dilation size (default :1)
