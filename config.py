model_config = {
    "lstm": {  # Case 1. model = lstm
        "model": 'lstm',
        "best_model_path": './ckpt/lstm.pt',  # 학습 완료 모델 저장 경로
        "parameter": {
            "input_size" : 1,  # 데이터 변수 개수, int
            "window_size" : 48,  # input sequence의 길이, int
            "forecast_step" : 1,  # 예측할 미래 시점의 길이, int
            "num_layers" : 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
            "hidden_size" : 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
            "dropout" : 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            "bidirectional" : True,  # 모델의 양방향성 여부, bool(default: True)
            "num_epochs" : 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            "lr" : 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
        }
    },
    'gru': {  # Case 2. model = gru
        "model": 'gru',
        "best_model_path": './ckpt/gru.pt',  # 학습 완료 모델 저장 경로
        "parameter": {
            "input_size" : 1,  # 데이터 변수 개수, int
            "window_size" : 48,  # input sequence의 길이, int
            "forecast_step" : 1,  # 예측할 미래 시점의 길이, int
            "num_layers" : 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
            "hidden_size" : 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
            "dropout" : 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            "bidirectional" : True,  # 모델의 양방향성 여부, bool(default: True)
            "num_epochs" : 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            "lr" : 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
        }
    },
    'informer': {  # Case 3. model = informer
        "model": 'informer',
        "best_model_path": './ckpt/informer.pt',  # 학습 완료 모델 저장 경로
        "parameter": {
            "input_size" : 1,  # 데이터 변수 개수, int
            "window_size" : 48,  # input sequence의 길이, int
            "forecast_step" : 1,  # 예측할 미래 시점의 길이, int
            "label_len" : 12,  # Decoder의 start token 길이, int(default: 12)
            "d_model" : 512,  # 모델의 hidden dimension, int(default: 512)
            "e_layers" : 2,  # encoder layer 수, int(default: 2)
            "d_layers" : 1,  # decoder layer 수, int(default: 1)
            "d_ff" : 2048,  # fully connected layer의 hidden dimension, int(default: 2048)
            "factor" : 5, # 모델의 ProbSparse Attention factor, int(default: 5)
            "dropout" : 0.05, # dropout ratio, int(default: 0.05)
            "attn" : 'prob', # 모델의 attention 계산 방식, (default: 'prob', ['prob', 'full'] 중 선택)
            "n_heads" : 8, # multi-head attention head 수, int(default: 8)
            "embed" : 'timeF', # time features encoding 방식, (default: 'timeF', ['timeF', 'fixed', 'learned'] 중 선택)
            "num_epochs" : 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            "lr" : 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            "lradj" : 'type1', # learning rate 조정 방식, (default: 'type1', ['type1', 'type2'] 중 선택)
            "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)   
        }
    },
    'scinet': {  # Case 4. model = scinet
        "model": 'scinet',
        "best_model_path": './ckpt/scinet.pt',  # 학습 완료 모델 저장 경로
        "parameter": {
            "input_size" : 1,  # 데이터 변수 개수, int
            "window_size" : 48,  # input sequence의 길이, int
            "forecast_step" : 1,  # 예측할 미래 시점의 길이, int
            "levels" : 2, # Tree의 depth, int(default: 2, 범위: input sequence의 로그 값 이하, 2~4 설정 권장)
            "stacks" : 1, # SCINet 구조를 쌓는 횟수, int(default: 1, 범위: 3 이하)
            "num_epochs" : 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            "lr" : 0.00005,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
        }
    }
}
