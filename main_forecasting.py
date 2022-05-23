import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error 

from models.rnn.trainer import Trainer_RNN
from models.informer.trainer import Trainer_Informer
from models.scinet.trainer import Trainer_SCINet


class Forecasting():
    def __init__(self, config, train_data, test_data, test_date):
        """
        Initialize Forecasting class

        :param config: config
        :type config: dictionary

        :param train_data: train data whose shape is (# time steps, )
        :type train_data: numpy array

        :param test_data: test data whose shape is (# time steps, )
        :type test_data: numpy array

        :param test_date: test date whose shape is (# time steps, )
        :type test_date: numpy array

        example
            >>> config1 = {
                    "model": 'lstm',
                    "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
                    "best_model_path": './ckpt/lstm.pt',  # 학습 완료 모델 저장 경로
                    "parameter": {
                        "input_size" : 1,  # 데이터 변수 개수, int
                        "window_size" : 48,  # input sequence의 길이, int
                        "forecast_step" : 24,  # 예측할 미래 시점의 길이, int
                        "num_layers" : 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
                        "hidden_size" : 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
                        "dropout" : 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
                        "bidirectional" : True,  # 모델의 양방향성 여부, bool(default: True)
                        "num_epochs" : 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
                        "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
                        "lr" : 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
                        "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
                    }
                }
            >>> data_forecast = mf.Forecasting(config, train_data, test_data, test_date)
            >>> init_model = data_forecast.build_model()  # 모델 구축
            >>> if config["training"]:
            >>>     best_model = data_forecast.train_model(init_model)  # 모델 학습
            >>>     data_forecast.save_model(best_model, best_model_path=config["best_model_path"])  # 모델 저장
            >>> pred, mse, mae = data_forecast.pred_data(init_model, best_model_path=config["best_model_path"])  # 예측
        """

        self.test_data = test_data
        self.test_date = test_date
        
        self.model_name = config['model']
        self.parameter = config['parameter']

        self.train_loader, self.valid_loader, self.test_loader, self.scaler = self.get_loaders(
            train_data, test_data, self.parameter['window_size'],
            self.parameter['forecast_step'], self.parameter['batch_size']
            )

    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model_name == 'lstm':
            model = Trainer_RNN(self.model_name, self.parameter)
        elif self.model_name == 'gru':
            model = Trainer_RNN(self.model_name, self.parameter)
        elif self.model_name == 'informer':
            model = Trainer_Informer(self.parameter)
        elif self.model_name == 'scinet':
            model = Trainer_SCINet(self.parameter)
        return model

    def train_model(self, init_model):
        """
        Train model and return best model

        :param init_model: initialized model
        :type init_model: model

        :return: best trained model
        :rtype: model
        """

        print("Start training model\n")

        # train model
        best_model = init_model.fit(self.train_loader, self.valid_loader)
        return best_model

    def save_model(self, best_model, best_model_path):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        :param best_model_path: path for saving model
        :type best_model_path: str
        """

        # save model
        torch.save(best_model.state_dict(), best_model_path)

    def pred_data(self, init_model, best_model_path):
        """
        Predict future data based on the best trained model
        :param init_model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted values with date
        :rtype: DataFrame

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        print("Start testing data\n")

        # load best model
        init_model.model.load_state_dict(torch.load(best_model_path))

        # get prediction results
        # the number of prediction data = forecast_step * ((len(test_data)-window_size-forecast_step) // forecast_step + 1)
        # start time point of prediction = window_size
        # end time point of prediction = len(test_data) - (len(test_data)-window_size-forecast_step) % forecast_step - 1
        pred_data = self.trainer.test(init_model, self.test_loader)  # shape=(the number of prediction data, 1)

        # inverse normalization
        pred_data = self.scaler.inverse_transform(pred_data)
        pred_data = pred_data.squeeze(-1)  # shape=(the number of prediction data, )

        # select time index for prediction data
        start_idx = self.parameter['window_size']
        end_idx = len(self.test_data) - (len(self.test_data)-self.parameter['window_size']-self.parameter['forecast_step']) % self.parameter['forecast_step'] - 1
        test_date = self.test_date[start_idx:end_idx+1]

        # merge prediction data with time index
        pred_df = pd.DataFrame()
        pred_df['date'] = test_date
        pred_df['predicted_value'] = pred_data

        # calculate performance metrics
        true_data = self.test_data[start_idx:end_idx+1]
        mse = mean_squared_error(true_data, pred_data)
        mae = mean_absolute_error(true_data, pred_data)
        return pred_df, mse, mae

    def get_loaders(self, train_data, test_data, window_size, forecast_step, batch_size):
        """
        Get train, validation, and test DataLoaders

        :param train_data: train data whose shape is (# time steps, )
        :type train_data: numpy array

        :param test_data: test data whose shape is (# time steps, )
        :type test_data: numpy array

        :param window_size: window size
        :type window_size: int

        :param forecast_step: forecast step size
        :type forecast_step: int
        
        :param batch_size: batch size
        :type batch_size: int

        :return: train, validation, and test dataloaders
        :rtype: DataLoader
        """
        
        # shape 변환 for normalization: shape=(# time steps, 1)
        train_data = np.expand_dims(train_data, axis=-1)
        test_data = np.expand_dims(test_data, axis=-1)

        # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
        train_data, valid_data = train_test_split(train_data, test_size=0.2, shuffle=False)

        # normalization
        scaler = StandardScaler()
        scaler = scaler.fit(train_data)
        
        train_data = scaler.transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)

        # train/validation/test 데이터를 기반으로 window_size 길이의 input으로 forecast_step 시점 만큼 미래의 데이터를 예측하는 데이터 생성
        datasets = []
        for dataset in [train_data, valid_data, test_data]:
            T = dataset.shape[0]

            # 전체 데이터를 forecast_step 크기의 sliding window 방식으로 window_size 크기의 time window로 분할하여 input 생성
            windows = [dataset[i : i+window_size] for i in range(0, T-window_size-forecast_step, forecast_step)]

            # input time window에 대하여 forecast_step 시점 만큼 미래의 데이터를 도출하여 예측 time window 생성
            targets = [dataset[i+window_size : i+window_size+forecast_step] for i in range(0, T-window_size-forecast_step, forecast_step)]

            datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets)))

        # train/validation/test DataLoader 구축
        # windows: shape=(batch_size, window_size, 1) & targets: shape=(batch_size, forecast_step, 1)
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader, scaler
