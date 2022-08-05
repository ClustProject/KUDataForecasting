import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error 

from models.rnn.trainer import Trainer_RNN
from models.informer.trainer import Trainer_Informer
from models.scinet.trainer import Trainer_SCINet


class Forecasting():
    def __init__(self, config):
        """
        Initialize Forecasting class

        :param config: config
        :type config: dictionary

        example (training)
            >>> model_name = 'lstm'
            >>> model_params = config.model_config[model_name]
            >>> data_forecast = mf.Forecasting(model_params)
            >>> best_model = data_forecast.train_model(train_data, valid_data)  # 모델 학습
            >>> data_forecast.save_model(best_model, best_model_path=model_params["best_model_path"])  # 모델 저장
        
        example (testing)
            >>> model_name = 'lstm'
            >>> model_params = config.model_config[model_name]
            >>> data_forecast = mf.Forecasting(model_params)
            >>> pred, mse, mae = data_forecast.pred_data(test_data, scaler, best_model_path=model_params["best_model_path"])  # 예측
        """

        self.model_name = config['model']
        self.parameter = config['parameter']

    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model_name == 'lstm':
            model = Trainer_RNN(self.parameter, model_name='lstm')
        elif self.model_name == 'gru':
            model = Trainer_RNN(self.parameter, model_name='gru')
        elif self.model_name == 'informer':
            model = Trainer_Informer(self.parameter)
        elif self.model_name == 'scinet':
            model = Trainer_SCINet(self.parameter)
        return model

    def train_model(self, train_data, valid_data):
        """
        Train model and return best model

        :param train_data: train data whose shape is (# time steps, 1)
        :type train_data: numpy array

        :param valid_data: validation data whose shape is (# time steps, 1)
        :type valid_data: numpy array

        :return: best trained model
        :rtype: model
        """

        print(f"Start training model: {self.model_name}")

        # build train/validation dataloaders
        train_loader = self.get_dataloader(train_data, self.parameter['window_size'],
            self.parameter['forecast_step'], self.parameter['batch_size'], shuffle=True)
        valid_loader = self.get_dataloader(valid_data, self.parameter['window_size'],
            self.parameter['forecast_step'], self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()

        # train model
        best_model = init_model.fit(train_loader, valid_loader)
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

    def pred_data(self, test_data, scaler, best_model_path):
        """
        Predict future data for test dataset using the best trained model

        :param test_data: test data whose shape is (# time steps, 1)
        :type test_data: numpy array

        :param scaler: scaler fitted on train dataset
        :type: MinMaxScaler

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: true values and predicted values
        :rtype: DataFrame

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        print(f"Start testing model: {self.model_name}")

        # build test dataloader
        test_loader = self.get_dataloader(test_data, self.parameter['window_size'],
            self.parameter['forecast_step'], self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()

        # load best model
        init_model.model.load_state_dict(torch.load(best_model_path))

        # get prediction results
        # the number of predicted values = forecast_step * ((len(test_data)-window_size-forecast_step) // forecast_step + 1)
        # start time point of prediction = window_size
        # end time point of prediction = len(test_data) - (len(test_data)-window_size-forecast_step) % forecast_step - 1
        pred_data = init_model.test(test_loader)  # shape=(the number of predicted values, 1)
        
        # select true data whose times match that of predicted values
        start_idx = self.parameter['window_size']
        end_idx = len(test_data) - (len(test_data)-self.parameter['window_size']-self.parameter['forecast_step']) % self.parameter['forecast_step'] - 1
        true_data = test_data[start_idx:end_idx+1]

        # inverse normalization to original scale
        true_data = scaler.inverse_transform(np.expand_dims(true_data, axis=-1))
        pred_data = scaler.inverse_transform(pred_data)
        true_data = true_data.squeeze(-1)  # shape=(the number of predicted values, )
        pred_data = pred_data.squeeze(-1)  # shape=(the number of predicted values, )

        # calculate performance metrics
        mse = mean_squared_error(true_data, pred_data)
        mae = mean_absolute_error(true_data, pred_data)
        
        # merge true value and predicted value
        pred_df = pd.DataFrame()
        pred_df['actual_value'] = true_data
        pred_df['predicted_value'] = pred_data
        return pred_df, mse, mae

    def get_dataloader(self, dataset, window_size, forecast_step, batch_size, shuffle):
        """
        Get DataLoader

        :param dataset: data whose shape is (# time steps, )
        :type dataset: numpy array

        :param window_size: window size
        :type window_size: int

        :param forecast_step: forecast step size
        :type forecast_step: int
        
        :param batch_size: batch size
        :type batch_size: int

        :param shuffle: shuffle for making batch
        :type shuffle: bool

        :return: dataloader
        :rtype: DataLoader
        """

        # data dimension 확인 및 변환 => shape: (# time steps, 1)
        if len(dataset.shape) == 1:
            dataset = np.expand_dims(dataset, axis=-1)
            
        # input: window_size 길이의 시계열 데이터
        # 전체 데이터를 sliding window 방식(slide 크기=forecast_step)으로 window_size 길이의 time window로 분할하여 input 생성
        T = dataset.shape[0]
        windows = [dataset[i : i+window_size] for i in range(0, T-window_size-forecast_step+1, forecast_step)]

        # target: input의 마지막 시점 이후 forecast_step 시점만큼의 미래 데이터 (예측 정답)
        targets = [dataset[i+window_size : i+window_size+forecast_step] for i in range(0, T-window_size-forecast_step+1, forecast_step)]

        # torch dataset 구축
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets))

        # DataLoader 구축
        # windows: shape=(batch_size, window_size, 1) & targets: shape=(batch_size, forecast_step, 1)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
