import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.rnn.model  import RNN
class Trainer_RNN:
    def __init__(self, model_name, config):
        """
        Initialize class

        :param config: configuration
        :type config: dictionary
        """
        
        # lsrm or gru
        self.model_name = model_name

        self.model = RNN(
            config['input_size'], 
            config['hidden_size'], 
            config['num_layers'], 
            config['bidirectional'], 
            self.model_name, 
            config['forecast_step'],
            config['device'])
        
        self.num_epochs = config['num_epochs']
        self.dropout = config['dropout']
        self.lr = config['lr']
        self.device = config['device']
        
        self.model = self.model.to(self.device)
        

    def fit(self, train_loader, valid_loader):
        """
        Train the model and return the best trained model

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: trained model
        :rtype: model
        """

        since = time.time()

        val_rmse_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_rmse = 10000000

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            print('-' * 10)

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 모델을 training mode로 설정
                    dataloader = train_loader
                else:
                    self.model.eval()   # 모델을 validation mode로 설정
                    dataloader = valid_loader
                running_loss = 0.0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = self.model(inputs)
                        loss = criterion(outputs.unsqueeze(2), labels)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_total += labels.size(0)

                # epoch의 loss 및 RMSE 도출
                epoch_loss = running_loss / running_total
                epoch_rmse = np.sqrt(running_loss / running_total)

                print('{} Loss: {:.4f} RMSE: {:.4f}'.format(phase, epoch_loss, epoch_rmse))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_rmse < best_rmse:
                    best_rmse = epoch_rmse
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_rmse_history.append(epoch_rmse)

            print()

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val RMSE: {:4f}'.format(best_rmse))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        self.model.load_state_dict(best_model_wts)
        
        return self.model
    
    def test(self, test_loader):
        """
        Predict future values based on the best trained model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array
        """

        self.model.eval()
        preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds)
        preds = preds.reshape(-1, 1)
        
        return preds