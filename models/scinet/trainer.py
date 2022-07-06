import os
import time
import copy
import torch.nn as nn
import numpy as np
from models.scinet.SCINet import SCINet
from models.scinet.utils.tools import EarlyStopping
from torch import optim


class Trainer_SCINet:
    def __init__(self, config):
        """
        Initialize class

        :param config: configuration , train_config
        :type config: dictionary
        """
        
        self.output_len = config['forecast_step']
        self.input_len = config['window_size']
        self.input_dim = config['input_size']
        self.lr = config['lr'] 
        self.num_epochs = config['num_epochs']
        
        self.model = SCINet(self.output_len, self.input_len, self.input_dim)
   
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
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = 100000
        
        for epoch in range(self.num_epochs):
            train_loss = []
            
            self.model.train()
           
            for i, (batch_x,batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                
                pred = self.model(batch_x)
                true = batch_y
                
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                    
                loss.backward()
                model_optim.step()
                
            train_loss = np.average(train_loss)
            valid_loss = self.valid(valid_loader, criterion)
            
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
                print('train Loss: {:.4f} RMSE: {:.4f}'.format(train_loss, np.sqrt(train_loss)))
                print('val Loss: {:.4f} RMSE: {:.4f}'.format(valid_loss, np.sqrt(valid_loss)))
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_val_loss))
        
        self.model.load_state_dict(best_model_wts)
        return self.model
    
    def valid(self, valid_loader, criterion):
        """
        Evaluate the model in training step 

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :param criterion: criterion for caculating validation loss
        :type criterion: Class

        :return: average validation loss for all validation dataset
        :rtype: Tensor
        """

        self.model.eval()

        total_loss = []
        for i, (batch_x, batch_y) in enumerate(valid_loader):
            pred = self.model(batch_x)
            true = batch_y
            
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        
        total_loss = np.average(total_loss)    
        return total_loss
    
    def test(self, test_loader):
        """
        Predict future values based on the best trained model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array
        """
        
        self.model.eval()
                
        preds, trues = [], []
   
        for i, (batch_x, batch_y) in enumerate(test_loader):
            pred = self.model(batch_x)
            true = batch_y
                      
            preds.extend(pred.detach().cpu().numpy())
            trues.extend(true.detach().cpu().numpy())
            
        preds = np.array(preds)
        preds = np.squeeze(preds, axis=2).ravel()
        preds = np.expand_dims(preds, axis=-1)
        return preds