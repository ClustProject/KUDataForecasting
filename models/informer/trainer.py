from .layers.model import Informer

from .utils.tools import adjust_learning_rate

import torch
import torch.nn as nn
from torch import optim

import os
import time
import copy
import numpy as np

import warnings
warnings.filterwarnings('ignore')

        
class Trainer_Informer:
    def __init__(self, config):
        """
        Initialize class

        :param config: configuration
        :type config: dictionary
        """
        
        self.config = config
        self.model = Informer(
            self.config['input_size'],
            self.config['input_size'], 
            self.config['input_size'], 
            self.config['window_size'], 
            self.config['label_len'],
            self.config['forecast_step'], 
            self.config['factor'],
            self.config['d_model'], 
            self.config['n_heads'], 
            self.config['e_layers'],
            self.config['d_layers'], 
            self.config['d_ff'],
            self.config['dropout'], 
            self.config['attn'],
            self.config['embed'],
            device = self.config['device']
        ).float().to(self.config['device'])

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
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        criterion = nn.MSELoss()
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = 100000000

        for epoch in range(self.config['num_epochs']):
            train_loss = []
            
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                
                pred, true = self._process_one_batch(batch_x, batch_y)
                
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()
            
            train_loss = np.average(train_loss)
            valid_loss = self.valid(valid_loader, criterion)
            
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, self.config['num_epochs']))
                print('train Loss: {:.4f} RMSE: {:.4f}'.format(train_loss, np.sqrt(train_loss)))
                print('val Loss: {:.4f} RMSE: {:.4f}'.format(valid_loss, np.sqrt(valid_loss)))
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            adjust_learning_rate(model_optim, epoch + 1, self.config)
            
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_val_loss))
        
        self.model.load_state_dict(best_model_wts)
        return self.model
    
    def _process_one_batch(self, batch_x, batch_y):
        """
        Train the model for one batch

        :param batch_x: batch data for input
        :type batch_x: Tensor

        :param batch_y: batch data for target label
        :type batch_y: Tensor

        :return: outputs from the model and target label
        :rtype: Tensor
        """

        batch_x = batch_x.float().to(self.config['device'])
        batch_y = batch_y.float()

        # decoder input
        # zero padding
        dec_inp = torch.zeros([batch_y.shape[0], self.config['forecast_step'], batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.config['label_len'],:], dec_inp], dim=1).float().to(self.config['device'])
        
        # encoder - decoder
        outputs = self.model(batch_x, dec_inp)
        batch_y = batch_y[:,-self.config['forecast_step']:,:].to(self.config['device'])
        return outputs, batch_y
    
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
            pred, true = self._process_one_batch(batch_x, batch_y)
            
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
        
        preds = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            pred, true = self._process_one_batch(batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds)
        preds = preds.reshape(-1, preds.shape[-1])
        return preds