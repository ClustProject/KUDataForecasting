from ..exp.exp_basic import Exp_Basic
from ..layers.model import Informer

from ..utils.tools import EarlyStopping, adjust_learning_rate

import numpy as np

import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, **kwargs):
        super(Exp_Informer, self).__init__(**kwargs)
    
    def _build_model(self):
        model = Informer(
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
            device=self.device
        ).float()
            
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                batch_x, batch_y)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, train_loader, vali_loader):

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.config['num_epochs']):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.config['num_epochs'] - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            
            print("-----------------------------------------------------------------")
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            adjust_learning_rate(model_optim, epoch+1, self.config)
            
        return self.model

    def test(self, test_loader):

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds)
        preds = preds.reshape(-1, preds.shape[-1])
        
        return preds

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        # decoder input
        # zero padding
        dec_inp = torch.zeros([batch_y.shape[0], self.config['forecast_step'], batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.config['label_len'],:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        outputs = self.model(batch_x, dec_inp)
        batch_y = batch_y[:,-self.config['forecast_step']:,:].to(self.device)

        return outputs, batch_y
