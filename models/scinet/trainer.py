import os
import time
import torch.nn as nn
import numpy as np
from models.scinet.SCINet import SCINet
from utils.tools import EarlyStopping , save_model
from torch import optim

class Trainer_SCINet:
    def __init__(self, config):
        """
        Initialize class

        :param config: configuration , train_config
        :type config: dictionary
        """
        ## 학습시키고, 모델정의해서, bestmodel 리턴하라고?? 
        #train_config = config
        self.output_len = config['forecast_step']
        self.input_len = config['window_size']
        self.input_dim = config['input_size']
        self.patience = 5 ## config rewrite
        self.lr = config['lr'] 
        self.model = SCINet(self.output_len,self.input_len,self.input_dim)
        self.num_epochs = config['num_epochs']
        print(self.model)
   
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
        #모델 학습 시키고, 베스트모델만 리턴시켜라!!!!
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        epoch_start = 0 
        best_loss = 100000
        for epoch in range(epoch_start, self.num_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
           
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                pred = self.model(batch_x)
                true = batch_y
                ## 역변환 후 loss 계산하는 과정이라 생략함 
                # pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                #     train_data, batch_x, batch_y)
                
                #print(f'{pred.shape},{true.shape}')
                
                loss = criterion(pred, true)
                
                train_loss.append(loss.item())
                
                if (i+1) % 10==0:
                    print(f"\t iters: {i+1}, epoch: {epoch+1} | loss: {loss.item()}")
                    iter_count = 0
                    
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_loader,criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss ))
            
            if best_loss > valid_loss :
                print('update model')
                best_model = self.model
            early_stopping(valid_loss, self.model,None)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        #save_model(epoch, self.lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        #best_model_path = path+'/'+'checkpoint.pth'
        #self.model.load_state_dict(torch.load(best_model_path))
        return best_model



    def test(self, test_loader):
        """
        Predict future values based on the best trained model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array
        """
        
        #print('def in trainer class \n')
        self.model.eval()
                
        preds = []
        trues = []
   
        for i, (batch_x, batch_y) in enumerate(test_loader):
            
            pred = self.model(batch_x)
            true = batch_y
                      
            preds.extend(pred.detach().cpu().numpy())
            trues.extend(true.detach().cpu().numpy())
        
            
        preds = np.array(preds)
        preds = np.squeeze(preds,axis=2)
        # shape=(the number of prediction data, 1)
        return preds
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        #elif losstype == "mae":
        #    criterion = nn.L1Loss()
        return criterion
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return model_optim
    
    def valid(self,valid_loader,criterion) :
        self.model.eval()
        total_loss = []

    
        #pred_scales = []
        #true_scales = []
        
        for i, (batch_x, batch_y) in enumerate(valid_loader):
            
            pred = self.model(batch_x)
            true = batch_y
            
            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            # preds.append(pred.detach().cpu().numpy())
            # trues.append(true.detach().cpu().numpy())
            # pred_scales.append(pred_scale.detach().cpu().numpy())
            # true_scales.append(true_scale.detach().cpu().numpy())
            total_loss.append(loss)
        total_loss = np.average(total_loss)    
        
        #TODO : metric 
        # preds = np.array(preds)
        # trues = np.array(trues)
        
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        return total_loss
            
        
   