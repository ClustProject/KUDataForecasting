import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, **kwargs):
        self.config = kwargs
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        return torch.device(self.config['device'])

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    