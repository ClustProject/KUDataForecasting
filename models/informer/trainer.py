from .exp.exp_informer import Exp_Informer

class Trainer_Informer:
    def __init__(self, config):
        """
        Initialize class

        :param config: configuration
        :type config: dictionary
        """
        self.exp = Exp_Informer(**config)
        self.model = self.exp.model

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
    
        best_model = self.exp.train(train_loader, valid_loader)
        
        return best_model
    
    def test(self, test_loader):
        """
        Predict future values based on the best trained model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array
        """
        
        pred_data = self.exp.test(test_loader)

        return pred_data