class Trainer_SCINet:
    def __init__(self, config):
        """
        Initialize class

        :param config: configuration
        :type config: dictionary
        """

        pass

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

        pass

    def test(self, test_loader):
        """
        Predict future values based on the best trained model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array
        """

        pass