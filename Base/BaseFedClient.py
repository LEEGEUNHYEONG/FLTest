import time

import pandas as pd
from tensorflow.python.training.input import batch

from Base import BaseModel


class BaseFedClient:
    model = None

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    epoch = 20
    batch_size = 10

    def __init__(self):
        self.model = BaseModel.BaseModel()
        pass

    '''
        1. data split ?
        2. epoch, batch size 설정
        3. fit
        4. evaluate
    '''

    def set(self, train_data, train_label, test_data, test_label, epoch=epoch, batch_size=batch_size):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.epoch = epoch
        self.batch_size = batch_size


    def run_federate(self):
        start_time = time.time()
        print("start federated !!!")
        history = self.model.fit(self.train_data, self.train_label, epoch=self.epoch, batch_size=self.batch_size)
        print("time : {}".format(time.time()-start_time))
        return history


    def run_evaluate(self):
        loss, mae, mse = self.model.local_evaluate(self.test_data, self.test_label)
        print("run_evaluate : loss : {},  mae : {},  mse : {}".format(loss, mae, mse))


    def run_predict(self):
        result = self.model.model.predict(self.test_data).flatten()
        return result


    def test(self):
        result = self.model.model.predict(self.test_data).flatten()
        real_price_list = self.test_label.values.tolist()

        for i in range(len(result)):
            real_price = real_price_list[i]
            predicted_price = result[i]
            print("{} >>> {}".format(real_price, predicted_price))

    def test2(self):
        test_prediction = self.model.model.predict(self.test_data).flatten()
        return test_prediction



