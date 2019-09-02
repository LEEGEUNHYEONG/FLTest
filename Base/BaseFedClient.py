import time

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
        hist, weight = self.model.fit(self.train_data, self.train_label, epoch=self.epoch, batch_size=self.batch_size)
        print("time : {}".format(time.time()-start_time))

    def run_evaluate(self):
        result = self.model.local_evaluate(self.test_data, self.test_label)
        print(result)

