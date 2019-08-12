# %%
import numpy as np


class Server:
    value = 0.0
    count = 0
    weight_list = []

    __instance = None

    @classmethod
    def getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.getInstance()
        return cls.__instance

    def __init__(self):
        print("Server init")

    def update_value(self, list):
        # server weight init :  (784, 512) (512,) (512, 10) (10,)

        if len(self.weight_list) == 0:
            self.weight_list = list[:]
            print("server weight init : ", self.weight_list[0].shape, self.weight_list[1].shape, self.weight_list[2].shape, self.weight_list[3].shape)
            self.count +=1

        else:
            print("server weight update")
            self.count += 1

            self.weight_list[0] = self.weight_list[0] + list[0]
            self.weight_list[1] = self.weight_list[1] + list[1]
            self.weight_list[2] = self.weight_list[2] + list[2]
            self.weight_list[3] = self.weight_list[3] + list[3]

            self.weight_list[0] = np.divide(self.weight_list[0], self.count)
            self.weight_list[1] = np.divide(self.weight_list[1], self.count)
            self.weight_list[2] = np.divide(self.weight_list[2], self.count)
            self.weight_list[3] = np.divide(self.weight_list[3], self.count)

            print("server weight updated success")

        return self.weight_list

    def clear_weight(self):
        self.weight_list = []
        self.count = 0


    def get_weight(self):
        return self.weight_list
