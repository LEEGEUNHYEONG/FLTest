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
        if len(self.weight_list) == 0 :
            self.weight_list = list[:]
        else:
            temp = np.array([self.weight_list[0], list[0]])
            self.weight_list[0] = temp.mean(0)

            temp = np.array([self.weight_list[1], list[1]])
            self.weight_list[1] = temp.mean(0)

            temp = np.array([self.weight_list[2], list[2]])
            self.weight_list[2] = temp.mean(0)

            temp = np.array([self.weight_list[3], list[3]])
            self.weight_list[3] = temp.mean(0)

        return self.weight_list