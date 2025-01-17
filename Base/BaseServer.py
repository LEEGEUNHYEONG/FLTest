import copy
import numpy as np
class BaseServer:
    __instance = None

    server_weight_avg = None
    count = 0

    @classmethod
    def __get_instance(cls):
        return cls.__instance

    @classmethod
    def instance(cls):
        cls.__instance = cls()
        cls.instance = cls.__get_instance()
        return cls.__instance

    def __init__(self):
        print("Base server init")

    temp_weight = []
    def update_weight(self, local_weight_list):
        local_avgs = copy.deepcopy(local_weight_list)
        self.temp_weight = local_avgs
        self.count += 1

        if self.server_weight_avg is None:
            self.server_weight_avg = local_avgs[0]
        else:
            ''' average '''
            self.cal_average(local_avgs)
            #self.temp(local_avgs)
            #self.cal_harmonic_mean(local_avgs)

        return self.server_weight_avg

    def get_weight(self):
        #print("get_weight : ", self.server_weight_avg, " /// ", self.count)
        if self.server_weight_avg is not None or self.count > 0:
            return self.server_weight_avg
        return None

    def init_weight(self):
        self.server_weight_avg = None
        self.count = 0

    def cal_average(self, local_avgs):
        for i in range(len(local_avgs)):
            for j in range(len(local_avgs[i])):
                self.server_weight_avg[j] += local_avgs[i][j]
        self.server_weight_avg = np.divide(self.server_weight_avg, len(local_avgs) + 1)



    def cal_harmonic_mean(self, local_weights):
        print("cal_harmonic_mean")

    def temp(self, local_weights):
        print("temp")
        sorted(local_weights)
        print("sorted")






