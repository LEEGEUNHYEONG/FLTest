import copy

import numpy as np


class BaseServer:
    __instance = None

    server_weight_avg = None
    count = 0

    @classmethod
    def get_instance(cls):
        return cls.__instance

    @classmethod
    def instance(cls):
        cls.__instance = cls()
        cls.instance = cls.get_instance()
        return cls.__instance

    def __init__(self):
        print("Base server init")

    def update_weight(self, local_weight_list):
        local_avgs = copy.deepcopy(local_weight_list)
        self.count += 1

        if self.server_weight_avg is None:
            self.server_weight_avg = local_avgs[0]
        else:
            for i in range(len(local_avgs)):
                for j in range(len(local_avgs[i])):
                    self.server_weight_avg[j] += local_avgs[i][j]
            self.server_weight_avg = np.divide(self.server_weight_avg, len(local_avgs) + 1)

        return self.server_weight_avg

    def get_weight(self):
        if self.server_weight_avg or self.count > 0:
            return self.server_weight_avg
        return []