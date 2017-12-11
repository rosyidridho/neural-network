import numpy as np
import math
from operator import itemgetter

class LVQ:
    def __init__(self, vektor, weight, alpha, dec_alpha):
        self.vektor = vektor
        self.weight = weight
        self.alpha = alpha
        self.dec_alpha = dec_alpha

    def get_weight(self, vektor, weight):
        new = []
        for w in weight:
            new.append((math.sqrt(np.sum(map(lambda x, y: math.pow(float(x) - float(y), 2), vektor[0], w[0]))), w[1]))
        return new
    
    def check_weight(self, get_weight):
        return min(enumerate(get_weight), key=itemgetter(1))[1]

    def get_new_weight(self, check_weight, weight, alpha, vektor):
        for w in weight:
            if w[1] == check_weight[1]:
                w[0] = map(lambda x, y: (float(x) + (float(alpha) * (float(y)-float(x)))), w[0], vektor[0])
        return weight

    def train(self, epoh=None):
        if epoh == None:
            pass
        else:
            for i in range(epoh):
                for v in self.vektor:
                    g = self.get_weight(v, self.weight)
                    n = self.check_weight(g)
                    self.weight = self.get_new_weight(n, self.weight, self.alpha, v)
                    self.alpha = self.dec_alpha * self.alpha
        return self.weight

    def test(self, vektor, weight):
        a = []
        for v in vektor:
            data = self.get_weight(v, weight)
            b = self.check_weight(data)
            a.append(b)
        return a