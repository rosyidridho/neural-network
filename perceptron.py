import numpy as np

class Perceptron:
    def __init__(self,data,bobot,treshold, alpha):
        self.data = data
        self.bobot = bobot
        self.treshold = treshold
        self.alpha = alpha
        self.results = []

    def activation(self, bobot, data_input):
        return np.sum(map(lambda x, y: float(x) * float(y), data_input, bobot[0])) + bobot[1]
            
    def func_activation(self, treshold, activation):
        return 1 if activation>treshold else 0 if activation==treshold else -1

    def check_func_activation(self, target, func_activation):
        return target == func_activation
    
    def weight_new(self, alpha, target, data_input, check):
        if not check:
            a = np.asarray(data_input) * alpha * target
            b = alpha * target
            return (np.ndarray.tolist(a), b)
        else:
            a = (np.asarray(data_input) * alpha * target) * 0
            b = 0
            return (np.ndarray.tolist(a), b)

    def final_bobot(self, weight_old, weight_new, target):
        new = np.asarray(weight_old[0]) + np.asarray(weight_new[0])
        bias = weight_old[1] + weight_new[1]
        return (np.ndarray.tolist(new), bias)

    def execute(self):
        epoh = 1
        while True:
            c = []
            for x in self.data:
                ac = self.activation(self.bobot, data_input=x[0])
                fac = self.func_activation(self.treshold, activation=ac)
                cfac = self.check_func_activation(x[1], fac)
                cw = self.weight_new(self.alpha, x[1], x[0], cfac)
                self.bobot = self.final_bobot(self.bobot, cw, x[1])            
                c.append(cfac)
                data = [epoh, [ac,fac,cfac,cw,self.bobot]]
                
                self.results.append(data)
            epoh = epoh+1
                
            d = filter(lambda x: x==False, c)
            if len(d) == 0:
                break
            
        return self.results


