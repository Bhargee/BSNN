import math

class _TempScheduler():
    def __init__(self, temps):
        self.temps = temps
        self.time_step = 0
       
    def step(self):
        self.time_step += 1

class TempVar():
    def __init__(self, val=-math.inf):
        self.val = val

    def __repr__(self):
        return self.val.__repr__()


class ExponentialScheduler(_TempScheduler):
    def __init__(self, temps, start, minn, epochs):
        super(ExponentialScheduler, self).__init__(temps)
        self.start = start
        self.r = (math.log(start)-math.log(minn))/epochs
        self.time_step = -1
        self.step()

    def step(self):
        super(ExponentialScheduler, self).step()
        tau = self.start*math.exp(-self.r*self.time_step)
        for temp in self.temps:
            temp.val = tau


class LinearScheduler(_TempScheduler):
    def __init__(self, temps, start, minn, epochs):
        super(LinearScheduler, self).__init__(temps)
        self.start = start
        self.m = (minn-start)/epochs
        self.time_step = -1
        self.step()

    def step(self):
        super(LinearScheduler, self).step()
        tau = (self.m*self.time_step) + self.start
        for temp in self.temps:
            temp.val = tau


class ConstScheduler(_TempScheduler):
    def __init__(self, temps, const):
        super(ConstScheduler, self).__init__(temps)
        self.const = const
        self.set = False
        self.step()

    def step(self):
        super(ConstScheduler, self).step()
        if self.set:
            return
        else:
            for temp in self.temps:
                temp.val = self.const
            self.set = True
