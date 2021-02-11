import math

import torch
from torch import nn
from torch import exp, log

from src.optim import TempVar

class GumbelLayer(nn.Module):
    Forward_Onehot = False
    ''' 
    Base class for all gumbel-stochastic layers
    '''
    def __init__(self, inner, norm):
        '''
        inner:
            pytorch module subclass instance used as deterministic inner class
        norm:
           batch norm object
        '''
        super(GumbelLayer, self).__init__()
        self.inner = inner
        self.need_grads = False
        if norm:
            self.norm = norm
        else:
            self.norm = nn.Identity()

        self.temp = TempVar()


    def forward(self, x):
        l = self.inner(x)
        l = self.norm(l)
        # Change p to double so that gumbel_softmax func works
        delta = 1e-5
        p = torch.clamp(torch.sigmoid(l).double(), min=delta, max=1-delta)
        o = self.sample(p)
        # Change output back to float for the next layer's input
        return o.float()


    def sample(self, p):
        if self.training or self.need_grads:
            # sample relaxed bernoulli dist
            return self._gumbel_softmax(p) 
        else:
            return torch.bernoulli(p).to(self.inner.weight.device)


    def _gumbel_softmax(self, p):
        g1 = self._sample_gumbel_dist(p.shape)
        g2 = self._sample_gumbel_dist(p.shape)
        p_term = exp((log(p) + g1) / self.temp.val)
        q_term = exp((log(1-p) + g2) / self.temp.val)
        y_soft = p_term / (p_term + q_term)
        if GumbelLayer.Forward_Onehot:
            y_hard = torch.argmax(torch.stack((q_term, p_term)), dim=0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
 
        
    def _sample_gumbel_dist(self, input_size):
        if self.inner.weight.device == torch.device('cpu'):
            u = torch.FloatTensor(input_size).uniform_()
        else:
            u = torch.cuda.FloatTensor(input_size,
                    device=self.inner.weight.device).uniform_()
        return -log(-log(u))


class Linear(GumbelLayer):
    def __init__(self, input_dim, output_dim, norm=False):
        inner = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(inner.weight)
        norm_obj = nn.BatchNorm1d(output_dim) if norm else None
        super(Linear, self).__init__(inner, norm_obj)


class Conv2d(GumbelLayer):
    def __init__(self, inc, outc, norm=False, **kwargs):
        inner = nn.Conv2d(inc, outc, **kwargs)
        norm_obj = nn.BatchNorm2d(outc) if norm else None
        super(Conv2d, self).__init__(inner, norm_obj)


