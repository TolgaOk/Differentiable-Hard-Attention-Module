import numpy as np
import itertools
import torch
from torch.nn import Module
from torch.nn import Softmax
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax
from torch.nn import Softmax
import time



def mesh_2d(x, y):
    """ Converts a 1D torch tensor into 2D grid.
    """
    if not isinstance(x, Variable):
        x, y= Variable(x), Variable(y)
    Y, X = y.size()[0], x.size()[0]
    XX = x.view(1, -1).repeat(Y, 1)
    YY = x.view(-1, 1).repeat(1, X)
    return XX, YY

def expected_value(density):
    """ Returns the expected value for last 2 dimensions
     for given 4D density function.
    """
    B, C, Y, X = density.size()
    
    x_indexes = Variable(density.data.new(np.linspace(-1, 1, X)))
    y_indexes = Variable(density.data.new(np.linspace(-1, 1, Y)))

    print(x_indexes)
    return (density.sum(-2)*x_indexes).sum(-1), (density.sum(-1)*y_indexes).sum(-1)


def standart_deviation(density):
    """ Returns the standart deviation for last 2 dimensions
     for given 4D density function.
    """
    B, C, Y, X = density.size()    
    XX, YY = mesh_2d(density.data.new(np.linspace(-1, 1, X)),
                        density.data.new(np.linspace(-1, 1, Y)))

    expected_x, expected_y = expected_value(density)
    x_indexes = Variable(density.data.new(np.linspace(-1, 1, X)))
    y_indexes = Variable(density.data.new(np.linspace(-1, 1, Y)))
    norm_4d = torch.sqrt( (x_indexes.view(1, 1, 1, X)-expected_x.view(B, C, 1, 1)).pow(2)+
                          (y_indexes.view(1, 1, Y, 1)-expected_y.view(B, C, 1, 1)).pow(2))

    return (norm_4d*density).sum(-1).sum(-1)


if __name__ == "__main__":
    B, C, Y, X = 1, 1, 6, 6
    vect_x = Variable(torch.linspace(-1, 1, Y).cuda())
    vect_y = Variable(torch.linspace(-1, 1, X).cuda())
    soft_max = Softmax(-1)

    XX, YY = mesh_2d(vect_x, vect_y)
    density_original = Variable(torch.normal(torch.zeros(1, 1, 6, 6), torch.ones(1, 1, 6, 6)).cuda(), requires_grad=True)
    density = softmax(density_original.view(B, C, -1), dim=-1).view(B, C ,Y, X)
    Ex, Ey = expected_value(density)
    S = standart_deviation(density)
    
    S.backward(torch.ones(1, 1).cuda())
    
    print(density_original.grad)
    
