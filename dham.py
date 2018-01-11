import numpy as np
import itertools
import torch
from torch.nn import Module
from torch.nn import Softmax
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from PIL import Image
import matplotlib.pyplot as plt


class Dham(Module):
    """
    This module is a transformation module for Differentiable Hard Attention.
    """

    def __init__(self, output_size):
        """
        Parameters:
            - output_size is a 2 dimensional tuple that determines the
         output image's spatial dimensions.
        """
        super(Dham, self).__init__()
        self.output_size = output_size

        # These indexes are going to be used multiple times.
        Y, X = self.output_size
        self.y_index = torch.linspace(-1, 1, Y).view(-1, 1).repeat(1, X)
        self.x_index = torch.linspace(-1, 1, X).view(1, -1).repeat(Y, 1)

        self.x_index = Variable(self.x_index.view(1, 1, Y, X))
        self.y_index = Variable(self.y_index.view(1, 1, Y, X))

    def transform_paramters(self, feature_map, scale_factor=1.0):
        """
        Parameters:
            - feature_map is where parameters are calculated from.
         It is expected to have a size of (B, C, Y, X).
            - scale_facor is used to scale the expected norm.
        """
        B, C, Y, X = feature_map.size()
        indexes_x = self.y_index.data.new(np.linspace(-1, 1, X))
        indexes_y = self.y_index.data.new(np.linspace(-1, 1, Y))        
   
        grid_x = Variable(indexes_x.view(1, 1, -1))
        grid_y = Variable(indexes_y.view(1, 1, -1))

        softmaxed_map = torch.nn.functional.softmax(feature_map.view(B, C, -1), dim=-1).view(B, C, Y, X)

        # Expected values
        mean_x = (softmaxed_map.sum(-2)*grid_x).sum(-1)
        mean_y = (softmaxed_map.sum(-1)*grid_y).sum(-1)

        # Norms
        difference_x = (grid_x - torch.unsqueeze(mean_x, -1)).view(B, C, 1, X).detach()
        difference_y = (grid_y - torch.unsqueeze(mean_y, -1)).view(B, C, Y, 1).detach()
        
        # Expected L2 norm
        # scale = ((difference_x.pow(2) + difference_y.pow(2)).sqrt()*softmaxed_map).sum(-1).sum(-1)

        #Expected L1 norm
        scale = ((torch.abs(difference_x) + torch.abs(difference_y))*softmaxed_map).sum(-1).sum(-1)        

        self.last_attention_params = (mean_x, mean_y, scale*scale_factor) 
        return mean_x, mean_y, scale*scale_factor

    def bilinear_transorm(self, input, *raw_ind):
        """
        Transforms input tensor with decimal indexes by 
         bilinear transformation.

       YX * * * * * Yx       a a b b b b
        * * * * * * *        a a b b b b
        * * + * * * *        c c d d d d
        * * * * * * *   =>   c c d d d d
        * * * * * * *        c c d d d d
        * * * * * * *        c c d d d d
       yX * * * * * yx
       <----- 6 ----->
        
        Roi = I(YX)*(4/6)*(4/6) + I(Yx)*(4/6)*(2/6) +
            I(yX)*(2/6)*(4/6) + I(yx)*(2/6)*(2/6)
        """
        y_raw_ind, x_raw_ind = raw_ind
        B, C, Y, X = input.size()

        y_raw_ind = (y_raw_ind + 1)*(Y/2)
        x_raw_ind = (x_raw_ind + 1)*(X/2)

        x_u = Variable(torch.ceil(x_raw_ind).clamp(0, X-1).data)
        y_u = Variable(torch.ceil(y_raw_ind).clamp(0, Y-1).data)
        x_l = Variable(torch.floor(x_raw_ind).clamp(0, X-1).data)
        y_l = Variable(torch.floor(y_raw_ind).clamp(0, Y-1).data)
    
        return (self.gather2d(input, y_u.long(), x_u.long())*torch.unsqueeze((x_raw_ind - x_l)*(y_raw_ind - y_l), dim=1)+
                self.gather2d(input, y_u.long(), x_l.long())*torch.unsqueeze((x_u - x_raw_ind)*(y_raw_ind - y_l), dim=1)+
                self.gather2d(input, y_l.long(), x_u.long())*torch.unsqueeze((x_raw_ind - x_l)*(y_u - y_raw_ind), dim=1)+
                self.gather2d(input, y_l.long(), x_l.long())*torch.unsqueeze((x_u - x_raw_ind)*(y_u - y_raw_ind), dim=1))

    @staticmethod
    def gather2d(input, *indexes):
        """
        This method gathers indexed elements from the input tensor.
        Indexes tensors hold indexes for the last two dimension of the input
        for each element in first two dimensions.
        Parameters:
            - Input is expected to be 4 dimensional
         Example shape: (B, C, X, Y).
            - indexes are two 4 dimensional tensors
         Example shape: (B, c, y, x). It is important to give y indexes
         first.
        B and C sizes of indexes and Input should be the same.
        Operation:
            I[i, j][Y[i, j], X[i, j]] for all i and j
        Example Output tensor size: (B, C, c, y, x)
        Note: B holds for batch dimension, C holds for input channel
        dimension, and c holds for attention channel dimension.
        """
        indy, indx = indexes
        B, C = input.size()[:2]
        Ic, Iy, Ix =  indx.size()[1:]
        return torch.stack([input[b, c][indy[b], indx[b]]
                            for b, c in itertools.product(range(B), range(C))],
                             dim=0).view(B, C, Ic, Iy, Ix)

    def roi_indexes(self, *parameters):
        """
        Parameters:
            - paramters are output of the <transform_paramters> function.
         Expected size for each one is: (B, C)
        """
        mean_x, mean_y, scale = parameters
        
        y_ind = self.y_index*scale.view(*scale.size(), 1, 1) + mean_y.view(*mean_y.size(), 1, 1)
        x_ind = self.x_index*scale.view(*scale.size(), 1, 1) + mean_x.view(*mean_x.size(), 1, 1)
        #TODO: Implement Matmul version.

        return y_ind, x_ind
        
    def forward(self, images, feature_map):
        params = self.transform_paramters(feature_map)
        indexes = self.roi_indexes(*params)
        return self.bilinear_transorm(images, *indexes), params

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
        
        # <-------------- Addition -------------->
        self.x_index.data = fn(self.x_index.data)
        self.y_index.data = fn(self.y_index.data)
        # <--------------   End   --------------->
        
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self
