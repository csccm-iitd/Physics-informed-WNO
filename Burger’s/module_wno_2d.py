import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()

import matplotlib.pyplot as plt
from utilities3 import *

from timeit import default_timer
# from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)
from pytorch_wavelets import DTCWTForward, DTCWTInverse

torch.autograd.set_detect_anomaly(True)
    
torch.manual_seed(0)
np.random.seed(0)

# %%
""" Def: 2d Wavelet layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        # self.wavelet = 'coif2'
        dwt_ = DTCWTForward(J=self.level, biort='near_sym_b', qshift='qshift_b').to(dummy.device)
        mode_data, mode_coef = dwt_(dummy)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]

        self.scale = (1 / (in_channels * out_channels))
        self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights15r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights15c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # batchsize = x.shape[0]
        
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        # dwt = DWT(J=self.level, mode='symmetric', wave=self.wavelet).cuda()
        dwt = DTCWTForward(J=self.level, biort='near_sym_b', qshift='qshift_b').to(x.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply relevant Wavelet modes
        # out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights0)
        # Multiply the finer wavelet coefficients        
        x_coeff[-1][:,:,0,:,:,0] = self.mul2d(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights15r)
        x_coeff[-1][:,:,0,:,:,1] = self.mul2d(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights15c)
        x_coeff[-1][:,:,1,:,:,0] = self.mul2d(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights45r)
        x_coeff[-1][:,:,1,:,:,1] = self.mul2d(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights45c)
        x_coeff[-1][:,:,2,:,:,0] = self.mul2d(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights75r)
        x_coeff[-1][:,:,2,:,:,1] = self.mul2d(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights75c)
        x_coeff[-1][:,:,3,:,:,0] = self.mul2d(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights105r)
        x_coeff[-1][:,:,3,:,:,1] = self.mul2d(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights105c)
        x_coeff[-1][:,:,4,:,:,0] = self.mul2d(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights135r)
        x_coeff[-1][:,:,4,:,:,1] = self.mul2d(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights135c)
        x_coeff[-1][:,:,5,:,:,0] = self.mul2d(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights165r)
        x_coeff[-1][:,:,5,:,:,1] = self.mul2d(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights165c)
        
        # Return to physical space        
        # idwt = IDWT(mode='symmetric', wave=self.wavelet).cuda()
        idwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(x.device)
        x = idwt((out_ft, x_coeff))
        return x
    

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        # self.conv2 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        # self.conv3 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        # self.conv4 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv5 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 512)
        self.fc2 = nn.Linear(512, 1)
        # self.mu = param()

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding]) # padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)
        
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        # x = F.gelu(x)
        
        # x1 = self.conv4(x)
        # x2 = self.w4(x)
        # x = x1 + x2
        # x = F.gelu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # removing padding, when applicable
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
        