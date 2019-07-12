# -*- coding: utf-8 -*-
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from sympy.solvers import solve
from sympy import Symbol

def compute_padding_same(input_size, kernel, stride = 1, dilatation = 1):
    """ 
    Compute an adaquate padding to keep the same shape 
    Kernel should be odd for this to work
    """
    paddings = []
    input_size = input_size if type(input_size) == tuple else (input_size)
    kernel = kernel if type(kernel) == tuple else ([kernel]*len(input_size))
    for i_size, i_kernel in zip(input_size, kernel):
        output_size = i_size
        x = Symbol("padding")
        paddings.append(solve((i_size + x*2 - (i_kernel-1)*(dilatation) - 1)/stride + 1 - output_size, x)[0])
    return tuple(paddings)

def compute_output(input_size, kernel, padding, stride = 1, dilatation = 1):
    """ Compute the shape of the output """
    output = []
    input_size = input_size if type(input_size) == tuple else (input_size)
    kernel = kernel if type(kernel) == tuple else ([kernel]*len(input_size))
    padding = padding if type(padding) == tuple else ([padding]*len(input_size))

    for i_input_size, i_kernel, i_padding in zip(input_size, kernel, padding):
        output.append(int((i_input_size + i_padding*2 - (i_kernel-1)*(dilatation) - 1)/stride + 1 ))

    return tuple(output)

class PolecartLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.size = 32
        self.layers = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())

    def forward(self, x):
        return self.layers(x.flatten(start_dim=1))

class AtariConvolution(torch.nn.Module):

    def __init__(self):
        super(AtariConvolution, self).__init__()
        self.input_shape = (84,84)
        self.kernel_sizes = [5, 5]
        self.padding = [2, 2]
        self.stride = [3, 3]

        self.out_channels = [64, 64]

        self.in_channels = [4]+self.out_channels[:-1]

        self.conv = nn.ModuleList()

        shape = self.input_shape

        for i, (i_in_channel, i_out_channel, i_kernel, i_padding, i_stride) in enumerate(zip(self.in_channels, self.out_channels, self.kernel_sizes, self.padding, self.stride)):
            
            self.conv.append(nn.Conv2d(i_in_channel, i_out_channel, i_kernel, padding = i_padding, stride=i_stride))
            shape = compute_output(shape, i_kernel, i_padding, stride = i_stride) #Conv
            self.conv.append(nn.BatchNorm2d(i_out_channel))
            self.conv.append(nn.ReLU())
            #self.conv.append(nn.MaxPool2d(kernel_size = (2,2), stride = 2))
            #shape = compute_output(shape, (2,2), 0, 2) # MaxPooling 2

        self.size = shape[0]*shape[1]*self.out_channels[-1]

    def forward(self, x):
        for i_conv in self.conv:
            x = i_conv(x)
        return x


class DQN(torch.nn.Module):

    @staticmethod
    def compute_advantage_value(advantage, value):
        """ 
        Compute de advantage value
        From Dueling Network Architectures for Deep Reinforcement Learning
        at https://arxiv.org/pdf/1511.06581.pdf
        """
        relative_advantage = advantage - advantage.mean(1, keepdim = True)
        advantage_value =  value + relative_advantage
        return advantage_value

    @staticmethod
    def gather(advantage_value, actions = None):
        """ Gather the values corresponding to the played actions, everything by default"""
        return advantage_value if actions is None else torch.gather(advantage_value, 1, actions)

    def __init__(self, env_module, actions, layer_size = 1024):
        """
        env_module : An instance of the module that proceeds the state given by the environment
        actions : The number of action
        layer_size : The size of the intermediate layers of the DQN
        """
        super(DQN, self).__init__()
        self.input_layer = env_module
        self.value_layer = NoisyLinear(env_module.size,layer_size)
        self.advantage_layer = NoisyLinear(env_module.size,layer_size)

        self.value = NoisyLinear(layer_size, 1)
        self.advantage = NoisyLinear(layer_size, actions)

    def forward(self, x, actions = None):

        x = self.input_layer(x).flatten(start_dim=1)
        value = self.value(F.relu(self.value_layer(x)))
        advantage = self.advantage(F.relu(self.advantage_layer(x)))
        advantage_value =  self.compute_advantage_value(advantage, value)
        return DQN.gather(advantage_value, actions)

    def save_state_dict(self, PATH):
        torch.save(self.state_dict(), PATH)

class AtariDQN(DQN):
    """ DQN with adequate convolution layers for atari games"""
    def __init__(self, actions):
        super(AtariDQN, self).__init__(AtariConvolution(), actions)


class NoisyLinear(nn.Linear):
    """
    From Noisy Networks for Exploration at https://arxiv.org/abs/1706.10295
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

        def reset_parameters(self):
            std = math.sqrt(3 / self.in_features)
            nn.init.uniform(self.weight, -std, std)
            nn.init.uniform(self.bias, -std, std)

        def forward(self, input):
            torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
            bias = self.bias
            if bias is not None:
                 torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
                 bias = bias + self.sigma_bias * Variable(self.epsilon_bias)
            return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), bias)
