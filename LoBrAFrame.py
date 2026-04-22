import torch
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba

'''
If you found the study (https://ieeexplore.ieee.org/document/11279989) useful for you, 
please consider citing it.

https://github.com/wzhlearning/LoBrAFrame

@ARTICLE{Wang2025LoBrAFrame,
author  = {Wang, Zenghui and Du, Songlin},
journal = {IEEE Transactions on Industrial Informatics}, 
title   = {A Unified fNIRS Classification Framework Informed by Local Brain Activation Patterns}, 
year    = {2026},
volume  = {22},
number  = {3},
pages   = {1871-1881},
doi     = {10.1109/TII.2025.3632147}
}

'''


class Shared_Brain_Activation(torch.nn.Module):
    '''
    Shared brain activation (SBA) module
    '''
    def __init__(self, signal_channel, signal_length, d_model):
        super(Shared_Brain_Activation, self).__init__()

        self.conv_1xs = torch.nn.Conv2d(1, d_model, (1, signal_length)) 
        self.signal_channel = signal_channel
        self.in_layer = torch.nn.InstanceNorm2d(d_model) 
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(d_model, d_model, (1, 1), stride=(1, 1)), 
            torch.nn.Dropout2d(0.2),
            Rearrange('b c h w -> b (h w) c'))

    def forward(self, x):
        HbO = self.conv_1xs(x[:, :,                      :self.signal_channel ,  :])
        HbR = self.conv_1xs(x[:, :, self.signal_channel  :self.signal_channel*2, :])
        HbT = self.conv_1xs(x[:, :, self.signal_channel*2:self.signal_channel*3, :])
        HbD = self.conv_1xs(x[:, :, self.signal_channel*3:self.signal_channel*4, :])
        x_shared = torch.cat((HbO, HbR, HbT, HbD), dim=-2)

        HbO = self.in_layer(HbO)
        HbR = self.in_layer(HbR)
        HbT = self.in_layer(HbT)
        HbD = self.in_layer(HbD)
        hypersignals = torch.cat((HbO, HbR, HbT, HbD), dim=-2)
        hypersignals = self.block(hypersignals)
        return x_shared, hypersignals
    


class DWSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DWSConv, self).__init__()
        self.depth_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0, groups=in_channels)
        self.point_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        x = self.depth_conv(input)
        x = self.point_conv(x)
        return x
    

class Hypersignal_Learning(torch.nn.Module):
    '''
    Mamba-based hypersignal learning (HL) module
    '''
    def __init__(self, signal_channel, HL_layers, d_model, dropout):
        super(Hypersignal_Learning, self).__init__()

        self.ssm_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(HL_layers):
            self.ssm_layers.append(Mamba(d_model=d_model, # Model dimension d_model
                                         d_state=16, #  SSM state expansion factor
                                         d_conv=1, # Local convolution width 
                                         expand=2)) # Block expansion factor
            self.norms.append(torch.nn.LayerNorm(signal_channel*4))
            self.dropouts.append(torch.nn.Dropout1d(dropout))

    def forward(self, x):
        for layer, norm, dropout in zip(self.ssm_layers, self.norms, self.dropouts):
            z = x
            z = layer(z)
            z = dropout(z)
            x = z + x
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        return x


class Shared_Branch(torch.nn.Module):
    '''
    fNIRSNet (see \cite{wang2023fNIRSNet}) as a shared branch.

    https://ieeexplore.ieee.org/document/10311392

    https://github.com/wzhlearning/fNIRSNet

    @ARTICLE{wang2023fNIRSNet,
        author  = {Wang, Zenghui and Fang, Jihong and Zhang, Jun},
        journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
        title   = {Rethinking Delayed Hemodynamic Responses for fNIRS Classification}, 
        year    = {2023},
        volume  = {31},
        number  = {},
        pages   = {4528-4538},
        doi     = {10.1109/TNSRE.2023.3330911}
    }

    '''
    def __init__(self, signal_channel, d_model):
        super(Shared_Branch, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(d_model)
        self.conv2 = DWSConv(in_channels=d_model, out_channels=d_model, kernel_size=(signal_channel*4, 1)) 
        self.bn2 = torch.nn.BatchNorm2d(d_model) 
        self.act = torch.nn.Sigmoid()
        
    def forward(self, x_shared):
        x_branch = self.act(self.bn1(x_shared))
        x_branch = self.act(self.bn2(self.conv2(x_branch)))
        x_branch = x_branch.view(x_branch.size()[0], -1)
        return x_branch



class LoBrAMamba(torch.nn.Module):
    '''
    LoBrAMamba is an instantiated model of LoBrAFrame.

    Args:
        n_classes: Number of classes for classification.

        signal_channel: Number of input channels in the signal. For example, 
            `signal_channel` is set to 20 for a 20-channel fNIRS signal.

        signal_length: Length of the input signal sequence (time steps). 
            For example, the length of a 3-second sliding window is 40 for UFFT.

        d_model: Dimensionality of the model's hidden representations. The default is 40.

        HL_layers: Number of Mamba layers within the HL module. The default is 40.

        HL_dropout: Dropout rate of the HL module. The default is 0.8.

    '''
    def __init__(self, n_classes, signal_channel, signal_length, d_model=40, HL_layers=2, HL_dropout=0.8):
        super(LoBrAMamba, self).__init__()

        # Shared brain activation (SBA) module
        self.SBA_Module = Shared_Brain_Activation(signal_channel, signal_length, d_model)

        # shared branch
        self.shared_branch = Shared_Branch(signal_channel, d_model)

        # Mamba-based hypersignal learning (HL) module
        self.HL_Module = Hypersignal_Learning(signal_channel, HL_layers, d_model, HL_dropout)

        # classification
        self.fc = torch.nn.Sequential(torch.nn.Linear(signal_channel*4+d_model, n_classes)) 
        
        
    def forward(self, x):
 
        x_shared, x_hypersignals = self.SBA_Module(x)

        x_branch = self.shared_branch(x_shared)

        x_hypersignals = self.HL_Module(x_hypersignals)

        x = torch.cat((x_hypersignals, x_branch), dim=-1)

        x = self.fc(x)

        return x



class LoBrAFrame(torch.nn.Module):
    '''
    LoBrAFrame is a unified fNIRS classification framework. Researchers need only specify a 
    simple or off-the-shelf model to integrate into LoBrAFrame, thereby obtaining a 
    higher-performance instantiated model.

    Args:
        n_classes: Number of classes for classification.

        signal_channel: Number of input channels in the signal. For example, 
            `signal_channel` is set to 20 for a 20-channel fNIRS signal.

        signal_length: Length of the input signal sequence (time steps). 
            For example, the length of a 3-second sliding window is 40 for UFFT.

        d_model: Dimensionality of the model's hidden representations. The default is 40.

    '''
    def __init__(self, n_classes, signal_channel, signal_length, d_model=40):
        super(LoBrAMamba, self).__init__()

        # Shared brain activation (SBA) module
        self.SBA_Module = Shared_Brain_Activation(signal_channel, signal_length, d_model)

        # shared branch
        self.shared_branch = Shared_Branch(signal_channel, d_model)
        
        # a user-specified, simple or off-the-shelf model
        self.sample_model = Sample_Model()

        # classification
        self.fc = torch.nn.Sequential(torch.nn.Linear(signal_channel*4+d_model, n_classes)) 
        
        
    def forward(self, x):
        x_shared, x_hypersignals = self.SBA_Module(x)

        x_branch = self.shared_branch(x_shared)

        # a user-specified simple model learns knowledge representations from hypersignals.
        x_hypersignals = self.sample_model(x_hypersignals)

        x = torch.cat((x_hypersignals, x_branch), dim=-1)

        x = self.fc(x)

        return x


class Sample_Model(torch.nn.Module):
    '''
    A user-specified, simple or off-the-shelf model
    '''
    def __init__(self):
        super().__init__()


    def forward(self, x):

        return x
