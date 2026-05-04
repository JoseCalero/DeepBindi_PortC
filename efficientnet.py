import numpy as np
np.random.seed(0)
import math

import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

from torch.nn import Conv2d, BatchNorm2d, SiLU, AdaptiveAvgPool2d, Sigmoid, Identity

class DropLayers(nn.Module):
    
    def __init__(self, drop_prob = 0.2):
        super(DropLayers, self).__init__()
        
        self.p =  1 - drop_prob
        
    def forward(self, x):
        
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        
        return torch.div(x, self.p) * binary_tensor

''' A simple Convolution, Batch Normalization, and Activation Class'''
class ConvBnAct(nn.Module):
    
    def __init__(self, n_in, n_out, kernel_size = 3, stride = 1, 
                 padding = 0, groups = 1, bn = True, act = True,
                 bias = False
                ):
        
        super(ConvBnAct, self).__init__()
        
        self.conv = Conv2d(n_in, n_out, kernel_size = kernel_size,
                              stride = stride, padding = padding,
                              groups = groups, bias = bias
                             )
        self.batch_norm = BatchNorm2d(n_out) if bn else Identity()
        self.activation = SiLU() if act else Identity()
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
    
''' Squeeze and Excitation Block '''
class SqueezeExcitation(nn.Module):
    
    def __init__(self, n_in, reduced_dim):
        super(SqueezeExcitation, self).__init__()
             
        self.se = nn.Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(n_in, reduced_dim, kernel_size=1),
            SiLU(),
            Conv2d(reduced_dim, n_in, kernel_size=1),
            Sigmoid()
        )
       
    def forward(self, x):
        
        y = self.se(x)
        
        return x * y

class MBConvBlock(nn.Module):

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, n_in, n_out, kernel_size = 3, stride = 1, expansion_factor = 6,
                 se_block = True, reduction= 4, drop_connect_rate = 0.2):
        
        super(MBConvBlock,self).__init__()

        self.has_se = se_block
        self.skip_connection = (stride == 1 and n_in == n_out)
                
        # Expansion
        intermediate_size = int(n_in * expansion_factor) # number of out channels

        if expansion_factor != 1:
            self.expand = ConvBnAct(n_in= n_in, n_out=intermediate_size, kernel_size=1)
        else:
            self.expand = Identity()

        # Depthwise convolution
        padding = (kernel_size - 1)//2
        self.depthwise_conv = ConvBnAct(n_in=intermediate_size, n_out=intermediate_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding,groups=intermediate_size)
       
        # Squeeze and excitation
        if(self.has_se):
            reduced_dim = int(n_in//reduction)
            self.se_reduce = SqueezeExcitation(n_in=intermediate_size, reduced_dim=reduced_dim)
                   
        # Projection layer 
        self.project_conv = ConvBnAct(n_in=intermediate_size, n_out=n_out, kernel_size=1, act=False)      

        self.drop_layers = DropLayers(drop_prob=drop_connect_rate)


    def forward (self, x):

        residuals = x 
        
        # Expansion 
        x = self.expand(x)

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Squeeze and excitation
        if self.has_se:
            x = self.se_reduce(x)
        
        # Pointwise convolution
        x = self.project_conv(x)

        # Skip connection and drop connect
        if self.skip_connection:
            x = self.drop_layers(x)
            x = x + residuals
        
        return x

class EfficientNetB0(nn.Module):
    
    def __init__(self, width_mult = 1, depth_mult = 1, dropout = 0.2, num_classes = 1, scaling_params = [0.5,0.514]):
        
        super(EfficientNetB0, self).__init__()
      
        # En caso de seleccionar la efficientNet reducida, inicializamos los parametros de escalado
        dj = scaling_params[0]
        wi = scaling_params[1]
     
        self.dropout = dropout

        last_channel = int(math.ceil(1280 * width_mult * wi))
        self.features = self.feature_extractor(width_mult,depth_mult,last_channel,dj,wi)

        self.adaptative_avg_2d = AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.Linear(last_channel, num_classes))

    def forward(self, x):
        
        x = self.features(x)
        x = self.adaptative_avg_2d(x)
        x = self.classifier(x.view(x.shape[0], -1))
        x = torch.sigmoid(x)
        
        return x
         
    def feature_extractor(self, width_mult, depth_mult, last_channel,sf_dj,sf_wi):
        
        dj = sf_dj
        wi = sf_wi
        
        # Sabemos que la entrada del modelo de efficientNet tiene una entrada de 32 canales
        initial_channels = int(np.ceil(32*wi))
        channels = 4*math.ceil(int(initial_channels*width_mult) / 4)
        
        in_channels = 1
        layers = [ConvBnAct(in_channels, channels, kernel_size = 3, stride = 2, padding = 1)]
        in_channels = channels 
        
        kernels = [3, 3, 5, 3, 5, 5, 3]
        expansions = [1, 6, 6, 6, 6, 6, 6]
        num_channels = [16, 24, 40, 80, 112, 192, 320]
        num_layers = [1, 2, 2, 3, 3, 4, 1]
        strides =[1, 2, 2, 2, 1, 2, 1]

        expansions = [int(np.ceil(i * dj)) for i in num_layers]
        num_channels = [int(np.ceil(i * wi)) for i in num_channels]
        
        # Scale channels and num_layers according to width and depth multipliers.
        scaled_num_channels = [4*math.ceil(int(c*width_mult) / 4) for c in num_channels]
        scaled_num_layers = [int(d * depth_mult) for d in num_layers]

        # El stride diferente a 1 solo se utiliza en la primera repeticion de la capa
        for i in range(len(scaled_num_channels)):
             
            layers += [MBConvBlock(in_channels if repeat==0 else scaled_num_channels[i], 
                               scaled_num_channels[i],
                               kernel_size = kernels[i],
                               stride = strides[i] if repeat==0 else 1, 
                               expansion_factor = expansions[i]
                              )
                       for repeat in range(scaled_num_layers[i])
                      ]
            in_channels = scaled_num_channels[i]
        
        layers.append(ConvBnAct(in_channels, last_channel, kernel_size = 1, stride = 1, padding = 0))
    
        return nn.Sequential(*layers)
    




