import torch
import torchvision.models as models
import torch.nn as nn
torch.manual_seed(0)

import keras
from keras import layers
keras.utils.set_random_seed(0)

# Constant dropout for all model configurations
DROP_OUT = 0.2

"""
----------------------------------------------------------------------------------------------------
Model definition used for the publication DeepBindi: An End-to-End Fear Detection System Optimized
for Extreme-Edge Deployment (PYTORCH)
----------------------------------------------------------------------------------------------------
Configurations analyzed. Feature map size and optimal configuration selection
"""

# PYA: 2D-CNN square input. (2 conv + 2 fc)
class CNN_2D_v1(nn.Module):
    def __init__(self, in1= 1, out1=16, out2=32, h1=32, nlabels=1 , featmap_resolution=(57,57)):
        super().__init__()
       
        # ARCHITECTURE OF THE NETWORK CONFIGURATION
        # Definition of the parameters of the layers
        [ksize,strid,maxpol,dilation,pad] = [5,1,2,1,1]
        
        # Layers configuration of the network
        # Final length after applying the convolutions
        if type(featmap_resolution) == tuple:
            maxpol2 = 1    
            final_size1 = int((((featmap_resolution[0] +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol)) + 2* pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
            final_size2 = int((((featmap_resolution[1] +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol2)) + 2* pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        else:
            maxpol2 = 2
            final_size1 = int((((featmap_resolution +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol)) + 2* pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
            final_size2 = int((((featmap_resolution +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol2)) + 2* pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol)) 

        self.final_out = final_size1*final_size2*out2
        # Convolutional layers
        self.conv1 = nn.Conv2d(in1, out1, kernel_size=ksize, stride=strid, padding=pad)
        self.conv2 = nn.Conv2d(out1, out2, kernel_size=ksize, stride=strid, padding=pad)
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(out1)
        self.batch_norm2 = nn.BatchNorm2d(out2)
        self.batch_norm3 = nn.BatchNorm1d(h1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_out, h1)
        self.fc2 = nn.Linear(h1, nlabels)      
        # Dropout layer
        self.dropout = nn.Dropout(DROP_OUT)
        # Maxpooling
        self.maxPool1 = nn.MaxPool2d((maxpol, maxpol2))
        self.maxPool2 = nn.MaxPool2d((maxpol, maxpol))
        # Activation
        self.relu = nn.ReLU()     
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        # 1. Convolutional layers with ReLU activation and max pooling
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxPool1(x)
        x = self.dropout(x)
        # 2. Convolutional layers with ReLU activation and max pooling
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxPool2(x)
        # Flatten the output for the dense layers
        x = x.view(-1, self.final_out)
        # Dense layers with ReLU activation and dropout
        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Final dense layers with sigmoid activation 
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

# PYB: 2D-CNN square input. (2 conv + 2 fc)
# Two extra layers for feature correlation (kernels: #feat x 5 and #feat/2 x 5). Results concatenated  
class CNN_2D_v2(nn.Module):
    def __init__(self, in1= 1, out1=16, out2=32, h1=32, nlabels=1 , featmap_resolution=(57,57)):
        super().__init__()

        # ARCHITECTURE OF THE NETWORK CONFIGURATION
        # Definition of the parameters of the layers
        [ksize,strid,maxpol,maxpol2,dilation, pad, pad2] = [5,1,2,1,1,1,0]
        
        # Layers configuration of the network
        # Final length after applying the convolutions
        if type(featmap_resolution) == tuple:
            num_features = featmap_resolution[0]
            level1_1_vertical = int((featmap_resolution[0] + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
            level1_2_vertical = ((featmap_resolution[0] + 2 * pad2 - dilation * (num_features//2 - 1) - 1 + strid) // (strid))
            level1_3_vertical = ((featmap_resolution[0] + 2 * pad2 - dilation * (num_features - 1) - 1 + strid) // (strid))
            
            level1_vertical = int((level1_1_vertical + level1_2_vertical + level1_3_vertical))
            level1_horizontal = int((featmap_resolution[1] + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol2))           
        else:
            num_features = featmap_resolution
            level1_1_vertical = ((featmap_resolution + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
            level1_2_vertical = ((featmap_resolution + 2 * pad2 - dilation * (num_features//2 - 1) - 1 + strid) // (strid))
            level1_3_vertical = ((featmap_resolution + 2 * pad2 - dilation * (num_features - 1) - 1 + strid) // (strid))
            
            level1_vertical = int((level1_1_vertical + level1_2_vertical + level1_3_vertical))         
            level1_horizontal = int((featmap_resolution + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol2))

        final_size1 = int((level1_vertical + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
        final_size2 = int((level1_horizontal + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
        self.final_out = final_size1*final_size2*out2
        # Convolutional layers
        self.conv1_1 = nn.Conv2d(in1, out1, kernel_size=ksize, stride=strid, padding=pad)
        self.conv1_2 = nn.Conv2d(in1, out1, kernel_size=(num_features//2, ksize), stride=strid, padding=(pad2, pad))
        self.conv1_3 = nn.Conv2d(in1, out1, kernel_size=(num_features, ksize), stride=strid, padding=(pad2, pad))  
        self.conv2 = nn.Conv2d(out1, out2, kernel_size=ksize, stride=strid, padding=pad)       
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(out1)
        self.batch_norm2 = nn.BatchNorm2d(out2)
        self.batch_norm3 = nn.BatchNorm1d(h1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_out, h1)
        self.fc2 = nn.Linear(h1, nlabels)
        # Dropout layer
        self.dropout = nn.Dropout(DROP_OUT)
        # Maxpooling
        self.maxPool1_1 = nn.MaxPool2d((maxpol, maxpol2))
        self.maxPool1_2 = nn.MaxPool2d((maxpol2, maxpol2))
        self.maxPool2 = nn.MaxPool2d((maxpol, maxpol))
        # Activation
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        
        # 1A. Convolutional layers with ReLU activation and max pooling
        x_1 = self.conv1_1(x)
        x_1 = self.batch_norm1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.maxPool1_1(x_1) 
        # 1B. Convolutional layers with ReLU activation and max pooling
        x_2 = self.conv1_2(x)
        x_2 = self.batch_norm1(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.maxPool1_2(x_2)     
        # 1C. Convolutional layers with ReLU activation and max pooling
        x_3 = self.conv1_3(x)
        x_3 = self.batch_norm1(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.maxPool1_2(x_3) 
        # Concatenation of the matrices
        x = torch.cat([x_1,x_2,x_3], 2)       
        x = self.dropout(x)       
        # 2. Convolutional layers with ReLU activation and max pooling
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxPool2(x)
        x = self.dropout(x)       
        # Flatten the output for the dense layers
        x = x.view(-1, self.final_out)     
        # Dense layers with ReLU activation and dropout
        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)        
        # Final dense layers with sigmoid activation
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

# PYC: 2D-CNN square input. (2 conv + 2 fc)
# One extra layers for feature correlation (kernels: #feat x 5). Results concatenated  
class CNN_2D_v3(nn.Module):
    def __init__(self, in1= 1, out1=16, out2=32, h1=32, nlabels=1 , featmap_resolution=(57,57)):
        super().__init__()

        # ARCHITECTURE OF THE NETWORK CONFIGURATION
        # Definition of the parameters of the layers
        [ksize,strid,maxpol,maxpol2,dilation, pad, pad2] = [5,1,2,1,1,1,0]

        # Layers configuration of the network
        # Final length after applying the convolutions
        if type(featmap_resolution) == tuple:
            num_features = featmap_resolution[0]
            level1_1_vertical = ((featmap_resolution[0] + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
            level1_3_vertical = ((featmap_resolution[0] + 2 * pad2 - dilation * (num_features - 1) - 1 + strid) // (strid))
            
            level1_vertical = int((level1_1_vertical + level1_3_vertical))
            level1_horizontal = int((featmap_resolution[1] + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol2))

        else:
            num_features = featmap_resolution
            level1_1_vertical = ((featmap_resolution + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
            level1_3_vertical = ((featmap_resolution + 2 * pad2 - dilation * (num_features - 1) - 1 + strid) // (strid))

            level1_vertical = int((level1_1_vertical + level1_3_vertical))            
            level1_horizontal = int((featmap_resolution + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol2))

        final_size1 = int((level1_vertical + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
        final_size2 = int((level1_horizontal + 2 * pad - dilation * (ksize-1) - 1 + strid) // (strid*maxpol))
        self.final_out = final_size1*final_size2*out2
        # Convolutional layers
        self.conv1_1 = nn.Conv2d(in1, out1, kernel_size=ksize, stride=strid, padding=pad)
        self.conv1_3 = nn.Conv2d(in1, out1, kernel_size=(num_features,ksize), stride=strid, padding=(pad2, pad))
        self.conv2 = nn.Conv2d(out1, out2, kernel_size=ksize, stride=strid, padding=pad)
        # # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(out1)
        self.batch_norm2 = nn.BatchNorm2d(out2)
        self.batch_norm3 = nn.BatchNorm1d(h1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_out, h1)
        self.fc2 = nn.Linear(h1, nlabels)
        # Dropout layer
        self.dropout = nn.Dropout(DROP_OUT)
        # Maxpooling
        self.maxPool1_1 = nn.MaxPool2d((maxpol, maxpol2))
        self.maxPool1_2 = nn.MaxPool2d((maxpol2, maxpol2))
        self.maxPool2 = nn.MaxPool2d((maxpol, maxpol))
        # Activation
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):     
        # 1A. Convolutional layers with ReLU activation and max pooling
        x_1 = self.conv1_1(x)
        x_1 = self.batch_norm1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.maxPool1_1(x_1)
        # 1B. Convolutional layers with ReLU activation and max pooling
        x_3 = self.conv1_3(x)
        x_3 = self.batch_norm1(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.maxPool1_2(x_3)
        # Concatenation of the matrices
        x = torch.cat([x_1,x_3], 2)     
        x = self.dropout(x)       
        # 2. Convolutional layers with ReLU activation and max pooling
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxPool2(x)
        x = self.dropout(x)      
        # Flatten the output for the dense layers
        x = x.view(-1, self.final_out)       
        # Dense layers with ReLU activation and dropout
        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Final dense layers with sigmoid activation
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

# PYD: 1D-CNN non-necessary squared input. (1 conv + 1 fc)
class CNN_1D_v1(nn.Module):
    def __init__(self, in1= 57, out1=64, nlabels=1 , featmap_resolution=10):
        super().__init__()

        # ARCHITECTURE OF THE NETWORK CONFIGURATION
        # Definition of the parameters of the layers
        [ksize,strid,pad,dilation,maxpol] = [5,1,1,1,2]

        # Layers configuration of the network
        # Final length after applying the convolutions
        Lout1 = int((featmap_resolution +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        self.final_out = Lout1*out1
        # Convolutional layers
        self.conv1 = nn.Conv1d(in1, out1, kernel_size=ksize, stride=strid, padding=pad, dilation=dilation)
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(out1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_out, nlabels)
        # Dropout layer
        self.dropout = nn.Dropout(DROP_OUT)
        # Maxpooling
        self.maxPool = nn.MaxPool1d(maxpol)
        # Activation
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):      
        # Convolutional layers with ReLU activation and max pooling
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxPool(x)
        x = self.dropout(x)          
        # Flatten the output for the dense layers
        x = x.view(-1, self.final_out)       
        # Final dense layers with sigmoid activation
        x = self.fc1(x)
        x = self.sigmoid(x)      
        return x

# PYE: 1D-CNN non-necessary squared input. (2 conv + 1 fc)
class CNN_1D_v2(nn.Module): 
    def __init__(self, in1= 57, out1=32, out2 =64, nlabels=1 , featmap_resolution=10):
        super().__init__()

        # ARCHITECTURE OF THE NETWORK CONFIGURATION
        # Definition of the parameters of the layers
        [ksize,strid,pad,dilation,maxpol] = [5,1,1,1,2]
        
        # Layers configuration of the network
        # Final length after applying the convolutions
        Lout1 = int((featmap_resolution +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        Lout2 = int((Lout1 +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        self.final_out = Lout2*out2
        # Convolutional layers
        self.conv1 = nn.Conv1d(in1, out1, kernel_size=ksize, stride=strid, padding=pad,dilation=dilation)
        self.conv2 = nn.Conv1d(out1, out2, kernel_size=ksize, stride=strid, padding=pad,dilation=dilation)        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(out1)
        self.batch_norm2 = nn.BatchNorm1d(out2)
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_out, nlabels)
        # Droput layer
        self.dropout = nn.Dropout(DROP_OUT)
        # Maxpooling
        self.maxPool = nn.MaxPool1d(maxpol)
        # Activation
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        # 1. Convolutional layer with ReLU activation and max pooling
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxPool(x)
        x = self.dropout(x)
        # 2. Convolutional layer with ReLU activation and max pooling
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxPool(x)        
        # Flatten the output for the dense layers
        x = x.view(-1, self.final_out)       
        # Final dense layers with ReLU activation and dropout
        x = self.fc1(x)
        x = self.sigmoid(x)
        
        return x

# PYF: 1D-CNN non-necessary squared input. (1 conv + 2 fc)
class CNN_1D_v3(nn.Module):
    def __init__(self, in1= 57, out1=64, h1=32, nlabels=1 , featmap_resolution=10):
        
        super().__init__()

        # ARCHITECTURE OF THE NETWORK CONFIGURATION
        # Definition of the parameters of the layers
        [ksize,strid,pad,dilation,maxpol] = [5,1,1,1,2]

        # Layers configuration of the network
        # Final length after applying the convolutions
        Lout1 = int((featmap_resolution +2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        self.final_out = Lout1*out1
        # Convolutional layers
        self.conv1 = nn.Conv1d(in1, out1, kernel_size=ksize, stride=strid, padding=pad,dilation=dilation)   
        # # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(out1)
        self.batch_norm3 = nn.BatchNorm1d(h1)
        # v1
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_out, h1)
        self.fc2 = nn.Linear(h1, nlabels)
        # Dropout layer
        self.dropout = nn.Dropout(DROP_OUT)
        # Maxpooling 
        self.maxPool = nn.MaxPool1d(maxpol)
        # Activation
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        
        # Convolutional layer with ReLU activation and max pooling
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxPool(x)
        x = self.dropout(x)        
        # Flatten the output for the dense layers
        x = x.view(-1, self.final_out)    
        # Dense layer with ReLU activation and dropout
        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Final dense layers with sigmoid activation
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

# PYG: MobileNetV3 small - https://arxiv.org/abs/1905.02244
class MobileNetV3Custom(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        width_mult = 0.35
        base = models.mobilenet_v3_small(width_mult=width_mult)

        # Ajustar la primera capa para 1 canal (en vez de 3)
        first_conv = base.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        # Si quieres usar los pesos preentrenados en el canal único, promediar:
        new_first_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        base.features[0][0] = new_first_conv

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Hace que funcione con 57x57
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


"""
----------------------------------------------------------------------------------------------------
Model definition used for deployment (KERAS)
----------------------------------------------------------------------------------------------------
Specific transformations may be applied to ensure compatibility with the target device and with the 
supported operations. Currently, only a limited subset of operations is possible to run, and 
on-device training is still not supported in TFLite.

one-dimensional convolutions not supported -> two-dimensional configuration with kernels (1 x k)
final activation function sigmoid -> softmax

For the continual learning implementation the Adam optimizer -> SGD

"""

# TF1: 1D-CNN (1 conv + 2 fc - sigmoid)
class CNN_1d_tensorflow_sigmoid (keras.Model):
    
    def __init__(self, out1=64, h1=32, nlabels=1, final_activation = 'sigmoid'):
        super().__init__()
        # Definition of the parameters of the layers
        [ksize, maxpol] = [5,2]
        # Convolutional layer
        self.conv = layers.Conv1D(filters=out1, kernel_size=ksize, padding="valid")
        # Batch normalization layer
        self.batch_norm = layers.BatchNormalization()
        # Flatten layer
        self.flatten = layers.Flatten()
        # Fully connected layers
        self.fc1 = layers.Dense(h1, name="dense_1")
        self.fc2 = layers.Dense(nlabels, name="predictions")
        # Dropout layer
        self.dropout = layers.Dropout(0.2)
        # Maxpooling
        self.maxPool = layers.MaxPool1D(maxpol)
        # Activation
        self.relu = layers.Activation(activation='relu')
        self.final_act = layers.Activation(activation=final_activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxPool(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.final_act(x)
        
        return x

# TF2: 1D-CNN (1 conv + 2 fc - softmax)
# *IMPORTANTE: loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
class CNN_1d_tensorflow_softmax (keras.Model):
    
    def __init__(self, out1=64, h1=32, nlabels=2, final_activation = 'softmax'):
        super().__init__()
        # Definition of the parameters of the layers
        [ksize, maxpol] = [5,2]
        # Convolutional layer
        self.conv = layers.Conv1D(filters=out1, kernel_size=ksize, padding="valid")
        # Batch normalization layer
        self.batch_norm = layers.BatchNormalization()
        # Flatten layer
        self.flatten = layers.Flatten()
        # Fully connected layers
        self.fc1 = layers.Dense(h1, name="dense_1")
        self.fc2 = layers.Dense(nlabels, name="predictions")
        # Dropout layer
        self.dropout = layers.Dropout(0.2)
        # Maxpooling
        self.maxPool = layers.MaxPool1D(maxpol)
        # Activation
        self.relu = layers.Activation(activation='relu')
        self.final_act = layers.Activation(activation=final_activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxPool(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.final_act(x)
        
        return x

# TF3: 2D-CNN (microcontroller deployment)
class CNN_2d_tensorflow_softmax (keras.Model):
    
    def __init__(self, out1=64, h1=32, nlabels=2, final_activation = 'softmax'):
        super().__init__()
        # Definition of the parameters of the layers
        [ksize, maxpol] = [5,2]
        # Convolutional layer
        self.conv = layers.Conv2D(filters=out1, kernel_size=(1,ksize), padding="valid")
        # Batch normalization layer
        self.batch_norm1 = layers.BatchNormalization()
        # self.batch_norm2 = layers.BatchNormalization()
        # Flatten layer
        self.flatten = layers.Flatten()
        # Fully connected layers
        # self.fc1 = layers.Dense(h1, name="dense_1")
        self.fc2 = layers.Dense(nlabels, name="predictions")
        # Dropout layer
        self.dropout = layers.Dropout(0.2)
        # Maxpooling
        self.maxPool = layers.MaxPool2D((1,maxpol))
        # Activation
        self.relu = layers.Activation(activation='relu')
        self.final_act = layers.Activation(activation=final_activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxPool(x)
        
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.batch_norm2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.final_act(x)
        
        return x
