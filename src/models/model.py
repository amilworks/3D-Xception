
"""
    3DXception Network Architecture.
    Author/Maintainer: Amil Khan

    File any issues on GitHub! I won't leave you hanging! Maybe...
"""


import torch.nn as nn
import torch.nn.functional as F
import torch



class SeparableConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv3d,self).__init__()

        self.conv1     = nn.Conv3d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv3d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip   = nn.Conv3d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm3d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv3d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm3d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv3d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv3d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm3d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool3d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception3d(nn.Module):
    """
    
    +----------------------------------------------------------------------------------+    
    
        3DXception 
        Author/Maintainer: Amil Khan


        Specifically made for the Facebook DeepFake Challenge 2020, but can easily be 
        adapted for general video classification. This model, heavily influenced by
        the awesome Francois Chollet, proposes a novel deep convolutional neural network
        architecture inspired by Inception, where Inception modules have been replaced
        with depthwise separable 3D convolutions.
        
        Highly recommend reading Chollet's paper:
        
            Xception: Deep Learning with Depthwise Separable Convolutions
            https://arxiv.org/pdf/1610.02357.pdf
            
        3DXception is an implementation of Xception in 3D for video and volumetric
        data classification. 

        If you used the custom dataloader shipped with this repo, you should be good 
        to go in terms of running out of the box. If you are using your own custom 
        video dataloader, then turn on the print statements if necessary to fine tune
        the model for your input data.
        
    
    +----------------------------------------------------------------------------------+
    """
    def __init__(self, num_classes=1):
        """ 
        If you do not have a binary classification problem, please set the number of 
        classes to however many you have, or else you may end up like me without a
        NIPS paper. 
        
        
        Parameters:
        -----------
        num_classes : Int 
        
        """
        super(Xception3d, self).__init__()
        
        self.num_classes = num_classes

#----------------------------------------------------------------------------
# Reduce the size/dimension of the input videos 

        self.downsize = nn.MaxPool3d((10,3,3))


#----------------------------------------------------------------------------
# Entry Flow
        self.conv1   = nn.Conv3d(3, 32, 3,1, 1, bias=False)
        self.bn1     = nn.BatchNorm3d(32)
        self.relu1   = nn.ReLU(inplace=True)

        self.conv2   = nn.Conv3d(32,64,3,bias=False)
        self.bn2     = nn.BatchNorm3d(64)
        self.relu2   = nn.ReLU(inplace=True)

        self.block1  = Block(64,128,3,2,start_with_relu=False,grow_first=True)
        self.block2  = Block(128,256,3,1,start_with_relu=True,grow_first=True)
        self.block3  = Block(256,728,3,2,start_with_relu=True,grow_first=True)


#----------------------------------------------------------------------------
# Middle Flow

        self.block4  = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5  = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6  = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7  = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block8  = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9  = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10 = Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11 = Block(728,728,3,1,start_with_relu=True,grow_first=True)


#----------------------------------------------------------------------------
# Exit Flow

        self.block12 = Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        self.conv3   = SeparableConv3d(1024,1536,3,1,1)
        self.bn3     = nn.BatchNorm3d(1536)
        self.relu3   = nn.ReLU(inplace=True)
        self.conv4   = SeparableConv3d(1536,2048,3,1,1)
        self.bn4     = nn.BatchNorm3d(2048)


#----------------------------------------------------------------------------
# Audio Implementation

#         self.aud_batch_norm = nn.BatchNorm1d(1)
#         self.lstm = nn.LSTM(400320,10, bidirectional=True)
#         self.aud_conv1  = nn.Conv1d(1, 2048, kernel_size=11, padding=1, stride=1)
#         self.aud_pool   = nn.MaxPool1d(40, padding=1)
#         self.aud_fc1  = nn.Linear(10007, 1)


#----------------------------------------------------------------------------
# Fully Connected Layers

        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, num_classes)
#         self.fc3 = nn.Linear(84, 1)


    def features(self, input, audio=None):
        
#         y = (audio - audio.mean()) /  (audio.max()- audio.mean())
#         self.lstm.flatten_parameters()
        x = self.downsize(input)
#         print('Layer 1 Shape. ', x.shape)
        x = self.conv1(x)
#         print('Layer 1 Shape. ', x.shape)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
#         print('Layer 2 Shape. ', x.shape)
        x = self.block2(x)
#         print('Layer 2 Shape. ', x.shape)
        x = self.block3(x)
#         print('Layer 3 Shape. ', x.shape)
        x = self.block4(x)
#         print('Layer 4 Shape. ', x.shape)        
        x = self.block5(x)
        x = self.block6(x)
#         print('Layer 3 Shape. ', x.shape)
        x = self.block7(x)
#         print('Layer 7 Shape. ', x.shape)

#         print('Layer 3 Shape. ', x.shape)
        x = self.block8(x)
        x = self.block9(x)
# #         print('Layer 9 Shape. ', x.shape)
        x = self.block10(x)
#         print('Layer 10 Shape. ', x.shape)
        x = self.block11(x)
#         print('Layer 11 Shape. ', x.shape)
        x = self.block12(x)
#         print('Layer 12 Shape. ', x.shape)
        x = self.conv3(x)
#         print('ConvLayer 3 Shape. ', x.shape)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        
#         y = self.aud_batch_norm(y)
#         y = self.lstm(y)[0]
#         y = self.aud_batch_norm(y)
        
               
        return x #, y

    def logits(self, features, aud=None):
        x = nn.ReLU(inplace=True)(features)
#         y = nn.ReLU(inplace=True)(aud)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
#         print(x.shape, y.shape)
#         y = self.aud_fc1(y) 
#         print(x.shape, y.shape)
#         x = torch.cat((x, y.view(1, y.size(-1))), dim=1)
#         print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
#         x = self.fc3(x)
#         print(x.shape)
        return x

    def forward(self, input, audio=None):
#       x, y = self.features(input, audio)
        x = self.features(input)
        x = self.logits(x)
        return x