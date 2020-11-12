#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import *

class Model(nn.Module):
    def __init__(self,
                 Class = 2,
                 box_num = [5,4,4,4,4,4]
                ):
        super(Model,self).__init__()
        self.Class = Class
        self.box_num = box_num
    
        self.feature1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
           # BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
           # BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
           # BatchNorm2d(256),
            ReLU(inplace=True),
    
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
           # BatchNorm2d(512),
            ReLU(inplace=True),
    
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
           # BatchNorm2d(512),
            ReLU(inplace=True)
        )
    
        self.class1 = Conv2d(512, box_num[0]*Class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.loc1 = Conv2d(512, box_num[0]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        
        self.feature2 = nn.Sequential(
            Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
            #BatchNorm2d(1024),
            ReLU(inplace=True)
        )
    
        self.class2 = Conv2d(1024, box_num[1]*Class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.loc2 = Conv2d(1024, box_num[1]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    
        self.feature3 = nn.Sequential(
            Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
           # BatchNorm2d(512),
            ReLU(inplace=True)
        )
    
        self.class3 = Conv2d(512, box_num[2]*Class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.loc3 = Conv2d(512, box_num[2]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    
        self.feature4 = nn.Sequential(
            Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
           # BatchNorm2d(256),
            ReLU(inplace=True)
        )
    
        self.class4 = Conv2d(256, box_num[3]*Class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.loc4 = Conv2d(256, box_num[3]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        
        self.feature5 = nn.Sequential(
            Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
           # BatchNorm2d(256),
            ReLU(inplace=True)
        )
    
        self.class5 = Conv2d(256, box_num[4]*Class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.loc5 = Conv2d(256, box_num[4]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    
        self.feature6 = nn.Sequential(
            Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
    
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
           # BatchNorm2d(256),
            ReLU(inplace=True)
        )
    
        self.class6 = Conv2d(256, box_num[5]*Class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
        self.loc6 = Conv2d(256, box_num[5]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    
    def forward(self, x):
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        feature5 = self.feature5(feature4)
        feature6 = self.feature6(feature5)
        
         #然后按照预选框坐标的排列顺序view
        class1 = self.class1(feature1).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[0]*self.Class).contiguous().view(x.shape[0],-1,self.Class)
    
        class2 = self.class2(feature2).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[1]*self.Class).contiguous().view(x.shape[0],-1,self.Class)
    
        class3 = self.class3(feature3).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[2]*self.Class).contiguous().view(x.shape[0],-1,self.Class)
    
        class4 = self.class4(feature4).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[3]*self.Class).contiguous().view(x.shape[0],-1,self.Class)
    
        class5 = self.class5(feature5).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[4]*self.Class).contiguous().view(x.shape[0],-1,self.Class)
    
        class6 = self.class6(feature6).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[5]*self.Class).contiguous().view(x.shape[0],-1,self.Class)
    
    
        class_sum = torch.cat((class1,class2,class3,class4,class5,class6),dim=1)
        
    
    
        loc1 = self.loc1(feature1).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[0]*4).contiguous().view(x.shape[0],-1,4)
    
        loc2 = self.loc2(feature2).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[1]*4).contiguous().view(x.shape[0],-1,4)
    
        loc3 = self.loc3(feature3).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[2]*4).contiguous().view(x.shape[0],-1,4)
    
        loc4 = self.loc4(feature4).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[3]*4).contiguous().view(x.shape[0],-1,4)
    
        loc5 = self.loc5(feature5).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[4]*4).contiguous().view(x.shape[0],-1,4)
    
        loc6 = self.loc6(feature6).permute(0,2,3,1).view(x.shape[0],-1,self.box_num[5]*4).contiguous().view(x.shape[0],-1,4)
    
        loc_sum = torch.cat((loc1,loc2,loc3,loc4,loc5,loc6),dim=1)
    
    
        return class_sum, loc_sum
        #class_sum shape[batch,12728,2], loc_sumshape[batch,12728,4]

        
        
        

class UDA(nn.Module):
    '''
    对抗领域自适应
    无监督学习
    '''
    def __init__(self):
        
        super(UDA, self).__init__()
        
        self.conv = nn.Sequential(
            
            Conv2d(512,256,3,2,1),
            ReLU(inplace = True),
            Conv2d(256,256,3,2,1),
            ReLU(inplace = True),
            Conv2d(256,256,3,2,1),
            ReLU(inplace = True),
            Conv2d(256,128,3,1),
            ReLU(inplace = True)
            
        )
        
        self.fc = nn.Sequential(
            
            Dropout(0.5),
            Linear(2304,1024),
            ReLU(inplace = True),
            Dropout(0.5),
            Linear(1024,2)
            
        )
    
    
    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        
        return x
