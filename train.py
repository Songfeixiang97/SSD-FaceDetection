#!/usr/bin/env python
# coding: utf-8

import torch
from data_generator import Generator
from model import Model, UDA
from torch import nn
import numpy as np
from tools import load_model, save_model, accuracy, accuracy_uda
import os
import csv
import cv2 as cv
import pandas as pd
from torch.optim import Adam, lr_scheduler, SGD
from torch.nn import MSELoss, CrossEntropyLoss, Linear, Conv2d, SmoothL1Loss, NLLLoss
from torch.nn.init import xavier_normal_, constant_
    

class Train():
    def __init__(self,
                 batch = 64,
                 Class = 2,
                 box_num = [5,4,4,4,4,4],
                 lr = 0.001,
                 load_pretrain = False,
                 model = 1,
                 aug = False,
                 lr_updata = False,
                 uda = False
                 ):
    
        if model == 1:
    
            self.model = Model(Class = Class, box_num = box_num)
            self.m = 1
    
        elif model == 2:
    
            self.model = Inception(5)
            self.m = 2
    
        self.batch = batch
        self.Class = Class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
    
            print('Lets use', torch.cuda.device_count(), 'GPUs!')
            self.model = nn.DataParallel(self.model)
            
        self.uda = uda
        self.UDA_model = UDA().to(self.device)
        self.UDA_img = self.get_test_img()
        self.model = self.model.to(self.device)
        self.load_pretrain = load_pretrain
        self.optimizer = Adam(self.model.parameters(), lr = lr, weight_decay = 0.0005)
        self.optimizer1 = Adam(self.UDA_model.parameters(), lr = lr, weight_decay = 0.0005)

        self.lambda1 = lambda epoch : 0.95**epoch
        self.lambda2 = lambda epoch : 0.99**epoch
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda = self.lambda1)
        self.scheduler1 = lr_scheduler.LambdaLR(self.optimizer1, lr_lambda = self.lambda2)
    
        if self.load_pretrain == True:
    
            if lr_updata==True:
                load_model(self.model, m=self.m)
    
            else:
    
                load_model(self.model, self.optimizer, self.scheduler, self.m)
    
            self.train_loss = ['train_loss']
            self.train_acc = ['train_acc']
            self.verify_loss = ['verify_loss']
            self.verify_acc = ['verify_acc']
            self.lr = ['learning rate']

            data = pd.read_csv('./train_data'+str(self.m)+'.csv')
            self.train_loss.extend(data['train_loss'].tolist())
            self.train_acc.extend(data['train_acc'].tolist())
            self.verify_loss.extend(data['verify_loss'].tolist())
            self.verify_acc.extend(data['verify_acc'].tolist())
            self.lr.extend(data['learning rate'].tolist())
    
        else:
    
            self.train_loss = ['train_loss']
            self.train_acc = ['train_acc']
            self.verify_loss = ['verify_loss']
            self.verify_acc = ['verify_acc']
            self.lr = ['learning rate']

            for i in self.model.parameters():
                if len(i.shape)>=2:
                    xavier_normal_(i)
    
        self.train_generator = Generator(batch = batch, mode = 'train',aug=aug)
        self.train_generator2 = Generator(batch = batch, mode = 'val',aug=aug)
        self.test_generator = Generator(batch = batch, mode = 'val')

    def evaluate(self):
    
        self.model.eval()
        loss = 0.0
        acc = 0.0
    
        with torch.no_grad():
    
            for i in range(2):
    
                images, Offset, Judge1, Judge2, true_box_coors = next(self.test_generator)
    
                images, Offset, Judge1, Judge2 = images.to(self.device), Offset.to(self.device), Judge1.to(self.device), Judge2.to(self.device)
    
                class_sum, loc_sum = self.model(images)
                loss_l, loss_c = self.Loss(class_sum, loc_sum, Offset, Judge1, Judge2)
                loss += (10*loss_l + loss_c)
                acc += accuracy(class_sum, loc_sum, true_box_coors)
    
    
        self.model.train()
        return loss/2, acc/2
        
    
    def get_test_img(self):
        filename = os.listdir('./test')
        return ['./test/'+i for i in filename]
    
    
    def Loss(self, class_sum, loc_sum, Offset, Judge1, Judge2):
        '''
        class_sum shape[batch,12728,2] Tensor
        loc_sum shape[batch,12728,4] Tensor
        Offset shape[batch,12728,4] Tensor
        Judge1 shape[batch,12728,1] Tensor
        Judge2 shape[batch,12728] LongTensor
        '''
        # 如果为多类别Judge1 = Judge1!=0
    
        Loss_con = torch.nn.CrossEntropyLoss(weight = torch.Tensor([0.05,1]),reduction='none').to(self.device)
    
        Loss_loc = torch.nn.SmoothL1Loss(reduction='none').to(self.device)
    
        w = 1/torch.sum(Judge1,dim=1).unsqueeze(-1)
        v = w.squeeze(-1).expand(w.shape[0],Judge2.shape[1]).contiguous().view(-1)
        loss_l = torch.sum(Loss_loc(loc_sum*Judge1, Offset)*w)
        loss_c = torch.sum(Loss_con(class_sum.view(-1,self.Class),Judge2.view(-1))*v)
        return loss_l,loss_c
    
                          
    def train(self, epoch_num = 10, step_one_epoch = 20, save_frq = 1000, evl_frq = 500):
    
        self.model.train()
    
        if self.load_pretrain == True:
    
            global_step = int(os.listdir('./model' + str(self.m))[0])
            epoch = global_step//step_one_epoch
            _, max_acc = self.evaluate()
    
            self.train_loss = self.train_loss[0:global_step//evl_frq+1]
            self.train_acc = self.train_acc[0:global_step//evl_frq+1]
            self.verify_loss = self.verify_loss[0:global_step//evl_frq+1]
            self.verify_acc = self.verify_acc[0:global_step//evl_frq+1]
            self.lr = self.lr[0:global_step//evl_frq+1]
    
        else:
    
            global_step = 0
            epoch = 1
            max_acc = 0.0
        
        if self.uda == True:
            
            Label1 = torch.LongTensor([0 for i in range(self.batch)]).to(self.device)
            Label2 = torch.LongTensor([1 for i in range(self.batch)]).to(self.device)
            UDA_loss = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)
            D = True
       
        else:
 
            D = False
        
        while(epoch<epoch_num):
    
            for i in range(step_one_epoch):
    
                self.optimizer.zero_grad()
                if self.uda == True:
                    self.optimizer1.zero_grad()
   
                if i%10 == 0: 
                    images, Offset, Judge1, Judge2, true_box_coors = next(self.train_generator2) #加入验证集训练
                else:
                    images, Offset, Judge1, Judge2, true_box_coors = next(self.train_generator)
    
                images, Offset, Judge1, Judge2 = images.to(self.device), Offset.to(self.device), Judge1.to(self.device), Judge2.to(self.device)
                
                if self.uda == False or D == False:
                    class_sum, loc_sum = self.model(images)

                    loss_l, loss_c = self.Loss(class_sum, loc_sum, Offset, Judge1, Judge2)
                    loss = 10*loss_l + loss_c
                    loss.backward()
               # torch.nn.utils.clip_grad_norm_(self.model.parameters(),10)
                    self.optimizer.step()
                    
                    flag = True
                
                if self.uda == True:
                    '''
                    UDA对抗领域自适应
                    '''
                    
                    self.optimizer.zero_grad()
                    self.optimizer1.zero_grad()
                    if D == True:
                        '''
                        训练域分辨器
                        test_img 输出域标签为1
                        训练集输出域标签为0
                        '''
                        test_imgs = np.random.choice(self.UDA_img, self.batch)
                        test_images = []
                    
                        for img_path in test_imgs:
                            img = cv.imread(img_path)
                            img = cv.resize(img,(480,270))
                            test_images.append(img)
                        
                        test_images = np.array(test_images)/255.
                        test_images = torch.Tensor(test_images).permute(0,3,1,2).to(self.device)
                        feature1 = self.model.feature1(images)
                        feature2 = self.model.feature1(test_images)
                        feature = torch.cat((feature1,feature2),dim=0)
                        logits = self.UDA_model(feature)
                        uda_acc = accuracy_uda(logits, torch.cat((Label1,Label2),dim=0))
                        loss_uda = UDA_loss(logits, torch.cat((Label1,Label2),dim=0))
                        loss_uda.backward()
                        self.optimizer1.step()
                        
                        flag = False
                        if uda_acc>0.9:
                            D = False
                        
                        
                    elif D == False:
                        '''
                        训练主模型
                        训练集的输出域标签设为1，
                        令其输出域和实际测试集一样
                        '''
                        
                        self.optimizer.zero_grad()
                        self.optimizer1.zero_grad()
                        
                        
                        feature1 = self.model.feature1(images)
                        logits = self.UDA_model(feature1)
                        uda_acc = accuracy_uda(logits, Label2)
                        loss_uda = UDA_loss(logits, Label2)
                        loss_uda.backward()
                        self.optimizer.step()
                        
                        if uda_acc>0.9:
                            D = True
                
    
                if global_step%evl_frq==0:
                    if flag == True: 
                        train_loss = loss
                        train_acc = accuracy(class_sum, loc_sum, true_box_coors)
                        verify_loss, verify_acc = self.evaluate()
    
                        print('step:     {:}     learning rate:     {:6f}'.format(global_step,self.scheduler.get_lr()[0]))
    
                        print('loss_l:     {:4f}         loss_c:     {:4f}'.format(loss_l, loss_c))
    
                        print('train_loss:     {:4f}     train_acc:     {:4f}'.format(train_loss, train_acc))
    
                        print('verify_loss:    {:4f}     verify_acc:    {:4f}'.format(verify_loss, verify_acc))
    
                        self.train_loss.append(float(train_loss))
                        self.train_acc.append(train_acc)
                        self.verify_loss.append(float(verify_loss))
                        self.verify_acc.append(verify_acc)
                        self.lr.append(float(self.scheduler.get_lr()[0]))
    
                        train_data = [self.lr,self.train_loss,self.train_acc,self.verify_loss,self.verify_acc]
                        train_data = np.array(train_data).T
    
                        with open('train_data'+str(self.m)+'.csv','w') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(train_data)
    
                        if global_step%save_frq==0:
                            if verify_acc >= max_acc:
                                save_model(self.model,self.optimizer,self.scheduler, global_step, self.m)
                                max_acc = verify_acc
                                if max_acc>0.96:
                                    epoch=epoch_num
       
                global_step += 1
    
            epoch+=1
            self.scheduler.step(epoch)
            if self.uda == True:
                self.scheduler1.step(epoch)
