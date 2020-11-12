#!/usr/bin/env python
# coding: utf-8

from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 
import os
import shutil
import random
import numpy as np
import torch
import cv2 as cv
import math
import copy as cp
from data_augment import data_aug, data_aug2
from tools import center,feature_map_size, box_num, box_size
    

class Generator():
    
    def __init__(self,
                 batch = 64,
                 mode = 'train',
                 aug = False
                ):
    
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
        self.pre_box_center_coor = center(feature_map_size, box_num, box_size)#中心坐标，用来计算offset
    
        self.pre_box_coor = np.array(self.pre_box_center_coor)
        self.pre_box_coor[:,0] = self.pre_box_coor[:,0] - self.pre_box_coor[:,2]/2
        self.pre_box_coor[:,1] = self.pre_box_coor[:,1] - self.pre_box_coor[:,3]/2
        self.pre_box_coor = torch.Tensor(self.pre_box_coor).to(self.device)#左上wh坐标用来计算IOU
    
        self.img_path, self.true_box_coor = self.load_mat(mode)
        self.index = 0
        self.batch = batch
        self.aug = aug
        self.all = np.arange(len(self.img_path))
        self.len = len(self.img_path)
        random.shuffle(self.all)
        
    
    def load_mat(self, mode):
    
        m = loadmat('./WIDER_'+mode+'/'+mode+'.mat')
        data = m['face_bbx_list']
        img_name = m['file_list']
        img_path = []
        bbx_list = []
        filename = os.listdir('./WIDER_'+mode+'/images')
        filename.sort()
    
        for i in range(61):
    
            for j,file in enumerate(img_name[i][0]):
    
                if len(data[i][0][j][0]) < 4:
    
                    for k in range(10):
                        img_path.append('./WIDER_'+mode+'/images/'+filename[i]+'/'+file[0][0]+'.jpg')
                        bbx_list.append(data[i][0][j][0])
    
                elif len(data[i][0][j][0]) < 8:
    
                    img_path.append('./WIDER_'+mode+'/images/'+filename[i]+'/'+file[0][0]+'.jpg')
                    bbx_list.append(data[i][0][j][0])
    
        return img_path, bbx_list
    
    def __iter__(self):
        return
    
    
    def IOU(self, box1, box2):
    
        """
        可以广播
        输入左上坐标和wh
        Returns the IoU of two bounding boxes 
        得到bbox的坐标
        """
        # Get the coordinates of bounding boxes
    
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2]+box1[:, 0], box1[:, 3]+box1[:, 1]
    
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2]+box2[:, 0], box2[:, 3]+box2[:, 1]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Intersection area
        if torch.cuda.is_available():
    
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
                inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    
        else:
    
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
                inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou
    
    def Offset(self,box1, box2):
        '''
        使用中心坐标计算offset
        '''
        
        Ox = (box2[0]-box1[0])/box1[2]
        Oy = (box2[1]-box1[1])/box1[3]
        Ow = math.log(box2[2]/box1[2])
        Oh = math.log(box2[3]/box1[3])
        
        return [Ox,Oy,Ow,Oh]
    
    def __next__(self):
    
        if self.index + self.batch>=self.len:
            self.index = 0
            random.shuffle(self.all)
    
        images = []
        Offset = []
        Judge = []
        true_box_coors = []
    
        for i in self.all[self.index:self.index+self.batch]:
            
            img = cv.imread(self.img_path[i])
            img = cv.resize(img,(480,270))
           # img = (img-img.mean())/max(img.std(),1/360)
    
            if self.aug == True:
    
                if len(self.true_box_coor[i]<3):
    
                    for Bo in self.true_box_coor[i]:
    
                        img[int(Bo[1])+int(Bo[3]/2):int(Bo[1])+int(Bo[3]),int(Bo[0]):int(Bo[0])+int(Bo[2])] = data_aug2( img[int(Bo[1])+int(Bo[3]/2):int(Bo[1])+int(Bo[3]),int(Bo[0]):int(Bo[0])+int(Bo[2])]) 
    
            images.append(img)
            true_box_coors.append(self.true_box_coor[i])
            '''
            计算IOU
            true_box_center_coor的xy是中心坐标，便于计算offset
            self.true_box_coor[i]的xy是左上坐标，便于计算IOU
            '''
            true_box_center_coor = cp.copy(self.true_box_coor[i])
            true_box_center_coor[:,0] = true_box_center_coor[:,0] + true_box_center_coor[:,2]/2
            true_box_center_coor[:,1] = true_box_center_coor[:,1] + true_box_center_coor[:,3]/2
    
            for k,true_box in enumerate(self.true_box_coor[i]):
    
                true_box = torch.Tensor([true_box]).to(self.device)
    
                if k==0:
                    iou = self.IOU(true_box, self.pre_box_coor).unsqueeze(0)
    
                else:
                    iou = torch.cat((iou,self.IOU(true_box, self.pre_box_coor).unsqueeze(0)),dim=0)
                #iou.shape = [true_box_num,12728]
    
            max_iou, index0 = torch.max(iou,dim=0)#shape[12728]每个候选框对应的max_iou和true_box序号
            index1 = torch.argmax(iou,dim=1)#shape[true_box_num]每个true_box对应的iou最大的候选框序号
            judge = (max_iou>0.6).tolist() #将max_iou大于0.6的位置判别为1,即候选框为Pos  shape = [12728]
            '''
            计算offset
            true_box_center_coor的xy是中心坐标，便于计算offset
            根据index0和index1和judge来计算offset并将其存储在list中
            '''
            offset = [[0.,0.,0.,0.] for k in range(12728)]
            for j,k in enumerate(judge):
    
                if k==1:
    
                    offset[j] = self.Offset(self.pre_box_center_coor[j], true_box_center_coor[index0[j]])
    
            for j,k in enumerate(index1):
    
                judge[k] = 1
    
                offset[k] = self.Offset(self.pre_box_center_coor[k], true_box_center_coor[j])
    
            Offset.append(offset)
            Judge.append(judge)
    
        Judge1 = torch.Tensor(Judge).unsqueeze(2)
        Judge2 = torch.LongTensor(Judge)
        Offset = torch.Tensor(Offset)
        images = np.array(images)
    
        if self.aug == True:
            images = data_aug(images)
    
        images = images/255.
        images = torch.Tensor(images).permute(0,3,1,2)
        self.index += self.batch
    
        return images, Offset, Judge1, Judge2, true_box_coors