#!/usr/bin/env python
# coding: utf-8

import torch
import shutil
import os
from torch import nn
import math
import cv2 as cv

#box_size 为k-means得到的预选框的宽和高
    
box_size = [[1.9979231357574463, 2.1740126609802246],
            [2.878885269165039, 3.2853729724884033],
            [3.983743906021118, 4.229082107543945],
            [4.490837574005127, 5.660663604736328],
            [6.171291351318359, 6.180200576782227],
            [5.667664527893066, 8.211201667785645],
            [7.979061603546143, 8.73978328704834],
            [8.327091217041016, 13.036238670349121],
            [10.593515396118164, 10.945464134216309],
            [12.808866500854492, 14.513771057128906],
            [17.269454956054688, 17.22980308532715],
            [14.225340843200684, 21.617671966552734],
            [20.769611358642578, 22.505247116088867],
            [24.10456657409668, 29.25560760498047],
            [31.61034393310547, 32.94377899169922],
            [35.333595275878906, 44.258445739746094],
            [46.83838653564453, 50.70481872558594],
            [51.26889419555664, 68.76127624511719],
            [73.70413970947266, 68.53812408447266],
            [70.82294464111328, 94.19425201416016],
            [96.25972747802734, 113.97500610351562],
            [119.20806121826172, 150.9849395751953],
            [157.1614532470703, 191.70779418945312],
            [226.3541717529297, 134.91282653808594],
            [223.72767639160156, 244.4600372314453]]

feature_map_size = [[33,60],
                    [17,30],
                    [9,15],
                    [5,8],
                    [3,6],
                    [1,4]]

box_num = [5,4,4,4,4,4]

    
def pre_box_select(index, feature_map_size, box_num, box_size):
    '''
    根据索引index搜寻预选框,输出其想xywh中心坐标
    threshold = [0, 9900, 11940, 12480, 12640, 12712, 12728]
    '''
    threshold = [0]
    thre = 0
    b = 0
    
    for i,size in enumerate(feature_map_size):
    
        thre += size[0]*size[1]*box_num[i]
        threshold.append(thre)
    
    index1 = -1
    
    for i in threshold:
    
        if index>=i:
            index1+=1
    
    coor1 = (index - threshold[index1])//box_num[index1]
    coor2 = (index - threshold[index1])%box_num[index1]
    coor3 = coor1//feature_map_size[index1][1]
    coor4 = coor1%feature_map_size[index1][1]
    
    for i in range(index1):
        b += box_num[i]
    
    [w,h] = box_size[b:b+box_num[index1]][coor2]
    
    x = (float(coor4)+0.5)*480/feature_map_size[index1][1]
    y = (float(coor3)+0.5)*270/feature_map_size[index1][0]
    
    return [x,y,w,h]
    
def Box(offset, pre_box):
    
    '''
    根据预选框和offset得到预测框
    '''
    
    offset_x = offset[0]
    offset_y = offset[1]
    offset_w = offset[2] 
    offset_h = offset[3] 
    
    pre_center_x = pre_box[2]*offset_x + pre_box[0]
    pre_center_y = pre_box[3]*offset_y + pre_box[1]
    pre_center_w = pre_box[2]*(math.e**offset_w)
    pre_center_h = pre_box[3]*(math.e**offset_h)
    
    box = [pre_center_x-pre_center_w/2,pre_center_y-pre_center_h/2,pre_center_w,pre_center_h]
    
    return box

def IOU(box1, box2):
    
    """
    可以广播
    输入左上坐标和wh
    Returns the IoU of two bounding boxes 
    得到bbox的坐标
    """
    
    # Get the coordinates of bounding boxes
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    box1 = box1.to(device)
    box2 = box2.to(device)
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2]+box1[:, 0], box1[:, 3]+box1[:, 1]
    
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2]+box2[:, 0], box2[:, 3]+box2[:, 1]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1).to(device)
    inter_rect_y1 = torch.max(b1_y1, b2_y1).to(device)
    inter_rect_x2 = torch.min(b1_x2, b2_x2).to(device)
    inter_rect_y2 = torch.min(b1_y2, b2_y2).to(device)

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
    
def NMS(class_sum, loc_sum, threshold1, threshold2):
    
    '''
    class_sum shape[batch,12728,2]
    loc_sum shape[batch,12728,4]
    threshold∈(0,1)
    '''
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #挑选置信度高于threshold1的框作为集合
    #若为多分类logsoftmax = nn.LogSoftmax(dim=class+1),values1, class = torch.max(class_sum, dim = 2)
    #values2, indices = torch.sort(values, dim = 1, descending=True)
    
    logsoftmax = nn.LogSoftmax(dim=2)
    class_sum = logsoftmax(class_sum)
    class_sum = class_sum[:,:,1]
    values, indices = torch.sort(class_sum, dim = 1, descending=True)
    
    index = []
    for k,value in enumerate(values):
    
        for i,v in enumerate(value):
    
            if v <= math.log(threshold1):
                indice = indices[k,0:i+1]
                break
    
            if i==(len(value)-1):
                indice = indices[k]
    
        index.append(indice)
    
    boxess = []
    for k,ii in enumerate(index):
        boxes = []
    
        for i in ii:
            box = pre_box_select(i, feature_map_size, box_num, box_size)
            box = Box(loc_sum[k][i], box)
            boxes.append(box)
    
        boxess.append(boxes)
        
    #再利用NMS算法对boxess进行筛选
    pre_boxess = []
    for boxes in boxess:
    
        pre_boxes = [boxes[0]]
        del boxes[0]
        while(len(boxes)!=0):
    
            iou = IOU(torch.Tensor([pre_boxes[-1]]).to(device), torch.Tensor(boxes).to(device))
            values, indices = torch.sort(iou, dim = 0, descending=True)
            l = []
            for i,value in enumerate(values):
    
                if value>threshold2:
                    l.append(indices[i])
    
            l.sort(reverse=True)
            for i in l:
                del boxes[i]
    
            if len(boxes)!=0:
                pre_boxes.append(boxes[0])
                del boxes[0]
    
        pre_boxess.append(pre_boxes)
    
    return pre_boxess
    
    
    
def accuracy(class_sum, loc_sum, true_box_coors):
    
    pre_boxess = NMS(class_sum, loc_sum, threshold1=0.95, threshold2=0.45)
    acc = 0.0
    
    for i,pre_boxes in enumerate(pre_boxess):
        indice = []
        values = 0.0
    
        for pre_box in pre_boxes:
    
            iou = IOU(torch.Tensor([pre_box]), torch.Tensor(true_box_coors[i]))
            value,index = torch.max(iou,dim=0)
            '''
            可以用下面计算recall
            if index not in indice:
                values += value
            indice.append(index)
            '''
            values += value
    
        acc += values/len(pre_boxes)
    
    return float(acc/len(true_box_coors))
    
    
    
def center(feature_map_size, box_num, box_size):
    
    '''
    得到box_coordinate为12728个预选框的坐标list
    list长度为12728，元素为[x,y,w,h],x,y为中心坐标
    顺序为feature map由大到小
    每层feature map中的顺序为像素点由左到右，再由上到下
    每个像素点中的顺序为预选框大小由小到大
    '''
    
    box_coordinate = []
    index = 0
    for k, feature_map in enumerate(feature_map_size):
        box_size_son = box_size[index:index+box_num[k]]
    
        for i in range(feature_map[0]):
    
            for j in range(feature_map[1]):
    
                h_coor = (i+0.5)/feature_map[0]*270
                w_coor = (j+0.5)/feature_map[1]*480
    
                for num in range(box_num[k]):
                    box_coordinate.append([w_coor, h_coor]+box_size_son[num])
    
        index += box_num[k]
    
    return box_coordinate

def extract_img(path):
    '''
    从视频中抽取test集，也可用于无监督学习
    '''
    filename = os.listdir(path)
    k = 0
    
    for file in filename:
        
        path2 = os.path.join(path, file)
        filename2 = os.listdir(path2)
        
        for i in filename2:
            
            if i.endswith('.mp4') and (not i.endswith('merge.mp4')):
                
                capture = cv.VideoCapture(os.path.join(path2, i))
                ret, frame = capture.read()
                cv.imwrite('./test/'+str(k)+'.jpg',frame)
                k += 1

                
def accuracy_uda(logits, labels):
    
    pre_labels = logits.argmax(1)
    acc = float((pre_labels==labels).sum())/len(labels)
    
    return acc
                
def save_model(model, optimizer, scheduler, global_step, m = 1):
    
    filename = os.listdir('./model'+str(m))[0]
    shutil.rmtree(os.path.join('./model'+str(m), filename))
    
    file = os.path.join('./model'+str(m), str(global_step))
    os.mkdir(file)
    
    torch.save(model.state_dict(),os.path.join(file,'model.pt'))
    
    torch.save(optimizer.state_dict(),os.path.join(file,'optimizer.pt'))
    
    torch.save(scheduler.state_dict(),os.path.join(file,'scheduler.pt'))
    
    print('模型保存成功！')

def load_model(model = None, optimizer = None, scheduler = None, m = 1):
    
    filename = os.listdir('./model'+str(m))[0]
    
    model.load_state_dict(torch.load(os.path.join('./model'+str(m),filename)+'/model.pt', map_location='cpu'))
    
    if optimizer != None:
    
        optimizer.load_state_dict(torch.load(os.path.join('./model'+str(m),filename)+'/optimizer.pt'))
    
    if scheduler != None:
    
        scheduler.load_state_dict(torch.load(os.path.join('./model'+str(m),filename)+'/scheduler.pt'))
    
    print('模型参数加载成功！')