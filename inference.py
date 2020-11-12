#!/usr/bin/env python
# coding: utf-8

from model import Model
import torch
import math
import numpy as np
import cv2 as cv
from tools import *
import random
import matplotlib.pyplot as plt 
import shutil
import os

    
class Inference():
    
    def __init__(self):
    
        self.model = Model(Class = 2, box_num = box_num)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        load_model(self.model,m=1)
        self.model.eval()
        self.color = [i*0.01 for i in range(30,100)]
        
    def load_img(self, img_path):
    
        images = []
        for i in img_path:
    
            img = cv.imread(i)
            img = cv.resize(img,(480,270))
            #img = (img-img.mean())/max(img.std(),1/360)
            images.append(img)
    
        images = np.array(images)
        images = images/255.
        images = torch.Tensor(images).permute(0,3,1,2).to(self.device)
    
        return images
    
    def get_boxes(self, class_sum, loc_sum):
    
        pre_boxes = NMS(class_sum, loc_sum, threshold1=0.95, threshold2=0.3)#xywh左上坐标
    
        return pre_boxes
        
    def show_img(self, img_path):
    
        images = self.load_img(img_path).to(self.device)
        class_sum, loc_sum = self.model(images)
        boxes = self.get_boxes(class_sum, loc_sum)
        plt.figure(figsize=(80,60))
        size = int(math.sqrt(len(img_path))+0.9999999999)
    
        for i,path in enumerate(img_path):
    
            img = cv.imread(path)
            shape = img.shape
    
            for box in boxes[i]:
    
                box[0] = torch.round(box[0]*shape[1]/480)
                box[1] = torch.round(box[1]*shape[0]/270)
                box[2] = torch.round(shape[1]/480*box[2])
                box[3] = torch.round(shape[0]/270*box[3])
            
            plt.subplot(size,size,i+1)
            plt.title(str(i+1))
            plt.imshow(img)
    
            for j,box in enumerate(boxes[i]):
    
                plt.gca().add_patch(
                    plt.Rectangle((box[0],box[1]),box[2],box[3],
                                  edgecolor=[random.choice(self.color),random.choice(self.color),random.choice(self.color)],
                                  fill=False, linewidth=10))
    
            plt.axis('off')
    
        plt.show()
        
    
    def save_face_from_img_path(self, img_path):
    
        images = self.load_img(img_path).to(self.device)
        class_sum, loc_sum = self.model(images)
        boxes = self.get_boxes(class_sum, loc_sum)
    
        for i,path in enumerate(img_path):
    
            img = cv.imread(path)
            shape = img.shape
    
            x = int(torch.round(boxes[i][0][0]*shape[1]/480)-1)
            y = int(torch.round(boxes[i][0][1]*shape[0]/270)-1)
            w = int(torch.round(shape[1]/480*boxes[i][0][2])+1)
            h = int(torch.round(shape[0]/270*boxes[i][0][3])+1)
    
            cv.imwrite(img_path[i].split('.jpg')[0]+'_face.jpg',img[y:y+h,x:x+w])
            
    
    def save_face_from_video_path(self, video_path, save_frq = 60, batch = 64):
    
        '''
        video_path输入一个病人的视频地址list
        [./depression_data/chenliangjie/100_╥╜╔·╬╩_401.530-406.900.mp4]
        '''
    
        frames = []
        images = []
        names = []
        for path in video_path:
    
            capture = cv.VideoCapture(path)
    
            for j in range(100000):
    
                ret, frame = capture.read()
    
                if ret==False:
                    break
    
                if j%(save_frq-1) == 0:
    
                    shape = frame.shape
                    frames.append(frame)
                    frame = cv.resize(frame,(480,270))
                    images.append(frame)
                    names.append(path)
    
            if j < (save_frq) and j > save_frq//2:
    
                capture = cv.VideoCapture(path)
    
                for k in range(j-1):
    
                    ret, frame = capture.read()
    
                frames.append(frame)
                frame = cv.resize(frame,(480,270))
                images.append(frame)
                names.append(path)
    
        images = np.array(images)
        images = torch.Tensor(images).permute(0,3,1,2)
        
        '''
        分段输入model
        '''
        b = len(frames)//batch
        c = len(frames)%batch
        a = []
    
        for i in range(b):
            a.append([i*batch,(i+1)*batch])
    
        if 'i' in dir():
            if c != 0:
                a.append([(i+1)*batch,(i+1)*batch+c])
    
        elif c!=0:
            a.append([0,c])
        
        
        for index,l in enumerate(a):
    
            image = images[l[0]:l[1]].to(self.device)
            class_sum, loc_sum = self.model(image/255.)
            boxes = self.get_boxes(class_sum, loc_sum)
    
            for i,img in enumerate(frames[l[0]:l[1]]):
    
                x = int(torch.round(boxes[i][0][0]*shape[1]/480)-1)
                y = int(torch.round(boxes[i][0][1]*shape[0]/270)-1)
                w = int(torch.round(shape[1]/480*boxes[i][0][2])+1)
                h = int(torch.round(shape[0]/270*boxes[i][0][3])+1)
                
                
                if img[y:y+h,x:x+w].any() and w*h>=36000:
                    
                    cv.imwrite('../FaceEmotionExtraction/dataset/train/' +
                               names[l[0]:l[1]][i].split('/')[2] + '/' +
                               names[l[0]:l[1]][i].split('/')[3].split('_')[0]+'_'+ 
                               names[l[0]:l[1]][i].split('/')[3].split('_')[1] +
                               '_'+str(i+index*batch)+'.jpg',img[y:y+h,x:x+w])
                    
                    '''
                    [./depression_data/chenliangjie/100_╥╜╔·╬╩_401.530-406.900.mp4]
                    '''
                
                elif i==(len(frames[l[0]:l[1]])-1):

                    x = abs(x)
                    y = abs(y)
                    w = abs(w)
                    h = abs(h)

                    cv.imwrite('../FaceEmotionExtraction/dataset/train/' +
                               names[l[0]:l[1]][i].split('/')[2] + '/' +
                               names[l[0]:l[1]][i].split('/')[3].split('_')[0]+'_' + 
                               names[l[0]:l[1]][i].split('/')[3].split('_')[1] +
                               '_'+str(i+index*batch)+'.jpg',img[y:y+h+30,x:x+w+30])
                
                
                elif names[l[0]:l[1]][i].split('/')[3].split('_')[1]!=names[l[0]:l[1]][i+1].split('/')[3].split('_')[1]:
                    
                    x = abs(x)
                    y = abs(y)
                    w = abs(w)
                    h = abs(h)

                    cv.imwrite('../FaceEmotionExtraction/dataset/train/' + 
                               names[l[0]:l[1]][i].split('/')[2] + '/' +
                               names[l[0]:l[1]][i].split('/')[3].split('_')[0]+'_'+
                               names[l[0]:l[1]][i].split('/')[3].split('_')[1] +
                               '_'+str(i+index*batch)+'.jpg',img[y:y+h+30,x:x+w+30])

                    
    def Key(self, x):
    
        return int(x.split('_')[0])
    
    def get_data(self, save_frq = 60, batch = 64):
    
        shutil.rmtree('../FaceEmotionExtraction/dataset/train')
        os.mkdir('../FaceEmotionExtraction/dataset/train')
        filename = os.listdir('./depression_data')
    
        for file in filename:
    
            videos = os.listdir('./depression_data/'+file)
            videos.sort(key=self.Key)
            video_path = []
    
            for video in videos:
    
                if video.endswith('.mp4') and not video.endswith('merge.mp4'):
                    video_path.append('./depression_data/'+file+'/'+video)
    
            os.mkdir('../FaceEmotionExtraction/dataset/train/'+file)
            self.save_face_from_video_path(video_path, save_frq = save_frq, batch = batch)
        
