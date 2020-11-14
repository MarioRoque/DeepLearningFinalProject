# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:02:01 2020

@author: Eleanor
"""

import numpy as np 
import pandas as pd
import os
import xml.etree.ElementTree as et
import re
import glob
import cv2
import matplotlib as plt


def get_labels():
    dic = {"image": [],"Dimensions": []}
    simple_dic = {"label": []}
    for i in range(1,116):
    	dic[f'Object {i}']=[]
    print("Generating data in CSV format....")
    
    for file in os.listdir("../dataset/annotations"):
        row = []
        xml = et.parse("../dataset/annotations/"+file) 
        root = xml.getroot()
        img = root[1].text
        row.append(img)
        h,w = root[2][0].text,root[2][1].text
        row.append([h,w])
    
        for i in range(4,len(root)):
            temp = []
            temp.append(root[i][0].text)
            for point in root[i][5]:
                temp.append(point.text)
            row.append(temp)
        for i in range(len(row),119):
            row.append(0)
        #print(root[4][0].text)
        if(not "mask_weared_incorrect" in root[4][0].text):
            #simple_dic["label"].append(root[4][0].text)
            for i,each in enumerate(dic):
                dic[each].append(row[i])
    df = pd.DataFrame(dic)
    return df, simple_dic


def get_images(df, simple_dic):
    image_directories = []
    for i in df["image"] :
        image_directories.append("../dataset/images/" + i)
        
    classes = ["without_mask","with_mask"]
    labels = []
    data = []
    
    print("Extracting each data into respective label folders....")
    for idx,image in enumerate(image_directories):
        print(image)
        if(df["Object 1"][idx][0]=="mask_weared_incorrect"):
                print("hello1")
                break
            
        img  = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if(idx % 100==0):
            plt.pyplot.imshow(img)
            
            plt.pyplot.show()
        #scale to dimension
        X,Y = df["Dimensions"][idx]
        cv2.resize(img,(int(X),int(Y)))
        
        
        
        #find the face in each object
        info = df["Object 1"][idx]
        if info!=0:
            label = info[0]
            if(label=="mask_weared_incorrect"):
                print("hello2")
                break
            info[0] = info[0].replace(str(label), str(classes.index(label)))
            info=[int(each) for each in info]
            face = img[info[2]:info[4],info[1]:info[3]]
            if((info[3]-info[1])>40 and (info[4]-info[2])>40):
                try:
                    face = cv2.resize(face, (224, 224))
                    face = np.asarray(face)
                   # face = preprocess_input(face)
                    data.append(face)
                    labels.append(label)

                except:
                    print("excepcion")
                    pass
    return data, labels

df, simple_dic = get_labels()
data, labels =get_images(df, simple_dic)



