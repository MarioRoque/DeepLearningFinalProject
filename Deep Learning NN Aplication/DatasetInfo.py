# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:02:01 2020

@authors: Greg Espinosa 
          Mario Roque
         
"""

import numpy as np 
import pandas as pd
import os
import xml.etree.ElementTree as et
import cv2
import matplotlib.pyplot as plt

debug = False

def plot(idx,data, labels):
    classes = ["without_mask","with_mask"] #defining classes
    plt.title(classes[labels[idx]])
    plt.imshow(data[idx])
    
    
def get_main_person_info():
    dic = {"image": [],"Dimensions": [], "main_person": []}
    
    for file in os.listdir("../dataset/annotations"):
        row = []
        xmlfile = et.parse("../dataset/annotations/"+file)  ##Parsing the xml file with the annotations on each image
        root = xmlfile.getroot()
        img = root[1].text  ##Obtaining image name
        row.append(img)
        height,width = root[2][0].text,root[2][1].text ##Obtaining image size
        row.append([height,width]) 
    
        #Obtaining main_person_information about object 1    
        temp = []
        temp.append(root[4][0].text) #Obtaining label of object 1 column
        for point in root[4][5]:
            temp.append(point.text) #Obtaining the coordinates of the face on Obj1
        row.append(temp)

        #Cleanning the third clase to work with a binary problem
        if(not "mask_weared_incorrect" in root[4][0].text): ##Not appending the third class
            for i,each in enumerate(dic):
                dic[each].append(row[i])
    df = pd.DataFrame(dic)
    return df  ##Return the data for the pictures on class No mask and Mask.


def get_images(df):
    image_directories = []
    for i in df["image"] : ##Obtaining the path for images on the cleaned dataframe.
        image_directories.append("../dataset/images/" + i)
        
    classes = ["without_mask","with_mask"] #defining classes
    labels = []
    data = []
    
    print("Loading dataset....") 
    for idx,image in enumerate(image_directories):
            
        img  = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ##Reads image and transforms from BGR to RGB
    
        X,Y = df["Dimensions"][idx] #Obtains the the image to the desired size
        cv2.resize(img,(int(X),int(Y))) #Resizes the image to X,Y
        
        main_person_info = df["main_person"][idx]   #Obtains the face coordinates for object 1 and extracts it
        
        main_person_info[0] = main_person_info[0].replace(str(main_person_info[0]), str(classes.index(main_person_info[0])))
        
        main_person_info=[int(each) for each in main_person_info] #Parses all the string contents on main_person_info from Object 1 to integer values
        label = main_person_info[0]
        
        face = img[main_person_info[2]:main_person_info[4],main_person_info[1]:main_person_info[3]] ##Crops the face on the coordinates given by the Object 1
        
        if((main_person_info[3]-main_person_info[1])>40 and (main_person_info[4]-main_person_info[2])>40): #Makes sure that the object has a recognizable size
            try:
                face = cv2.resize(face, (64,64)) ##Resizes the image
                face = np.asarray(face)  #Makes sure the image is in a numpy array
                data.append(face)   #Appends the face
                labels.append(label) #Appends the labels

            except:
                print("There was an unhandled exeption")
                pass
    print("Dataset loaded!")
    return data, labels

if(debug):
    df = get_main_person_info()
    data, labels =get_images(df)


