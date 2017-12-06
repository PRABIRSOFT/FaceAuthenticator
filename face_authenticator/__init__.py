#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-23
# @Author  : Prabir Ghosh
# @Version : 0.1
#
# FaceAuthenticator

import os
import sys
import cv2
import pandas as pd
import numpy as np
import face_authenticator as __fa__

__installed_path__ = os.path.dirname(__fa__.__file__)

# Class FaceAuthenticator
class FaceAuthenticator():
    __face_cascade__     = cv2.CascadeClassifier(__installed_path__ + "/res/lbpcascade_frontalface.xml")
    __X__                = None
    __y__                = None
    __x__                = None
    __data_description__ = None
    __rootpath__         = None
    
    # function To Initialize Basic FaceAuthenticator
    # params 'datapath' = root path to store all the data    
    def __init__(self, datapath=None):
        if(datapath == None):
            datapath = "data/"

        try:
            os.stat(datapath)
        except:
            os.mkdir(datapath)

        if(not datapath.endswith("/")):
            self.__rootpath__ = datapath + "/"
        else:
            self.__rootpath__ = datapath
        
        try:
            os.stat(self.__rootpath__ + "faces/")
            self.__prepare_training_data__()
        except:
            os.mkdir(self.__rootpath__ + "faces/")
            
    # function To capture a face using webcam
    # params 'name' = Person Name whose record are going to save in memory
    # params 'sample_size' = size of the sample for that person        
    def capture_face(self, name = None, sample_size = 5):
        if(name == None):
            print("No Name Found")
            return
        for i in range(sample_size):
            self.__captur_sample__(name, i)        
        try:
            df = pd.read_csv(self.__rootpath__ + "sample.csv")
            df = df.append({"Name":name,"Images":sample_size}, 1)
        except:
            df = pd.DataFrame([{"Name":name,"Images":sample_size}])        
        df.to_csv(self.__rootpath__ + "sample.csv", index=False)
        self.__prepare_training_data__()
        
        
    # function To store sample faces
    # params 'train_name' = Person Name whose record are going to save in memory
    # params 'index' = index of the sample for that person        
    def __captur_sample__(self, train_name, index):
        train_path = self.__rootpath__ + "faces/" + train_name + "/"
        try:
            os.stat(train_path)
        except:
            os.mkdir(train_path)
        cap = cv2.VideoCapture(0)
        cap.set(3, 1024)
        cap.set(4, 768)
        while(True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.__face_cascade__.detectMultiScale(gray, 3, 5)
            if(len(faces) > 0):
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop_img = frame[y: y + h, x: x + w]
                crop_img = cv2.fastNlMeansDenoisingColored(crop_img,None,10,10,7,21)
                cv2.imwrite(train_path + str(index) + ".jpg", crop_img)
                break
        cap.release()
        cv2.destroyAllWindows()
        

    # function To capture and compare a person with a sample persons
    # return: Name of the Person whose face is in the image
    def authenticate(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        while(True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.__face_cascade__.detectMultiScale(gray, 2, 5)
            if(len(faces) > 0):
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop_img = frame[y: y + h, x: x + w]
                crop_img = cv2.fastNlMeansDenoisingColored(crop_img,None,10,10,7,21)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                break
        cap.release()
        self.__x__ = crop_img
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(self.__X__, np.array(self.__y__))
        df = pd.read_csv(self.__rootpath__ + "sample.csv")
        label = recognizer.predict_label(crop_img)
        return df.iloc[label]["Name"]
    

    # function To capture and compare a person with a sample persons
    # sets trainging data for face authentication    
    def __prepare_training_data__(self):
        dirs = os.listdir(self.__rootpath__ + "faces/")
        faces = []
        labels = []
        df = pd.read_csv(self.__rootpath__ + "sample.csv")
        for dir_name in dirs:
            image_count = df[df['Name'] == dir_name]['Images'].values[0]
            label = df[df['Name'] == dir_name].index[0]
            for i in range(image_count):
                img_path = self.__rootpath__ + "faces/" + dir_name + "/" + str(i) + ".jpg"
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces.append(gray)
                labels.append(label)
        self.__X__ = faces
        self.__y__ = labels