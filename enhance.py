# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:16:04 2020

@author: nmahe
"""


#import libraries 
import cv2 
import numpy as np 
import skimage
from PIL import Image, ImageEnhance 

#histogram equalization to improve contrast 
def improve_contrast_image_using_clahe(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(7, 7))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

####CROP, CONTRAST, and ENHANCE
images = ['image1.jpg'] #input appropriate image name(s) 
for name in images:   
    
    img = cv2.imread(name)
    
    ###CONTRAST
    improved = improve_contrast_image_using_clahe(img)
    
    ###CROP
    gray = cv2.cvtColor(improved, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # Find contour and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = improved[y:y+h, x:x+w]
        break
    cv2.imwrite(name[:-4]+'_enhanced.jpg',ROI)