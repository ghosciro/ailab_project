

import cv2
import numpy as np 
import pandas as pd 
import random
img=cv2.imread("i.jpg",0)
df  = pd.read_excel("coordinates.xlsx")
y0=df["bbox-0"]
x0=df["bbox-1"]
y1=df["bbox-2"]
x1=df["bbox-3"]
height_step=50
for i in range(len(x0)):
    rgb = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    
    cv2.line(img,(x0[i],0),(x0[i],img.shape[0]),rgb,1)
    cv2.line(img,(x1[i],0),(x1[i],img.shape[0]),rgb,1)
lines=cv2.HoughLinesP(img,  1, 1.34/180, 50, 50, 10 )
#for element in lines:
        #cv2.line( img, (element[0][0], element[0][1]), (element[0][2], element[0][3]), (0,0,255), 3)


cv2.imwrite("outputfile.jpg",img)
for indexes in range(img.shape[0],height_step,-height_step):
    element=img[indexes-height_step:indexes]
    slicing=12
    cv2.imshow(" ",element)
    key=cv2.waitKey(0)
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("s"):
        print("saved")
        cv2.imswrite("saved.jpg",element)
    if key == ord("k"):
        break