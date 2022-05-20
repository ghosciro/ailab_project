from multiprocessing.connection import wait
from xml.dom.minidom import Element
import cv2
from cv2 import line
from cv2 import destroyAllWindows
import numpy as np  
import math

from torch import nonzero
def make_things_better(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
    image=cv2.GaussianBlur(image,(5,5),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
    image = cv2.GaussianBlur(image,(3,3),0)
    ret, image = cv2.threshold(image, 125, 255, 0)
    return image




img=cv2.imread("i.jpg",cv2.IMREAD_GRAYSCALE)
notes=np.array(["a","b","c","d","e","a1","b1","c1","d1","e1","a2","b2","c2","d2","e2","a3","b3","c3","d3","e3"])
k=-1
l=0
list_of_notes=[]
height_step=50
for indexes in range(img.shape[0],height_step,-height_step):
    l+=1
    element=img[indexes-height_step:indexes]
    slicing=20
    element_list=[element[:,i*element.shape[1]//slicing:(i+1)*element.shape[1]//slicing] for i in range(0,slicing)]
    notes_s=[]
    for i,e in enumerate( element_list):  
        cv2.imshow(f"{i}",e)
        if np.count_nonzero(e)>50:
            notes_s.append(notes[i])
        if k==ord("e") or k==ord("k"):
            break
    print(notes_s)
    if notes_s : list_of_notes.append(notes_s)
    if k==ord("k"):
        break
    print(len(notes_s))
print(len(list_of_notes))
cv2.destroyAllWindows()