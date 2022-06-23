

import cv2
import numpy as np 
import pandas as pd 
import random
import skimage.io 
from skimage import measure
from skimage.color import label2rgb, rgb2gray
from rotellini.makespartito import rotellini

def make_things_better(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5,5), 0)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,1.2
    )
    image = cv2.GaussianBlur(image, (5, 5), 0)
    ret, image = cv2.threshold(image, 200, 255, 0)
    return image



img=cv2.imread("lower_image.jpg")
daf  = pd.read_excel("coordinates.xlsx")
x0=daf["bbox-1"]
names=daf["name"]
x1=daf["bbox-3"]
height_step=25

'''
notes=[]
key=-1
for indexes in range(img.shape[0],height_step,-height_step):
    element=img[indexes-height_step:indexes]
    gray = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)
    test = make_things_better(element)
    label_image = measure.label(test,connectivity=2)
    label_image_rgb = label2rgb(label_image, image=test, bg_label=0)
    props = measure.regionprops_table(label_image, element, properties=["label", "area", "mean_intensity", "bbox"],)
    df = pd.DataFrame(props)
    df = df[df["area"] < 2000 ]
    df = df[df["area"] > 200 ]
    note=[]
    for element in zip(df[["bbox-1", "bbox-3"]].values)  :
        for rectangle in element:
            media= (rectangle[0]+rectangle[1]) //2
            print(media)
            for i in range(len(x0)):
                if media>=x0[i] and media<=x1[i]:
                    print(names[i])
                    note.append(names[i])
    cv2.imshow("elementk.jpg", test)
    if(note != []):   notes.append(note)
    print(notes)
    cv2.imshow("test",label_image_rgb)
    key=cv2.waitKey(0)
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("s"):
        print("saved")
        skimage.io.imsave("save.jpg",label_image_rgb)
        cv2.imwrite("elementk.jpg", element)
    if key == ord("k"):
        break
rotellini(notes,"")
'''


element=img[0:img.shape[0]]
print(element.shape[0])
gray = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)
test = cv2.Canny(element,230,250)#make_things_better(element)
label_image = measure.label(test,connectivity=2)
label_image_rgb = label2rgb(label_image, image=test, bg_label=0)

props = measure.regionprops_table(
        label_image, gray, properties=[ "area","bbox"],
    )
df = pd.DataFrame(props)
df = df[df["area"] > 200 ]
df=df.iloc[::-1]
print(df)
skimage.io.imsave("save.jpg",label_image_rgb)
quarter=100
notes=[]
for i in range(element.shape[0],0,-quarter):
    note=[]
    for element in df.values:
        if element[1]<i and element[3]>i:
            media=(element[2]+element[4])//2
            print(i,"element:",element)
            for y in range(len(x0)):
                if media>=x0[y] and media<=x1[y]:
                    note.append(names[y])
    if note!=[]:
        notes.append(note)

print(len(notes))
rotellini(notes)
