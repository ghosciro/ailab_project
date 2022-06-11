

import cv2
import numpy as np 
<<<<<<< HEAD
import mingus.extra.lilypond as LilyPond
import mingus.containers as container
img=cv2.imread("i.jpg",cv2.IMREAD_GRAYSCALE)
notes=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
height_step=20
N = container.Track()
prova=[]
for indexes in range(img.shape[0],height_step,-height_step):
    element=img[indexes-height_step:indexes]
    slicing=12
    element_list=[element[:,i*element.shape[1]//slicing:(i+1)*element.shape[1]//slicing] for i in range(0,slicing)]
    for i,e in enumerate( element_list):  
        if np.count_nonzero(e)>50:
            N + notes[i]
            prova.append(notes[i])
bar = LilyPond.from_Track(N) 
print(prova)
LilyPond.to_png(bar, "my_first_bar")
=======
import pandas as pd 
import random
img=cv2.imread("i.jpg")
df  = pd.read_excel("output.xlsx")
for element in df.values:
    print(element)
y0=df["bbox-0"]
x0=df["bbox-1"]
y1=df["bbox-2"]
x1=df["bbox-3"]
height_step=50
for i in range(len(x0)):
    rgb = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    cv2.line(img,(x0[i],y0[i]),(x0[i],img.shape[0]),rgb,2)
    cv2.line(img,(x1[i],y1[i]),(x1[i],img.shape[0]),rgb,2)
cv2.imwrite("outputfile.jpg",img)
for indexes in range(img.shape[0],height_step,-height_step):
    element=img[indexes-height_step:indexes]
    slicing=12

    #element_list=[element[:,i*element.shape[1]//slicing:(i+1)*element.shape[1]//slicing] for i in range(0,slicing)]
    cv2.imshow(" ",element)
    key=cv2.waitKey(200)
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("k"):
        break
>>>>>>> 235cdfa4aae81dae6cc360363e80337db401c849
