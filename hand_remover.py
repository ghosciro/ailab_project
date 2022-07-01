import cv2
import numpy as np


#DEBUG=True 
DEBUG=False 


##filtri per immagine binario
def make_things_better(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.GaussianBlur(image,(3,5),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
    image = cv2.GaussianBlur(image,(5,5),0)
    ret, image = cv2.threshold(image, 200, 255, 0)
    return image

#wrapper del video
def read(video):
    success,frame=video.read()
    #frame=make_things_better(frame)
    return success,frame

video_name="video1.mp4"

video=cv2.VideoCapture(video_name) #aprire video

fps=int(video.get(cv2.CAP_PROP_FPS)) #sapere fps del video
print(fps)
succ,frame=read(video)
Y=350
y=275
key=0
print(video_name)
while succ and key!=ord("k") and not DEBUG:
    frame=frame[Y:y]
    cv2.imshow("",frame)
    key=cv2.waitKey((1000//fps))
    if key==ord("c"):
        #tagliare l'immagine
        print("top border to crop at :")
        Y+=int(input())
        print("bottom border to crop at:")
        y-=int(input())
        print(Y,y-frame.shape[0])
    succ,frame=read(video)
    if key==ord("k"):
        break
cv2.destroyAllWindows()



#dst=make_things_better(dst)

cv2.imshow("",dst)
cv2.imwrite(video_name+".jpg",dst)

cv2.waitKey(0)
