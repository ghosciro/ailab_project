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

video_name="video8.mp4"

video=cv2.VideoCapture(video_name) #aprire video

fps=int(video.get(cv2.CAP_PROP_FPS)) #sapere fps del video
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


video=cv2.VideoCapture(video_name)
succ,frame=read(video)
video_l=[]
i=0

print("cropping,wait for me")

DISTANCE=10
MAX_FRAME=300

#prendere MAX_FRAME frame del video ogni DISTANCE frame
while succ and len(video_l)<MAX_FRAME:
    if i==DISTANCE:
            video_l.append(frame[Y:y])
            i=0
            print(f"cropping frame n° {len(video_l)}")
    succ,frame=read(video)
    i+=1

#trasformo lista in np.array
video_l=np.array(video_l)

#rimuovo la mano
dst = video_l[0]
for i in range(1,len(video_l)):
        alpha = 1.0/(i)#1/2,1/3,1/4
        beta = 1.0 - alpha#1/2,2/3,3/4
        dst = cv2.addWeighted(video_l[i], alpha, dst, beta, 0.0)#somma pesata 

#dst=make_things_better(dst)

cv2.imshow("",dst)
cv2.imwrite(video_name+".jpg",dst)

cv2.waitKey(0)
