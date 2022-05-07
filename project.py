import cv2
import numpy as np

#DEBUG=True 
DEBUG=False 

def make_things_better(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.GaussianBlur(image,(3,5),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
    image = cv2.GaussianBlur(image,(5,5),0)
    ret, image = cv2.threshold(image, 200, 255, 0)
    return image
def read(video):
    success,frame=video.read()
    #frame=make_things_better(frame)
    return success,frame

video=cv2.VideoCapture("video1.mp4")
fps=int(video.get(cv2.CAP_PROP_FPS))
print(fps)
succ,frame=read(video)
y=frame.shape[0]-100
Y=475
X=frame.shape[1]
frame.shape
print(frame.shape)
key=0
while succ and key!=ord("k") and not DEBUG:
    cv2.imshow("",frame)
    key=cv2.waitKey((1000//fps))
    if key==ord("c"):
        print("Y to crop at :")
        Y+=int(input())
        print("y to crop at:")
        y-=int(input())
        print("X to crop at :")
        X-=int(input())
        print(X,Y,y-frame.shape[0])
    succ,frame=read(video)
    frame=frame[Y:y,:X]
cv2.destroyAllWindows()
video=cv2.VideoCapture("video1.mp4")
succ,frame=read(video)
video_l=[]
print("cropping,wait for me")
i=0
distance=30
while succ and len(video_l)<30:
    if i==distance:
            video_l.append(frame[Y:y,:X])
            i=0
            print(f"cropping frame n° {len(video_l)}")
    succ,frame=read(video)
    i+=1
video_l=np.array(video_l)
dst = video_l[0]
for i in range(1,len(video_l)):
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        dst = cv2.addWeighted(video_l[i], alpha, dst, beta, 0.0)
dst=make_things_better(dst)
cv2.imshow("",dst)
cv2.imwrite("keyboard.jpg",dst)

cv2.waitKey(0)
