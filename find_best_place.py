import cv2
from cv2 import meanShift
from cv2 import imshow
import numpy as np

Y=425
video_source="video.mp4"
def make_things_better(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.GaussianBlur(image,(11,11),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,0.5)
    image = cv2.GaussianBlur(image,(3,3),0)
    ret, image = cv2.threshold(image, 150, 255, 0)
    return image

def cut_and_return(video_l,n_cut):
    list_of_pieces=[]
    max=len(video_l)-1
    for screen in video_l:
        pieces=[]
        for y in range(0,screen.shape[0]-screen.shape[0]//n_cut,screen.shape[0]//n_cut):
            pieces.append(screen[y:y+screen.shape[0]//n_cut])
        list_of_pieces.append(pieces)
    foo=[[] for  _ in range(n_cut-1)]
    for i in range(n_cut-1):
        for y in range(max):
            piece=list_of_pieces[y][i]
            foo[i].append(piece)
    return foo

def cut_frames(video,max,skip=15):
    video_l=[]
    i=0
    succ=True
    while succ and len(video_l)<=max:
        if i==skip:
            print(f"taking frame n° {len(video_l)*skip}")
            frame=cv2.cvtColor(frame[:Y,:],cv2.COLOR_BGR2GRAY)
            video_l.append(frame)
            i=0
        i+=1
        succ,frame=video.read()
    return video_l



video=cv2.VideoCapture(video_source)
skip=15
max=50

video_l=cut_frames(video,max,skip)
n_cut=video_l[0].shape[0] //20
video_l=cut_and_return(video_l,n_cut)

means=[np.var(x) for x in  [[ img.mean() for img in ps ] for ps in video_l]]
minimu=min(means)
pos=means.index(minimu)
print(minimu,pos,means)
#show video

video=cv2.VideoCapture(video_source)
key=-1 
while True :
    succ,frame = video.read()
    frame=make_things_better(frame)
    cut=(pos)*frame[:Y,:].shape[0]//(n_cut)
    frame[cut:cut+frame[:Y,:].shape[0]//n_cut]=cv2.add(frame[cut:cut+frame[:Y,:].shape[0]//n_cut],-50)
    cv2.imshow("",frame)
    if key==ord("k"):
        break
    key=cv2.waitKey(33//4)
    if key==ord(" "):
        key = cv2.waitKey(0)
        print(chr(key))

#[[1,2,3],[1,2,3,],[1,2,3],[1,2,3]]
#[[1,1,1,1],[2,2,2,2],[3,3,3,3]]