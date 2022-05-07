import cv2
import numpy as np
from multiprocessing import Process
Y=300
video_source="video1.mp4"

def make_things_better(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
    image=cv2.GaussianBlur(image,(5,5),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    image = cv2.GaussianBlur(image,(5,5),0)
    ret, image = cv2.threshold(image, 125, 255, 0)
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
        succ,frame=video.read()
        video.set(cv2.CAP_PROP_POS_FRAMES, i*skip)
        print(f"taking frame n° {i*skip}")
        frame=cv2.cvtColor(frame[:Y,:],cv2.COLOR_BGR2GRAY)
        video_l.append(frame)
        i+=1
    return video_l



video=cv2.VideoCapture(video_source)
skip=15
max=50

video_l=cut_frames(video,max,skip)
video_l=video_l
n_cut=video_l[0].shape[0]//6
video_l=cut_and_return(video_l,n_cut)

means=[np.var(x) for x in  [[ img.mean() for img in ps ] for ps in video_l]]
minimu=min(means)
pos=means.index(minimu)
print(minimu,pos,means)
#show video


key=-1 

succ,frame = video.read()

cut=(pos)*frame[:Y,:].shape[0]//(n_cut)+10
v_stack=frame[cut:cut+frame[:Y,:].shape[0]//n_cut]
frame[cut:cut+frame[:Y,:].shape[0]//n_cut]=cv2.add(frame[cut:cut+frame[:Y,:].shape[0]//n_cut],50)
i=0
video=cv2.VideoCapture(video_source)
#cv2.imshow("",frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
n_frame=video.get(cv2. CAP_PROP_FRAME_COUNT)
while succ and i<n_frame:
    i+=1
    succ,frame = video.read()
    v_stack=np.vstack([frame[cut:cut+frame[:Y,:].shape[0]//n_cut],v_stack])
    print(f"Frame n°,{i} out of {n_frame}" )

    #frame=make_things_better(frame)
    # cv2.imshow("",frame[cut:cut+frame[:Y,:].shape[0]//n_cut])
    if not True:
        cv2.namedWindow("v stack", cv2.WINDOW_NORMAL)
        cv2.imshow("v stack",v_stack)
        if key==ord("k"):
            break
        key=cv2.waitKey(1)
        if key==ord(" "):
            key = cv2.waitKey(99999999)
print("done")
cv2.imwrite("vstack_th.jpg",make_things_better(v_stack))
cv2.imwrite("vstack_noth.jpg",v_stack)
