import cv2
from cv2 import rectangle
import numpy as np
from multiprocessing import Process,Value,Array,managers
Y=300
video_source="video1.mp4"
def shiftimage(img,x,y):
    M = np.float32([
	[1, 0, x],#xcoordinate
	[0, 1, y]])#ycoordinate
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted

def make_things_better(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
    image=cv2.GaussianBlur(image,(5,5),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
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
        frame=cv2.cvtColor(frame[:Y,:],cv2.COLOR_BGR2GRAY)
        video_l.append(frame)
        i+=1
    return video_l

def stack_it_all(id,video_name,start,stop,stack_list,cut_value,Y,n_cut):
    video=cv2.VideoCapture(video_name)
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    succ,frame = video.read()
    ##frame=cv2.Canny(frame,100,200)
    v_stack=frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut]
    i=0
    print(f"i'm worker n° {id} i will start at {start} and end at {stop}")
    while succ and start+i<stop:
        i+=1
        if i%500 == 1:
            print(f"i'm {id} at position {start+i} out of {stop}")
        ##frame=cv2.Canny(frame,100,200)
        v_stack=np.vstack([frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut],v_stack])
        succ,frame = video.read()
    print(f"ended after {i} iterations! id:{id}")
    stack_list[id]=[id,v_stack]
    return 




if __name__ == '__main__':
    video=cv2.VideoCapture(video_source)
    skip=15
    max=50
    video_l=cut_frames(video,max,skip)
    video_rev=video_l[::-1] # for reversing
    n_cut=video_l[0].shape[0]//6
    print(n_cut)
    video_l=cut_and_return(video_l,n_cut)
    video_rev=cut_and_return(video_rev,n_cut)

    means=[np.var(x) for x in  [[ img.mean() for img in ps ] for ps in video_l]]
    minimu=min(means)
    pos1=means.index(minimu)
    means=[np.var(x) for x in  [[ img.mean() for img in ps ] for ps in video_rev]]
    minimu=min(means)
    pos2=means.index(minimu)
#show video


    key=-1 
    print(pos1,pos2)
    succ,frame = video.read()
    cuts=[0,0]
    cuts[0]=((n_cut-1)-pos1)*frame[:Y,:].shape[0]//(n_cut) ## reversed image order 
    cuts[1]=((pos2)*frame[:Y,:].shape[0]//(n_cut)) #starting from the top
    print(cuts)
    n_frame=video.get(cv2. CAP_PROP_FRAME_COUNT)
    n_worker=5
    frame_per_process=int(n_frame//n_worker)
    manage=managers.SyncManager()
    manage.start()
    list_of=manage.dict()
    print("every worker has:",frame_per_process,"frames")
    flag=0
    for cut in cuts:
        print("dimensions:",cut,cut+frame[:Y,:].shape[0]//n_cut)
        worker=[]
        for i in range(0,n_worker):
            worker.append(Process(target=stack_it_all,args=(i,video_source,frame_per_process*i,frame_per_process*(i+1),list_of,cut,Y,n_cut)))
        for job in worker:
            job.start()
        for job in worker:
            job.join()
        v_stack=list_of[0][1]
        for i in range(1,n_worker):
            v_stack=np.vstack([list_of[i][1],v_stack])
        if flag==0:
            #(Ysopra-Ysotto)-2*pixelpresi-(pixelpresi/2)
            v_stack1=v_stack
            flag=1
        else:
            cv2.imwrite(f"{video_source}-vstack.jpg",v_stack)
            #h_stack=cv2.addWeighted(v_stack1,0.2,v_stack,1,0)#somma pesata 
            h_stack=np.hstack([v_stack1,v_stack])

