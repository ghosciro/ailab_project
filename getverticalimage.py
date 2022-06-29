import cv2
import numpy as np
from multiprocessing import Process,managers
Y=300
video_source="video1.mp4"


class Vertical_image:
    def __init__(self,video,ncut):
            self.kip=15
            self.max=50
            self.video=video
            self.n_cut=ncut
    def shiftimage(self,img,x,y):
        M = np.float32([
	    [1, 0, x],#xcoordinate
	    [0, 1, y]])#ycoordinate
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return shifted

    def make_things_better(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
        image=cv2.GaussianBlur(image,(5,5),0)
        image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        image = cv2.GaussianBlur(image,(5,5),0)
        ret, image = cv2.threshold(image, 125, 255, 0)
        return image

    def cut_and_return(self,video_l):
        list_of_pieces=[]
        maxi=len(video_l)-1
        for screen in video_l:
            pieces=[]
            for y in range(0,screen.shape[0]-screen.shape[0]//self.n_cut,screen.shape[0]//self.n_cut):
                pieces.append(screen[y:y+screen.shape[0]//self.n_cut])
            list_of_pieces.append(pieces)  
        foo=[[ [] for _ in range(maxi)] for  _ in range(self.n_cut-1) ]
        for i in range(self.n_cut-1):
            for y in range(maxi):
                foo[i][y]=list_of_pieces[y][i]
        return foo

    def cut_frames(self):
        video_l=[]
        i=0
        succ=True
        while succ and len(video_l)<=self.max:
            succ,frame=self.video.read()
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i*self.kip)
            frame=cv2.cvtColor(frame[:Y,:],cv2.COLOR_BGR2GRAY)
            video_l.append(frame)
            i+=1
        return video_l  
    def best_position(self):
        var=[]
        video_l=self.cut_frames()
        video=self.cut_and_return(video_l)
        for position in video:
            means=[]
            for img in position:
                means.append(np.mean(img))
            var.append(np.var(means))
        return(var.index(min(var)))
    def dovertical(self,cut,name):
        n_frame=self.video.get(cv2. CAP_PROP_FRAME_COUNT)
        n_worker=5
        frame_per_process=int(n_frame//n_worker)
        manage=managers.SyncManager()
        manage.start()
        list_of=manage.dict()
        print("every worker has:",frame_per_process,"frames")
        succ,frame=self.video.read()
        print("dimensions:",cut,cut+(frame[:Y,:].shape[0]//self.n_cut))
        worker=[]
        for i in range(0,n_worker):
            worker.append(Process(target=Vertical_image.stack_it_all,args=(i,video_source,frame_per_process*i,frame_per_process*(i+1),list_of,cut,Y,cut+frame[:Y,:].shape[0]//self.n_cut)))
        for job in worker:
            job.start()
        for job in worker:
            job.join()
            v_stack=list_of[0][1]
        for i in range(1,n_worker):
            v_stack=np.vstack([list_of[i][1],v_stack])
        return(v_stack)
    def stack_it_all(id,video_name,start,stop,stack_list,cut_value,Y,n_cut):
        video=cv2.VideoCapture(video_name)
        video.set(cv2.CAP_PROP_POS_FRAMES, start)
        succ,frame = video.read()
        v_stack=frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut]
        i=0
        print(f"i'm worker n° {id} i will start at {start} and end at {stop}")
        while succ and start+i<stop:
            i+=1
            if i%500 == 1:
                print(f"i'm {id} at position {start+i} out of {stop}")
            v_stack=np.vstack([frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut],v_stack])
            succ,frame = video.read()
        print(f"ended after {i} iterations! id:{id}")
        stack_list[id]=[id,v_stack]
        return 
