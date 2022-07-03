import cv2
import numpy as np
from multiprocessing import Process,managers
Y=300


''' the main function of this class is the one  of reading the entire video and having as a result a Vstack'''
class Vertical_image:
    def __init__(self,video,tries,Y):
            self.Y=Y
            self.video_source=video
            self.kip=15
            self.max=50
            self.video=cv2.VideoCapture(video)
            self.n_cut=self.get_speed(tries)
            self.video=cv2.VideoCapture(video)
    '''this function tries to find the average speed of the rectangles'''
    def get_speed(self,n_try):
        frame_count = int(self.video.get(cv2. CAP_PROP_FRAME_COUNT))
        self.video.set(cv2.CAP_PROP_POS_FRAMES,frame_count//2 )
        x_old=y_old=w_old=h_old=area_old = 0
        flag=1
        speed=[]
        ''' for a number of tries find all the rectangles and search for the same found before'''
        for i in range(n_try):
            _,frame = self.video.read()
            frame=frame[:Y]
            canny = cv2.Canny(frame, 230, 250)
            contours=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
            rectangles=[]
            for i in contours:
                x,y,w,h = cv2.boundingRect(i)
                if y+h<Y-10 and y>y_old :
                    rectangles.append([x,y,w,h,w*h])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            rectangles.sort(key= lambda x: x[1])
            speed.append((rectangles[0][1]+rectangles[0][3])-(y_old+h_old))
            x_old,y_old,w_old,h_old,area_old = rectangles[0]
        '''return the mean of the rectangles found'''
        return frame[:Y].shape[0]//int(np.mean(speed[1:]))
    '''this function is needed to try and get the most clean picture of the keyboard, removing hands if there are any'''
    def hand_remover(self,MAX_FRAME=300,DISTANCE = 10):
                
            video=cv2.VideoCapture(self.video_source)
            succ,frame=video.read()
            video_l=[]
            i=0
            '''getting  MAX_FRAME number of frame from the vide every DISTANCE frame'''
            while succ and len(video_l)<MAX_FRAME:
                if i==DISTANCE:
                        video_l.append(frame)
                        i=0
                succ,frame=video.read()
                i+=1
            video_l=np.array(video_l)
            '''using the addWeighted to try and remove the hand'''
            dst = video_l[0]
            for i in range(1,len(video_l)):
                    alpha = 1.0/(i)#1/2,1/3,1/4
                    beta = 1.0 - alpha#1/2,2/3,3/4
                    dst = cv2.addWeighted(video_l[i], alpha, dst, beta, 0.0)#somma pesata 
            return dst

    def make_things_better(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
        image=cv2.GaussianBlur(image,(5,5),0)
        image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        image = cv2.GaussianBlur(image,(5,5),0)
        ret, image = cv2.threshold(image, 125, 255, 0)
        return image

    ''' this method saves a max number of frame, that later are going to be used for finding the best spot'''
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

    '''this function takes as input a list of frames, cuts each of it in the selected number of parts and then return a matrix'''
    def cut_and_return(self,video_l):
        list_of_pieces=[]
        maxi=len(video_l)-1
        '''cropping each frame of the list of frames '''
        for screen in video_l:
            pieces=[]
            for y in range(0,screen.shape[0]-screen.shape[0]//self.n_cut,screen.shape[0]//self.n_cut):
                pieces.append(screen[y:y+screen.shape[0]//self.n_cut])
            list_of_pieces.append(pieces)  
        ''' rearranging the list in a way tho have a matrix (number of cuts X number of frames)'''
        foo=[[ [] for _ in range(maxi)] for  _ in range(self.n_cut-1) ]
        for i in range(self.n_cut-1):
            for y in range(maxi):
                foo[i][y]=list_of_pieces[y][i]
        return foo

 
    '''this method uses the two functions  above to evaluate the variance of the mean of each frame, selecting the one that has less variance '''
    def best_position(self):
        var=[]
        video_f=self.cut_frames()
        video_cf=self.cut_and_return(video_f)
        for position in video_cf:
            means=[]
            for img in position:
                means.append(np.mean(img))
            var.append(np.var(means))
        return(var.index(min(var)))

    '''this is the method called to have a vstack as a final result'''
    def dovertical(self,cut,name):
        n_frame=self.video.get(cv2. CAP_PROP_FRAME_COUNT)
        '''number of process to solve this operation, and initialization of the multiprocessing procedure'''
        n_worker=5
        frame_per_process=int(n_frame//n_worker)
        manage=managers.SyncManager()
        manage.start()
        list_of=manage.dict()
        print("every worker has:",frame_per_process,"frames")
        succ,frame=self.video.read()
        print("dimensions:",cut,cut+(frame[:Y,:].shape[0]//self.n_cut))
        worker=[]
        '''starting each process with it's own range of frame to crop'''
        for i in range(0,n_worker):
            worker.append(Process(target=Vertical_image.stack_it_all,args=(i,self.video_source,frame_per_process*i,frame_per_process*(i+1),list_of,cut,Y,self.n_cut)))
        for job in worker:
            job.start()
        for job in worker:
            job.join()
            v_stack=list_of[0][1]
        for i in range(1,n_worker):
            v_stack=np.vstack([list_of[i][1],v_stack])
        return(v_stack)

    '''method that each process solves'''
    def stack_it_all(id,video_name,start,stop,stack_list,cut_value,Y,n_cut):
        '''read the video and go to the starting point'''
        video=cv2.VideoCapture(video_name)
        video.set(cv2.CAP_PROP_POS_FRAMES, start)
        succ,frame = video.read()
        v_stack=frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut]
        i=0
        print(f"i'm worker n° {id} i will start at {start} and end at {stop}")
        '''for each frame crop it at the position we have found before'''
        while succ and start+i<stop:
            i+=1
            if i%500 == 1:
                print(f"i'm {id} at position {start+i} out of {stop}")
            v_stack=np.vstack([frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut],v_stack])
            succ,frame = video.read()
        print(f"ended after {i} iterations! id:{id}")
        stack_list[id]=[id,v_stack]
        return 
