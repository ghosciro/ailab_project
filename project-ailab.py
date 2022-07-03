from midiutil import MIDIFile
from multiprocessing import Process,managers
from collections import defaultdict
from itertools import combinations
from skimage import measure
from skimage.color import label2rgb
from console_progressbar import ProgressBar
import math
import pandas as pd
import os 
import pandas as pd
import cv2
import numpy as np

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
        succ,frame=self.video.read()
        worker=[]
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
        '''for each frame crop it at the position we have found before'''
        while succ and start+i<stop:
            i+=1
            if i%500 == 1:
                v_stack=np.vstack([frame[cut_value:cut_value+frame[:Y,:].shape[0]//n_cut],v_stack])
            succ,frame = video.read()
        stack_list[id]=[id,v_stack]
        return 
''' the main function of this class is the one  of converting a Vstack into a list of notes,'''
class noterecognito:
    def __init__(self, Vsatck, notees_database):
        self.Vsatck = Vsatck
        self.notes_database = notees_database
        self.bpm=0

    def chooseBPM(self,df):
        """
        It takes a dataframe of bounding boxes and returns the most likely BPM, the first note, and the last
        note
        """

        firstValue = df["bbox-0"].min()
        lastValue = df["bbox-2"].max()
        df = df[df["area"] > 400]

        df = df["bbox-2"].to_list()

        bestPick = {bpm: 0 for bpm in range(60, 240)}

        for bpm in bestPick:
            # A constant that is used to determine the range of values.

            # CONSTANT = (100 - bpm // 2) // 10

            # listOfStartingNotes = [y for x in df for y in range(x - CONSTANT, x + CONSTANT)]
            # setStarting = set(listOfStartingNotes)
            CONSTANT = 1
            bps = bpm / 60
            jump = 25 / bps

            one = set(range(-2, 3))
            two = set(range(int((jump - CONSTANT)), int((jump + CONSTANT)) + 1)).difference(
                one
            )
            four = set(
                range(int((jump - CONSTANT) * 4), int((jump + CONSTANT) * 4) + 1)
            ).difference(two)
            eight = set(
                range(int((jump - CONSTANT) * 8), int((jump + CONSTANT) * 8 + 1))
            ).difference(four)
            sixteen = set(
                range(int((jump - CONSTANT) * 16), int((jump + CONSTANT) * 16 + 1))
            ).difference(eight)
            threeTwo = set(
                range(int((jump - CONSTANT) * 32), int((jump + CONSTANT) * 32 + 1))
            ).difference(sixteen)
            sixFour = set(
                range(int((jump - CONSTANT) * 64), int((jump + CONSTANT) * 64 + 1))
            ).difference(threeTwo)
            for i in range(len(df)):
                for j in range(i, len(df)):
                    value = df[i] - df[j]
                    if value > (jump + CONSTANT) * 64:
                        break
                    if value in one:
                        bestPick[bpm] += 1
                    elif value in two:
                        bestPick[bpm] += 1
                    elif value in four:
                        bestPick[bpm] += 1
                    elif value in eight:
                        bestPick[bpm] += 1
                    elif value in sixteen:
                        bestPick[bpm] += 1
                    elif value in threeTwo:
                        bestPick[bpm] += 1
                    elif value in sixFour:
                        bestPick[bpm] += 1
        # Finding the maximum value in the dictionary and returning the key associated with it.

        return (
            max(bestPick, key=bestPick.get),
            firstValue,
            lastValue,
        )


    def get_rectangles(self,binary_img):
        ''' this function finds all the rectangles that are presents'''
        label_image = measure.label(binary_img, connectivity=2)
        label_image_rgb = label2rgb(label_image, image=binary_img, bg_label=0)
        props = measure.regionprops_table(label_image,binary_img, properties=["area", "bbox"],)
        df = pd.DataFrame(props)
        df = df[df["area"] > 10].iloc[::-1]
        return df
   
    def get_notes(self):
        ''' this is the main method of this class used in our program'''
        '''initialization of some variables'''
        k=-1
        notes = []
        media = 0
        '''reading the positions and the name   from the keyboard dataframe'''
        x0 = self.notes_database["bbox-1"] 
        names = self.notes_database["name"]
        x1 = self.notes_database["bbox-3"]
        '''doing operation on our image to feed it in get rectangles '''
        test = cv2.Canny(self.Vsatck, 230, 250)
        df=self.get_rectangles(test)
        '''finding the bmp thanks to the chooseBPM function'''
        bpm, firstValue, lastValue = self.chooseBPM(df)
        self.bpm=bpm
        '''evaluating the height of the sliding window'''
        height_step =((lastValue - firstValue) // bpm // 16)
        i=df.values[0][3].min()
        '''looping the sliding until we reach the top of the image '''
        while i>0:
            note = []
            '''looping for every rectangle found'''
            for element in df.values:
                '''checking if the rectangle is present in the current slice'''
                if element[1]+6 < i+height_step and element[3]-6 > i:
                    '''to solve the problem of two notes toghether, we do check if it's width is higher than one white note'''
                    if element[4]-element[2] > 30:
                        '''cropping the image at the current coordinates'''
                        doublenotes = test[i:i+height_step, element[2]:element[4]]
                        '''checking where the rectangles are in the current slice'''
                        if doublenotes[height_step//2-2:height_step//2+2, 0:5].any():
                            media = element[4]+5
                            for y in range(len(x0)):
                                if media >= x0[y] and media <= x1[y]:
                                    note.append(names[y])
                        if doublenotes[height_step//2-2:height_step//2+2, doublenotes.shape[1]-5:doublenotes.shape[1]].any():
                            media = element[4]-5
                            for y in range(len(x0)):
                                if media >= x0[y] and media <= x1[y]:
                                    note.append(names[y])
                    else:
                        '''evaluating the center point of the rectangle found'''
                        media = (element[2]+element[4])//2
                        '''iterating trough the notes and checking which is the correct note'''
                        for y in range(len(x0)):
                            if media >= x0[y] and media <= x1[y]:
                                note.append(names[y])
            notes.append(set(note))
        return(notes)
''' the main function of this class is the one of finding the position and name of the keys on the keyboard'''
class Coordinates:
    def __init__(self, img):
        self.keys_88 = [
            "A-0",
            "A#-0",
            "B-0",
            "C-1",
            "C#-1",
            "D-1",
            "D#-1",
            "E-1",
            "F-1",
            "F#-1",
            "G-1",
            "G#-1",
            "A-1",
            "A#-1",
            "B-1",
            "C-2",
            "C#-2",
            "D-2",
            "D#-2",
            "E-2",
            "F-2",
            "F#-2",
            "G-2",
            "G#-2",
            "A-2",
            "A#-2",
            "B-2",
            "C-3",
            "C#-3",
            "D-3",
            "D#-3",
            "E-3",
            "F-3",
            "F#-3",
            "G-3",
            "G#-3",
            "A-3",
            "A#-3",
            "B-3",
            "C-4",
            "C#-4",
            "D-4",
            "D#-4",
            "E-4",
            "F-4",
            "F#-4",
            "G-4",
            "G#-4",
            "A-4",
            "A#-4",
            "B-4",
            "C-5",
            "C#-5",
            "D-5",
            "D#-5",
            "E-5",
            "F-5",
            "F#-5",
            "G-5",
            "G#-5",
            "A-5",
            "A#-5",
            "B-5",
            "C-6",
            "C#-6",
            "D-6",
            "D#-6",
            "E-6",
            "F-6",
            "F#-6",
            "G-6",
            "G#-6",
            "A-6",
            "A#-6",
            "B-6",
            "C-7",
            "C#-7",
            "D-7",
            "D#-7",
            "E-7",
            "F-7",
            "F#-7",
            "G-7",
            "G#-7",
            "A-7",
            "A#-7",
            "B-7",
            "C-8",
        ]
        self.standard_C4_SHARP = 41
        self.img = img
        self.h, self.w = img.shape[:2]
        self.pts2 = np.float32([[0, 0], [1280, 0], [0, 200], [1280, 200]])
        self.rectangles = self.findRectangles()
        self.perspectiveRectangles = self.findPerspectiveRectangles()
        self.possible = self.findBestRectangle()
        self.coordinates = self.getCoordinates()

    def findRectangles(self):

        edges = cv2.Canny(self.img, 100, 200, None, 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200, None, 0, 0)

        lines = np.vstack(
            (
                lines,
                np.array(
                    [
                        [[0, np.pi / 2]],
                        [[self.h - 1, np.pi / 2]],
                        [[0, 0]],
                        [[self.w - 1, 0]],
                    ]
                ),
            )
        )

        segmented = self.segment_by_angle_kmeans(lines)

        comb_hor = list(combinations(segmented[0], 2))
        comb_ver = list(combinations(segmented[1], 2))

        rectangles = []

        for hor in comb_hor:
            for ver in comb_ver:
                vertici = []
                for ho in np.sort(hor, 0):
                    for v in np.sort(ver, 0):

                        rho1, theta1 = ho[0]
                        rho2, theta2 = v[0]

                        A = np.array(
                            [
                                [np.cos(theta1), np.sin(theta1)],
                                [np.cos(theta2), np.sin(theta2)],
                            ]
                        )
                        b = np.array([[rho1], [rho2]])
                        x0, y0 = np.linalg.solve(A, b)
                        x0, y0 = int(np.round(x0)), int(np.round(y0))
                        vertici.append([x0, y0])

                if abs(vertici[0][1] - vertici[-1][1]) > 30:
                    rectangles.append(vertici)
        return rectangles

    def findPerspectiveRectangles(self):
        perspectiveRectangles = []

        for i, pts1 in enumerate(self.rectangles):
            M = cv2.getPerspectiveTransform(np.float32(pts1), self.pts2)

            perspectiveRectangles.append(
                (i, cv2.warpPerspective(self.img, M, (1280, 200)))
            )
        return perspectiveRectangles

    # Sort by the mean of the last third of the image (since the bottom part of the keyboard is only white keys)
    # So the first rectangle of this list is the most likely to be the keyboard

    def findBestRectangle(self):

        temp = sorted(
            self.perspectiveRectangles,
            key=lambda x: np.mean(x[1][130:, :]),
            reverse=True,
        )
        possible = None

        for i, rect in temp:
            gray = cv2.cvtColor(rect[10:190], cv2.COLOR_BGR2GRAY)
            test = self.make_things_better(rect[10:190])

            label_image = measure.label(test, connectivity=2)
            label_image_rgb = label2rgb(label_image, image=test, bg_label=0)

            props = measure.regionprops_table(
                label_image,
                gray,
                properties=["label", "area", "mean_intensity", "bbox"],
            )

            df = pd.DataFrame(props)

            df = df[df["area"] > 900]

            if not possible or len(df) > len(possible[2]):

                possible = (i, np.mean(rect[130:, :]), df)

                #cv2.imshow("label_image_rgb", label_image_rgb)
                # cv2.imshow("test", test)
                # cv2.imshow("img", rect)
                #cv2.waitKey()
        return possible

    def getCoordinates(self):

        inv_M = cv2.getPerspectiveTransform(
            self.pts2, np.float32(self.rectangles[self.possible[0]])
        )

        bbox0 = []
        bbox1 = []
        bbox2 = []
        bbox3 = []

        df = self.possible[2].copy()
        points = []
        for index, elem in enumerate(self.possible[2].values):

            xleft, yleft = cv2.perspectiveTransform(
                np.array([[[elem[4], elem[3]]]]), inv_M
            )[0][0]
            xright, yright = cv2.perspectiveTransform(
                np.array([[[elem[6], elem[5]]]]), inv_M
            )[0][0]

            points.append(((xleft, yleft), (xright, yright)))

            bbox0.append(int(yleft))
            bbox1.append(int(xleft))
            bbox2.append(int(yright))
            bbox3.append(int(xright))

        df["bbox-0"] = bbox0
        df["bbox-1"] = bbox1
        df["bbox-2"] = bbox2
        df["bbox-3"] = bbox3
        df["label"] = list(range(1, len(bbox0) + 1))

        for topleft, botright in points:

            cv2.rectangle(
                self.img,
                [int(topleft[0]), int(topleft[1])],
                [int(botright[0]), int(botright[1])],
                color=(0, 255, 255),
            )

        C4_SHARP = self.find_middle_C4_SHARP(df)

        shift = abs(C4_SHARP - self.standard_C4_SHARP)

        df["name"] = self.keys_88[shift : len(self.possible[2]) + shift]
        return df

    def segment_by_angle_kmeans(self, lines, k=2, **kwargs):
        """Groups lines based on angle with k-means.

        Uses k-means on the coordinates of the angle on the unit circle
        to segment `k` angles inside `lines`.
        """

        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get("criteria", (default_criteria_type, 10, 1.0))
        flags = kwargs.get("flags", cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get("attempts", 10)

        # returns angles in [0, pi] in radians
        angles = np.array([line[0][1] for line in lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array(
            [[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles],
            dtype=np.float32,
        )

        # run kmeans on the coords
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)  # transpose to row vec

        # segment lines based on their kmeans label
        segmented = defaultdict(list)
        for i, line in enumerate(lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())
        return segmented

    def printLines(self, lines, img, h, w, color):
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + w * (-b)), int(y0 + h * (a)))
                pt2 = (int(x0 - w * (-b)), int(y0 - h * (a)))
                cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

    def make_things_better(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0.5
        )
        image = cv2.GaussianBlur(image, (3, 3), 0)
        ret, image = cv2.threshold(image, 150, 255, 0)
        return image

    def find_middle_C4_SHARP(self, df):

        temp_df = df[df["mean_intensity"] < df["mean_intensity"].max() // 3]
        octaves = []
        remainder = []
        tempone = list(temp_df["label"])
        possible = tempone[:5]
        i = 5

        while not octaves:

            if (
                (possible[1] - possible[0] == 2)
                and (possible[2] - possible[1] == 3)
                and (possible[3] - possible[2] == 2)
                and (possible[4] - possible[3] == 2)
            ):

                octaves.append(possible)
            else:
                remainder.append(possible.pop(0))
                possible.append(tempone[i])
                i += 1

        possible = []
        for elem in tempone[i:]:
            possible.append(elem)
            if len(possible) == 5:
                octaves.append(possible)
                possible = []

        C4_SHARP = octaves[(len(octaves) - 1) // 2][0]

        return C4_SHARP
''' the main function of this class is the one of converting a list of notes into a midi playable file'''
class tomidi:
    def __init__(self,bpm,notes):
        self.toMIDI(bpm,notes)
    def Magic_spartito(self,perception):    # note con intervalli più lunghi di 8 da fixare
        
        perception = perception + [[], [], [], [], [], [], [], []]
        note_interval={"A-0":0, "A#-0":0, "B-0":0, "C-1":0, "C#-1":0, "D-1":0, "D#-1":0, "E-1":0, "F-1":0, "F#-1":0, "G-1":0, "G#-1":0, "A-1":0, "A#-1":0, "B-1":0,
                "C-2":0, "C#-2":0, "D-2":0, "D#-2":0, "E-2":0, "F-2":0, "F#-2":0, "G-2":0, "G#-2":0, "A-2":0, "A#-2":0, "B-2":0,
                "C-3":0, "C#-3":0, "D-3":0, "D#-3":0, "E-3":0, "F-3":0, "F#-3":0, "G-3":0, "G#-3":0, "A-3":0, "A#-3":0, "B-3":0,
                "C-4":0, "C#-4":0, "D-4":0, "D#-4":0, "E-4":0, "F-4":0, "F#-4":0, "G-4":0, "G#-4":0, "A-4":0, "A#-4":0, "B-4":0,
                "C-5":0, "C#-5":0, "D-5":0, "D#-5":0, "E-5":0, "F-5":0, "F#-5":0, "G-5":0, "G#-5":0, "A-5":0, "A#-5":0, "B-5":0,
                "C-6":0, "C#-6":0, "D-6":0, "D#-6":0, "E-6":0, "F-6":0, "F#-6":0, "G-6":0, "G#-6":0, "A-6":0, "A#-6":0, "B-6":0,
                "C-7":0, "C#-7":0, "D-7":0, "D#-7":0, "E-7":0, "F-7":0, "F#-7":0, "G-7":0, "G#-7":0, "A-7":0, "A#-7":0, "B-7":0, "C-8":0}

        result = [[] for i in range(len(perception))]
        repeated=set()
        c=0
        for i,note_list in enumerate(perception):
            c+=1
            if note_list or repeated:
                for note in repeated.copy():
                    
                    if note not in note_list :
                        result[i-note_interval[note]].append((note, note_interval[note]))  #note interval and additional time for the legature
                        note_interval[note] = 0
                        repeated.remove(note)
                for note in note_list:
                    if note not in repeated:
                        repeated.add(note)
                        note_interval[note] += 1
                    else:
                        note_interval[note] += 1
            else:
                result[i] = []
            if c==16:
                c=0
        return result

    def toMIDI(self,bpm, perception):
        translation= {'C-0': 0, 'C#-0': 1, 'D-0': 2, 'D#-0': 3, 'E-0': 4, 'F-0': 5, 'F#-0': 6, 'G-0': 7, 'G#-0': 8, 'A-1': 9, 'A#-1': 10, 'B-1': 11, 'C-1': 12, 'C#-1': 13, 'D-1': 14, 'D#-1': 15, 'E-1': 16, 'F-1': 17, 'F#-1': 18, 'G-1': 19, 'G#-1': 20, 'A-2': 21, 'A#-2': 22, 'B-2': 23, 'C-2': 24, 'C#-2': 25, 'D-2': 26, 'D#-2': 27, 'E-2': 28, 'F-2': 29, 'F#-2': 30, 'G-2': 31, 'G#-2': 32, 'A-3': 33, 'A#-3': 34, 'B-3': 35, 'C-3': 36, 'C#-3': 37, 'D-3': 38, 'D#-3': 39, 'E-3': 40, 'F-3': 41, 'F#-3': 42, 'G-3': 43, 'G#-3': 44, 'A-4': 45, 'A#-4': 46, 'B-4': 47, 'C-4': 48, 'C#-4': 49, 'D-4': 50, 'D#-4': 51, 'E-4': 52, 'F-4': 53, 'F#-4': 54, 'G-4': 55, 'G#-4': 56, 'A-5': 57, 'A#-5': 58, 'B-5': 59, 'C-5': 60, 'C#-5': 61, 'D-5': 62, 'D#-5': 63, 'E-5': 64, 'F-5': 65, 'F#-5': 66, 'G-5': 67, 'G#-5': 68, 'A-6': 69, 'A#-6': 70, 'B-6': 71, 'C-6': 72, 'C#-6': 73, 'D-6': 74, 'D#-6': 75, 'E-6': 76, 'F-6': 77, 'F#-6': 78, 'G-6': 79, 'G#-6': 80, 'A-7': 81, 'A#-7': 82, 'B-7': 83, 'C-7': 84, 'C#-7': 85, 'D-7': 86, 'D#-7': 87, 'E-7': 88, 'F-7': 89, 'F#-7': 90, 'G-7': 91, 'G#-7': 92, 'A-8': 93, 'A#-8': 94, 'B-8': 95, 'C-8': 96, 'C#-8': 97, 'D-8': 98, 'D#-8': 99, 'E-8': 100, 'F-8': 101, 'F#-8': 102, 'G-8': 103, 'G#-8': 104, 'A-9': 105, 'A#-9': 106, 'B-9': 107, 'C-9': 108, 'C#-9': 109, 'D-9': 110, 'D#-9': 111, 'E-9': 112, 'F-9': 113, 'F#-9': 114, 'G-9': 115, 'G#-9': 116, 'A-10': 117, 'A#-10': 118, 'B-10': 119, 'C-10': 120, 'C#-10': 121, 'D-10': 122, 'D#-10': 123, 'E-10': 124, 'F-10': 125, 'F#-10': 126, 'G-10': 127}
        magic_spartito=Magic_spartito(perception)
        midi = MIDIFile(1)
        midi.addTempo(0, 4, bpm)
        time=0
        for note_list in magic_spartito:
            
            for note in note_list:
                midi.addNote(0, 0, translation[note[0]], time, note[1]/4, 100) #note[1]/4 for 1/8 #note[1]/8 for 1/32
            time += 0.25 #for 1/16
            #time += 0.03125 #for 1/32
        with open("./spartito.mid", "wb") as file:
            midi.writeFile(file)
        os.system('cmd /c "MidiSheetMusic-2.6.2.exe spartito.mid"')
        return

if __name__ == '__main__':
    pbar =ProgressBar(total=10,prefix='Here', suffix='Now', decimals=3, length=50, fill='*', zfill='-')
    video_source="video8.mp4"
    video=cv2.VideoCapture(video_source)
    '''initializing an instance of Vertical image class'''
    vertical_image=Vertical_image(video_source,10,3)
    pbar.print_progress_bar(1)
    '''finding the image of the keyboard without hands'''
    nohand=vertical_image.hand_remover()
    pbar.print_progress_bar(2)
    '''finding the best position to crop the image'''
    best_position=vertical_image.best_position()
    pbar.print_progress_bar(3)
    '''creating the Vstack'''
    cut=((best_position)*((video.read()[1][:Y,:].shape[0])//vertical_image.n_cut))
    v_stack=(vertical_image.dovertical(cut,video_source))
    pbar.print_progress_bar(5)

    '''using the function to return a data frame of keys'''
    keys=Coordinates(nohand).coordinates
    pbar.print_progress_bar(7)
    '''initializing an instance of noterocognito class '''
    notes=noterecognito(v_stack,keys)
    pbar.print_progress_bar(9)
    '''using the get notes function to find all the keys pressed'''
    soundednotes=notes.get_notes()
    pbar.print_progress_bar(10)
    '''conversion to sheet'''
    tomidi(notes.bpm,soundednotes)