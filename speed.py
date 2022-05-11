from sre_constants import SUCCESS
import cv2
import numpy as np
def find_speed(video,y1,y2,px):
    success,frame=video.read()
    pixels=frame[y1:y1+px]
    found=False
    i=0
    while success:
            print(frame[y1:y1+px]==pixels)
            if not (frame[y1:y1+px]==pixels).all():
                print("start counting")
                pixels=frame[y1:y1+px]
                found=True
                i=1
            if i!=0:
                i+=1
            if (frame[y2:y2+px]==pixels).all() and found:
                return i
            frame[y1:y1+px]=cv2.add(frame[y1:y1+px],50)
            frame[y2:y2+px]=cv2.add(frame[y2:y2+px],50)
            cv2.imshow("",frame)
            k=cv2.waitKey(33)
            success,frame=video.read()
            if k==ord("k"):
                return 0
print(find_speed(cv2.VideoCapture("video1.mp4"),276,18,6))