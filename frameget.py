import cv2
from matplotlib.image import imsave
import numpy as np
video=cv2.VideoCapture("video1.mp4")
print(video.get(cv2. CAP_PROP_FRAME_COUNT))
video.set(cv2.CAP_PROP_POS_FRAMES, 4800)
succ,frame=video.read()
imsave("frame_rotello.jpg",frame)