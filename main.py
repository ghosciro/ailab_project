from getverticalimage import Vertical_image
from findontes import noterecognito
import pandas as pd
import cv2





video_source="video1.mp4"
Y=300
N_cut=6
if __name__ == '__main__':
    ##get V_stack
    video=cv2.VideoCapture(video_source)
    vertical_image=Vertical_image(video,N_cut)
    best_position=vertical_image.best_position()
    cut=((best_position)*((video.read()[1][:Y,:].shape[0])//N_cut))
    v_stack=(vertical_image.dovertical(cut,video_source))
    cv2.imwrite("a.jpg",v_stack)

    ##implement_keyboard recognition


    #note recognition
    notes=noterecognito(v_stack,pd.read_excel("coordinatesvid1.xlsx"))
    print(notes.get_notes(25))

    #rotellini conversion to sheet
    