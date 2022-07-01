from getverticalimage import Vertical_image
from findontes import noterecognito
from makespartito_copy import toMIDI
from aosa import Coordinates
import pandas as pd
import cv2





video_source="video8.mp4"
Y=300
if __name__ == '__main__':
    ##get V_stack
    video=cv2.VideoCapture(video_source)
    vertical_image=Vertical_image(video_source,10,3)
    nohand=vertical_image.hand_remover()
    N_cut=vertical_image.n_cut
    best_position=vertical_image.best_position()
    cut=((best_position)*((video.read()[1][:Y,:].shape[0])//N_cut))
    v_stack=(vertical_image.dovertical(cut,video_source))
    ##implement_keyboard recognition

    rectangles=Coordinates(nohand).coordinates
    #note recognition
    notes=noterecognito(v_stack,rectangles)
    soundednotes=notes.get_notes()
    #print(soundednotes)
    #rotellini conversion to sheet
    toMIDI(notes.bpm,soundednotes)