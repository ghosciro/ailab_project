from console_progressbar import ProgressBar
from getverticalimage import Vertical_image
from findontes import noterecognito
from makespartito_copy import toMIDI
from aosa import Coordinates
import pandas as pd
import cv2





video_source="video8.mp4"
video=cv2.VideoCapture(video_source)
Y=300
if __name__ == '__main__':
    pbar =ProgressBar(total=7,prefix='Here', suffix='Now', decimals=3, length=50, fill='*', zfill='-')
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
    pbar.print_progress_bar(4)

    '''using the function to return a data frame of keys'''
    keys=Coordinates(nohand).coordinates
    pbar.print_progress_bar(5)
    '''initializing an instance of noterocognito class '''
    notes=noterecognito(v_stack,keys)
    pbar.print_progress_bar(6)
    '''using the get notes function to find all the keys pressed'''
    soundednotes=notes.get_notes()
    pbar.print_progress_bar(7)
    '''conversion to sheet'''
    tomidi(notes.bpm,soundednotes)