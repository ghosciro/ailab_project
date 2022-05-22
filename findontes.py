
import cv2
import numpy as np 
import mingus.extra.lilypond as LilyPond
import mingus.containers as container
img=cv2.imread("i.jpg",cv2.IMREAD_GRAYSCALE)
notes=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
height_step=20
N = container.Track()
for indexes in range(img.shape[0],height_step,-height_step):
    element=img[indexes-height_step:indexes]
    slicing=12
    element_list=[element[:,i*element.shape[1]//slicing:(i+1)*element.shape[1]//slicing] for i in range(0,slicing)]
    for i,e in enumerate( element_list):  
        if np.count_nonzero(e)>50:
            N + notes[i]
    N.add_bar(container.Bar().place_rest())
bar = LilyPond.from_Track(N) 
print(bar)
LilyPond.to_png(bar, "my_first_bar")
