

import cv2

import pandas as pd

import skimage.io
from skimage import measure
from skimage.color import label2rgb


class noterecognito:
    def __init__(self, Vsatck, notees_database):
        self.Vsatck = Vsatck
        self.notes_database = notees_database
    
    def get_rectangles(self,binary_img):
        label_image = measure.label(binary_img, connectivity=2)
        label_image_rgb = label2rgb(label_image, image=binary_img, bg_label=0)
        props = measure.regionprops_table(label_image,binary_img, properties=["area", "bbox"],)
        df = pd.DataFrame(props)
        df = df[df["area"] > 200].iloc[::-1]
        skimage.io.imsave("save.jpg", label_image_rgb)
        return df
   
    def get_notes(self, height_step):
        x0 = self.notes_database["bbox-1"]
        names = self.notes_database["name"]
        x1 = self.notes_database["bbox-3"]
        test = cv2.Canny(self.Vsatck, 230, 250)
        df=self.get_rectangles(test)
        notes = []
        media = 0
        for i in range(df.values[0][3], 0, -height_step):
            note = []
            for element in df.values:
                if element[1] <= i+height_step and element[3] >= i:
                    if element[4]-element[2] > 30:
                        doublenotes = test[i:i+height_step, element[2]:element[4]]
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
                        media = (element[2]+element[4])//2
                        for y in range(len(x0)):
                            if media >= x0[y] and media <= x1[y]:
                                note.append(names[y])
            if note != []:
                notes.append(note)
        return(notes)

