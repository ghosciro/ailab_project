

import cv2

import pandas as pd

import skimage.io
from skimage import measure
from skimage.color import label2rgb


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
        df = df[df["area"]>(df["area"].max()/1.3)]
        df = [x for x in df["bbox-2"].to_list()[::-1]]
    
        lastValue = max(df)
        bestPick = {bpm: 0 for bpm in range(60, 140)}

        for bpm in bestPick:
            # A constant that is used to determine the range of values.
            CONSTANT = (200 - bpm) // 10
            listOfStartingNotes = [y for x in df for y in range(x - CONSTANT, x + CONSTANT)]

            jump = (lastValue - firstValue) / bpm/16
            count = firstValue

            while count < lastValue:
                if int(count) in listOfStartingNotes:
                    bestPick[bpm] += 1
                count += jump

        # Finding the maximum value in the dictionary and returning the key associated with it.
        maxForNow = 0
        solution = None
        for elem in bestPick:
            if bestPick[elem] >= maxForNow:
                maxForNow = bestPick[elem]
                solution = elem
        return solution, firstValue, lastValue


    def get_rectangles(self,binary_img):
        label_image = measure.label(binary_img, connectivity=2)
        label_image_rgb = label2rgb(label_image, image=binary_img, bg_label=0)
        props = measure.regionprops_table(label_image,binary_img, properties=["area", "bbox"],)
        df = pd.DataFrame(props)
        df = df[df["area"] > 10].iloc[::-1]
        skimage.io.imsave("save.jpg", label_image_rgb)
        return df
   
    def get_notes(self):
        k=-1
        x0 = self.notes_database["bbox-1"]
        names = self.notes_database["name"]
        x1 = self.notes_database["bbox-3"]
        test = cv2.Canny(self.Vsatck, 230, 250)
        df=self.get_rectangles(test)
        df.to_excel("rect.xlsx")
        bpm, firstValue, lastValue = self.chooseBPM(df)
        self.bpm=bpm
        print(bpm,firstValue,lastValue)
        height_step =((lastValue - firstValue) // bpm // 16)
        print(height_step)
        notes = []
        media = 0
        for i in range(df.values[0][3].min(), 0, -height_step):
            #print(i)
            note = []
            for element in df.values:
                if element[1]+5 < i+height_step and element[3]-5 > i:
                    if element[4]-element[2] > 30:
                        #print("found double note")
                        doublenotes = test[i:i+height_step, element[2]:element[4]]
                        #cv2.imshow("output",test[i+5:i+height_step-5, element[2]:element[4]]) 
                        #cv2.waitKey(0) 
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
                if k!=ord("k"):
                    cv2.imshow("output",test[i+3:i+height_step-3]) 
                    print(note)
                    k=cv2.waitKey(0)
                if k==ord("k"):
                    cv2.destroyAllWindows()

                notes.append(set(note))

        return(notes)

