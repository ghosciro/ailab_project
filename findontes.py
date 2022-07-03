

import cv2

import pandas as pd

import skimage.io
from skimage import measure
from skimage.color import label2rgb

''' the main function of this class is the one  of converting a Vstack into a list of notes, 
    Input: Vstack, Keys position in a dataframe'''
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
            # print(bpm, jump)
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

        # print(sorted(bestPick, key=bestPick.get, reverse=True))
        # print(bestPick)
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
        df.to_excel("rect.xlsx")
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

