import pandas as pd
import cv2


def chooseBPM(df):
    """
    It takes a dataframe of bounding boxes and returns the most likely BPM, the first note, and the last
    note
    """

    firstValue = df["bbox-0"].min()
    df = [x for x in df["bbox-2"].to_list()[::-1]]
    lastValue = max(df)
    bestPick = {bpm: 0 for bpm in range(60, 140)}

    for bpm in bestPick:
        # A constant that is used to determine the range of values.
        CONSTANT = (200 - bpm) // 10
        listOfStartingNotes = [y for x in df for y in range(x - CONSTANT, x + CONSTANT)]

        jump = (lastValue - firstValue) / bpm
        count = firstValue

        while count < lastValue:
            if int(count) in listOfStartingNotes:
                bestPick[bpm] += 1
            count += jump

    print(bestPick)
    # Finding the maximum value in the dictionary and returning the key associated with it.
    maxForNow = 0
    solution = None
    for elem in bestPick:
        if bestPick[elem] >= maxForNow:
            maxForNow = bestPick[elem]
            solution = elem
    return solution, firstValue, lastValue


df = pd.read_excel("rect.xlsx")
bpm, firstValue, lastValue = chooseBPM(df)

print("bpm", bpm)

img = cv2.imread("save.jpg")
count = firstValue
jump = (lastValue - firstValue) / bpm / 4
while count < lastValue:

    cv2.line(img, (0, int(count)), (1280, int(count)), (0, 255, 0), thickness=2)
    count += jump

cv2.imwrite("prova.jpg", img)
