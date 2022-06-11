from turtle import pos
import cv2
import numpy as np
import math
from collections import defaultdict
from itertools import combinations
from skimage import measure
from skimage.color import label2rgb, rgb2gray
import pandas as pd

keys_88 = [
    "A-0",
    "A#-0",
    "B-0",
    "C-1",
    "C#-1",
    "D-1",
    "D#-1",
    "E-1",
    "F-1",
    "F#-1",
    "G-1",
    "G#-1",
    "A-1",
    "A#-1",
    "B-1",
    "C-2",
    "C#-2",
    "D-2",
    "D#-2",
    "E-2",
    "F-2",
    "F#-2",
    "G-2",
    "G#-2",
    "A-2",
    "A#-2",
    "B-2",
    "C-3",
    "C#-3",
    "D-3",
    "D#-3",
    "E-3",
    "F-3",
    "F#-3",
    "G-3",
    "G#-3",
    "A-3",
    "A#-3",
    "B-3",
    "C-4",
    "C#-4",
    "D-4",
    "D#-4",
    "E-4",
    "F-4",
    "F#-4",
    "G-4",
    "G#-4",
    "A-4",
    "A#-4",
    "B-4",
    "C-5",
    "C#-5",
    "D-5",
    "D#-5",
    "E-5",
    "F-5",
    "F#-5",
    "G-5",
    "G#-5",
    "A-5",
    "A#-5",
    "B-5",
    "C-6",
    "C#-6",
    "D-6",
    "D#-6",
    "E-6",
    "F-6",
    "F#-6",
    "G-6",
    "G#-6",
    "A-6",
    "A#-6",
    "B-6",
    "C-7",
    "C#-7",
    "D-7",
    "D#-7",
    "E-7",
    "F-7",
    "F#-7",
    "G-7",
    "G#-7",
    "A-7",
    "A#-7",
    "B-7",
    "C-8",
]
standard_C4_SHARP = 41


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get("criteria", (default_criteria_type, 10, 1.0))
    flags = kwargs.get("flags", cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get("attempts", 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array(
        [[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32
    )

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def printLines(lines, img, h, w, color):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + w * (-b)), int(y0 + h * (a)))
            pt2 = (int(x0 - w * (-b)), int(y0 - h * (a)))
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


def make_things_better(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0.5
    )
    image = cv2.GaussianBlur(image, (3, 3), 0)
    ret, image = cv2.threshold(image, 150, 255, 0)
    return image


def find_middle_C4_SHARP(df):

    temp_df = df[df["mean_intensity"] < 20]
    octaves = []
    remainder = []
    tempone = list(temp_df["label"])
    possible = tempone[:5]
    i = 5

    while not octaves:

        if (
            (possible[1] - possible[0] == 2)
            and (possible[2] - possible[1] == 3)
            and (possible[3] - possible[2] == 2)
            and (possible[4] - possible[3] == 2)
        ):

            octaves.append(possible)
        else:
            remainder.append(possible.pop(0))
            possible.append(tempone[i])
            i += 1

    possible = []
    for elem in tempone[i:]:
        possible.append(elem)
        if len(possible) == 5:
            octaves.append(possible)
            possible = []

    C4_SHARP = octaves[(len(octaves) - 1) // 2][0]

    return C4_SHARP


img = cv2.imread("video1.mp4_snapshot_03.11.846.jpg")
# img = cv2.imread("video1.mp4_snapshot_00.13.237.jpg")
img = cv2.imread("testino.jpg")

h, w = img.shape[:2]

edges = cv2.Canny(img, 100, 200, None, 3)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 200, None, 0, 0)

lines = np.vstack(
    (lines, np.array([[[0, np.pi / 2]], [[h - 1, np.pi / 2]], [[0, 0]], [[w - 1, 0]]]))
)

segmented = segment_by_angle_kmeans(lines)


comb_hor = list(combinations(segmented[0], 2))
comb_ver = list(combinations(segmented[1], 2))


rectangles = []

for hor in comb_hor:
    for ver in comb_ver:
        vertici = []
        for ho in np.sort(hor, 0):
            for v in np.sort(ver, 0):

                rho1, theta1 = ho[0]
                rho2, theta2 = v[0]

                A = np.array(
                    [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
                )
                b = np.array([[rho1], [rho2]])
                x0, y0 = np.linalg.solve(A, b)
                x0, y0 = int(np.round(x0)), int(np.round(y0))
                vertici.append([x0, y0])

        if abs(vertici[0][1] - vertici[-1][1]) > 30:
            rectangles.append(vertici)


perspectiveRectangles = []

pts2 = np.float32([[0, 0], [1400, 0], [0, 300], [1400, 300]])

for i, pts1 in enumerate(rectangles):
    M = cv2.getPerspectiveTransform(np.float32(pts1), pts2)
    perspectiveRectangles.append((i, cv2.warpPerspective(img, M, (1400, 300))))

# Sort by the mean of the last third of the image (since the bottom part of the keyboard is only white keys)
# So the first rectangle of this list is the most likely to be the keyboard
temp = sorted(perspectiveRectangles, key=lambda x: np.mean(x[1][200:, :]), reverse=True)

possible = None

for i, rect in temp:

    gray = cv2.cvtColor(rect[10:280, :], cv2.COLOR_BGR2GRAY)
    test = make_things_better(rect[10:280])

    ret, thresh1 = cv2.threshold(
        gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    label_image = measure.label(test, connectivity=thresh1.ndim)
    label_image_rgb = label2rgb(label_image, image=test, bg_label=0)

    props = measure.regionprops_table(
        label_image, gray, properties=["label", "area", "mean_intensity", "bbox"],
    )

    df = pd.DataFrame(props)

    df = df[df["area"] > 1500]

    if not possible or len(df) > len(possible[2]):

        possible = (i, np.mean(rect[200:, :]), df)

        # cv2.imshow("label_image_rgb", label_image_rgb)
        # cv2.imshow("test", test)
        # cv2.imshow("img", rect)
        # cv2.waitKey()


C4_SHARP = find_middle_C4_SHARP(possible[2])

shift = abs(C4_SHARP - standard_C4_SHARP)

possible[2]["name"] = keys_88[shift : len(possible[2]) + shift]

possible[2].to_excel("output_test.xlsx")

printLines(segmented[0], img, h, w, (255, 0, 255))
printLines(segmented[1], img, h, w, (0, 0, 255))
cv2.imshow("", img)
cv2.waitKey(0)
