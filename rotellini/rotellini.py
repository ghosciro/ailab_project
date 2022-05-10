import cv2
import matplotlib.pyplot as plt

method = cv2.TM_SQDIFF_NORMED

# Read the images from the file
small_image = cv2.imread('./cutted4.jpg')
large_image = cv2.imread('./prova3.jpg')

result = cv2.matchTemplate(small_image, large_image, method)

# We want the minimum squared difference
mn,_,mnLoc,_ = cv2.minMaxLoc(result)

# Draw the rectangle:
# Extract the coordinates of our best match
MPx,MPy = mnLoc

# Step 2: Get the size of the template. This is the same size as the match.
trows,tcols = small_image.shape[:2]

# Step 3: Draw the rectangle on large_image
cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

# Display the original image with the rectangle around the match.
cv2.imshow('not scaled match',large_image)
prova = cv2.Canny(small_image,50,200)
cv2.imshow('prova',prova)





image = cv2.imread('./prova3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None

import numpy as np
import imutils

template = cv2.imread("./cutted4.jpg")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

#plt.imshow(template)
#plt.show()
# loop over the scales of the image
for scale in np.linspace(0.8, 1.0, 20)[::-1]:

	# resize the image according to the scale, and keep track
	# of the ratio of the resizing
	resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
	r = gray.shape[1] / float(resized.shape[1])

	# if the resized image is smaller than the template, then break
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break
	
	# detect edges in the resized, grayscale image and apply template
	# matching to find the template in the image
	edged = cv2.Canny(resized, 50, 200)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

	# if we have found a new maximum correlation value, then update
	# the bookkeeping variable
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("scaled result", image)
cv2.waitKey(0)