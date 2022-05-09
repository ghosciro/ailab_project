
import cv2

def make_things_better(image,m1,m2,m3):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
    image=cv2.GaussianBlur(image,m1,0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,m2[0],m2[1])
    image = cv2.GaussianBlur(image,m3,0)
    ret, image = cv2.threshold(image, 125, 255, 0)
    return image

img=cv2.imread("vstack_noth.jpg")
m1=(3,3)
m2=(5,1)
m3=(3,3)
while True:
    img_m=make_things_better(img,m1,m2,m3)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame",img_m)
    key= cv2.waitKey(1)
    if key==ord("k"):
        break
    if key==ord("c"):
        m1=(int(input()),int(input()))
        m2=(int(input()),float(input()))
        m3=(int(input()),int(input()))
