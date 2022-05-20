
import cv2
import numpy as np
def shiftimage(img,x,y):
    M = np.float32([
	[1, 0, x],#xcoordinate
	[0, 1, y]])#ycoordinate
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted

def make_things_better(image):
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #nohist,55,3,1,nogaussian2,nothreshold2
    image=cv2.GaussianBlur(image,(11,11),0)
    image =cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,1)
    image = cv2.GaussianBlur(image,(11,11),0)
    ret, image = cv2.threshold(image, 125, 255, 0)
    return image

def functions(frame):          
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 150, 200)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_g=[]
    for ct in contours:
        perimeter = cv2.arcLength(ct,True)
        area = cv2.contourArea(ct)
        if area>20 and perimeter >50:
            epsilon = 0.001*cv2.arcLength(ct,True)
            approx = cv2.approxPolyDP(ct,epsilon,True)
            contours_g.append(approx)
        #x,y,w,h = cv2.boundingRect(approx)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    return contours_g
img2=cv2.imread("lower_image.jpg")
img1=cv2.imread("Top_image.jpg")
contours1=functions(img1)
contours2=functions(img2)
dimension=[]
for ct in contours1:
    (x,y,w,h) = cv2.boundingRect(ct)
    area=w*h
    if area >1000:
        dimension.append([w,h,x,y])
dimension2=[]
y_shift=[]
for ct in contours2:
    (x,y,w,h) = cv2.boundingRect(ct)
    ds=[x for x  in dimension if( x[0]-w>=-5 and x[0]-w<=5 )and( x[1]-h>=-5 and x[1]-h<=5) ]
    if ds: 
        ds=np.array(ds)
        ds=ds[:,2:4]
        x=np.min(np.abs(ds[:,0]-x))
        if x<10 and x>-10:
            y=(np.min(np.abs(ds[:,1]-y)))
            if y>200 and y<600:
                y_shift.append(y)
                print(y_shift[-1])
y_shift=np.mean(y_shift)
print(y_shift)
newimage=cv2.addWeighted(img1,0.2,shiftimage(img2,0,-y_shift),1,0)
#newimage=cv2.Canny(newimage,100,180)
img=cv2.Canny(newimage,150,200)
cv2.imwrite("i.jpg",img)
