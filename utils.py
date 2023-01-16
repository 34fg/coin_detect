import cv2
import numpy as np

def threshSetting(a):
    pass

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings",640,240)
cv2.createTrackbar("thresh1","Settings",50,255, threshSetting)
cv2.createTrackbar("thresh2","Settings", 100,255, threshSetting)


def imgProcessing(img):
    img_out= cv2.GaussianBlur(img,(5,5),3)
    thresh1= cv2.getTrackbarPos("thresh1","Settings") 
    thresh2= cv2.getTrackbarPos("thresh2","Settings") 
    img_out= cv2.Canny(img_out,thresh1,thresh2)
    kernel=np.ones((3,3),np.uint8)
    img_out=cv2.dilate(img_out,kernel,iterations=1)
    img_out=cv2.morphologyEx(img_out,cv2.MORPH_CLOSE,kernel)

    return img_out


def findContours(img, img_out, minArea=1000, sort=True, filter=0, drawCon=True, c=(255, 0, 0)):
    conFound = []
    imgContours = img.copy()
   # imgPre = cv2.cvtColor(imgPre, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            if len(approx) == filter or filter == 0:
                if drawCon: cv2.drawContours(imgContours, cnt, -1, c, 3)
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + (w // 2), y + (h // 2)
                cv2.rectangle(imgContours, (x, y), (x + w, y + h), c, 2)
                cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv2.FILLED)
                conFound.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort:
        conFound = sorted(conFound, key=lambda x: x["area"], reverse=True)

    return imgContours, conFound