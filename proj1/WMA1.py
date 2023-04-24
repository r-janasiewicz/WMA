import numpy as np
import cv2 as cv

cap = cv.VideoCapture('movingball.avi')
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((15,15),np.uint8)

while(1):
    # Take each frame
    _, frame = cap.read()
    
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    
    # Convert BGR to HSV
    
    hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    
    lower_red = np.array([0,50,50])
    upper_red = np.array([4,255,255])
    
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    
    full_mask = mask1 + mask2
    
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(resized,resized, mask= full_mask)
    opening = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel2)
    
    # convert image to grayscale image
    gray_image = cv.cvtColor(closing, cv.COLOR_BGR2GRAY)
 
    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(gray_image,127,255,0)
 
    # calculate moments of binary image
    M = cv.moments(thresh)
 
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
    # put highlight in the center
    cv.circle(resized, (cX, cY), 4, (255, 230, 0), -1)
    #cv.putText(img, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
    # display the image
    cv.imshow('frame',resized)
    cv.imshow('mask',full_mask)
    cv.imshow('res',closing)
    k = cv.waitKey(7) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()