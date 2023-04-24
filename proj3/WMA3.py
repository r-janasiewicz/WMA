import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture('sawmovie.avi')
img1 = cv.imread('saw1.jpg',cv.IMREAD_GRAYSCALE)          # queryImage

scale_percent = 25 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv.resize(img1, dim, interpolation = cv.INTER_AREA)
    
    
# Initiate SIFT detector
    
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
    
kp1, des1 = sift.detectAndCompute(resized,None)

while(1):

    # Take each frame
    
    _, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizedf = cv.resize(gray_frame, dim, interpolation = cv.INTER_AREA)
    
    # find the keypoints and descriptors with SIFT
       
    kp2, des2 = sift.detectAndCompute(resizedf,None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
                    
    img3 = cv.drawMatchesKnn(resized,kp1,resizedf,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   
     
    cv.imshow('saw',img3)
    
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()