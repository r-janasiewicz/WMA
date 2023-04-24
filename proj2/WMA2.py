import cv2 as cv
import numpy as np

def chceck(x,y,x0,y0,x1,y1):
    if( x0 < x < x1 and y0 < y < y1):
        #srodek
        return 1
    else:
        #zewnatrz
        return 0
trays=[1,2,3,4,5,6,7,8]
color = (127,0,255)
for i in trays:
        print('tray'+str(i)+'.jpg')      
        img = cv.imread('tray'+ str(i) +'.jpg', cv.IMREAD_COLOR)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        #img for circles
        img_cir = cv.GaussianBlur(img,(5,5),0)
        img_cir = cv.medianBlur(img_cir,5)
        img_cir = cv.cvtColor(img_cir,cv.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        dst = cv.filter2D(img_cir,-1,kernel)
        
        edges = cv.Canny(gray,50,150,apertureSize = 3)
        lines = cv.HoughLinesP(edges ,1,np.pi/180,100, minLineLength=50, maxLineGap=10)

        x = []
        y = []

        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
            y.append(line[0][1])
            y.append(line[0][3])

        x0 = min(x)
        x1 = max(x)
        y0 = min(y)
        y1 = max(y)        

        for line in lines:
            x2,y2,x3,y3 = line[0]
            cv.line(img,(x2,y2),(x3,y3),(0,255,0),2)

        cv.line(img,(x0,y0),(x1,y0),color,5)
        cv.line(img,(x0,y0),(x0,y1),color,5)
        cv.line(img,(x1,y1),(x1,y0),color,5)
        cv.line(img,(x1,y1),(x0,y1),color,5)

        circles = cv.HoughCircles(dst,cv.HOUGH_GRADIENT,1,15, param1=50,param2=30,minRadius=20,maxRadius=40)
        circles = np.uint16(np.around(circles))
        radiusSize = []
        for i in circles[0,:]:
            radiusSize.append(i[2])

        sumIn = 0
        countIn =0
        sumOut = 0
        countOut =0

        for i in circles[0,:]:
            zm = 0.05       
            if(i[2] > (max(radiusSize)+min(radiusSize))/2):
                zm = 5
                
            cv.circle(img,(i[0],i[1]),i[2],color,2)
            
            if (chceck(i[0],i[1],x0,y0,x1,y1) == 0):
                sumOut = sumOut + zm
                countOut = countOut + 1
            else:
                sumIn = sumIn + zm
                countIn = countIn +1
            cv.circle(img,(i[0],i[1]),2,color,3)


        print('count in: ', countIn, 'sum in: ', round(sumIn,2))
        print('count out: ', countOut, 'sum out: ', round(sumOut,2))
        cv.imshow('img',img)
        
        k = cv.waitKey(0)
#k = cv.waitKey(0)