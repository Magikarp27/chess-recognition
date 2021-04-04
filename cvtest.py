import numpy as np
import cv2 as cv
from operator import itemgetter


def harris(image, opt=1):

    blockSize = 2
    apertureSize = 3
    k = 0.04

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
    dst = cv.cornerHarris(thresh, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    c=[]

    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > 120:
                #cv.circle(image, (j, i), 2, (0, 255, 0), 2)
                c.append([j,i])


    # output
    return image,c


src = cv.imread("testpic/pic19.jpg")

result ,c= harris(src)
#cv.imshow('result', result)

min1=min(c,key=itemgetter(0))[0]
min2=min(c,key=itemgetter(1))[1]
max1=max(c,key=itemgetter(0))[0]
max2=max(c,key=itemgetter(1))[1]
corner1=[min1,min2]
corner2=[max1,max2]
print(corner1,corner2)

cv.circle(src, (min1,min2), 2, (0, 255, 0), 2)
cv.circle(src, (max1,max2), 2, (0, 255, 0), 2)
cv.imshow('src', src)

cv.imwrite('result19.jpg', src)
cv.waitKey(0)