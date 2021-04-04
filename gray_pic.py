import cv2
import os
import numpy as np

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 512
    height_new = 512
    img_new = cv2.resize(image, (width_new,height_new))
    return img_new



'''
for root, dirs, files in os.walk('chessboard'):
    files.remove('.DS_Store')
    for file in files:
        print(file)
        img=cv2.imread('chessboard/'+file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        img_new=img_resize(thresh)
        cv2.imwrite('newchessboard/'+file,img_new)


img=cv2.imread('test.png')
img=img_resize(img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('b',gray)
ret,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)#二值化
cv2.imshow('a',thresh)
cv2.waitKey(0)


'''
for root, dirs, files in os.walk('/Users/pengboguo/Desktop/未命名文件夹/'):
    #files.remove('.DS_Store')
    for file in files:
        img=cv2.imread('/Users/pengboguo/Desktop/未命名文件夹/' + file)
        img=img_resize(img)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        cv2.imwrite('/Users/pengboguo/Desktop/未命名文件夹/' + file,img)

