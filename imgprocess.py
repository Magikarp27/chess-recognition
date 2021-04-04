import cv2
from cv2 import circle
import numpy as np
img = cv2.imread('pic5_2.jpg')#读取图片
b=[]
size=512#图像大小
f=open('pictest2.txt','r')
for line in f.readlines():
     a = line.replace("\n", "")
     b.append(a.split(" "))
b.remove(['4'])
for i in range(0,4):
     b[i][0]=int(float(b[i][0])*size)
     b[i][1]=int(float(b[i][1])*size)
print(b)
distance=(abs(b[0][0]-b[2][0]))/18
print(distance)
point=np.zeros((19,19,2),dtype=np.int)
for i in range(0,19):
     for j in range(0, 19):
          point[i][j]=([int(b[0][0]+distance*i),int(b[0][1]+distance*j)])


def blaorwhi(img,point):
     sum=0
     flag=0
     global thirddis

     for i in range(-5,5):
          for j in range(-5,5):
               if (img[point[0]+i, point[1]+j, 0] < 50):#判断区域内是否有黑点
                    flag = 1
               sum=sum+(img[point[0]+i,point[1]+j,0])
     color=sum/100

     if(color<125):
          return(1)
     elif(flag ):
          return(0)
     else:
          return(2)
chessboard=np.zeros((19,19),dtype=np.int)
'''
for i in range(0,19):
     for j in range(0, 19):
          print(img[point[i][j][0],point[i][j][1],0])
'''
for i in range(0,19):
     for j in range(0, 19):
          chessboard[i][j]=blaorwhi(img,point[i][j])
print(chessboard)


for i in range(0,4):
     circle(img,(b[i][0],b[i][1]),3,(0,0,255))
for i in range(0,19):
     for j in range(0, 19):
          if(chessboard[i][j]==1):
               circle(img,(point[i][j][0],point[i][j][1]),2,(0,0,255))
          if(chessboard[i][j]==2):
               circle(img, (point[i][j][0], point[i][j][1]), 2, (255, 0, 0))
'''
for i in range(0,19):
     for j in range(0, 19):
          circle(img,(point[i][j][0],point[i][j][1]),2,(0,0,255))
'''
cv2.imshow('img', img)
cv2.waitKey(0)
