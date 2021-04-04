from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os

from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

trainpath='color_/'
testpath='testpic/'
FILE_PATH='savetrain/'
batch_size=10

def __data_label__(path):
     datalist = []
     labellist = []
     size = 512  # 图像大小
     for root, dirs, files in os.walk(path):
          files.remove('.DS_Store')
          for file in files:
               img=cv2.imread(path + file) # 读取图片
               datalist.append(img)
               b = []
               file = file.split()
               file = file[0]
               file=file.replace('.jpg','')
               f = open('pictext/'+file+'.txt', 'r')
               for line in f.readlines():
                    a = line.replace("\n", "")
                    b.append(a.split(" "))
               b.remove(['4'])

               c=[]
               for i in range(0, 4):

                    c.append(int(float(b[i][0]) * size))
                    c.append( int(float(b[i][1]) * size))

               labellist.append(c)
     label = np.array(labellist)
     img_data = np.array(datalist)
     img_data = img_data.astype('float32')
     return img_data, label


def __chessre__():
     model = Sequential()  # 178*178*3
     model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 3)))
     model.add(Activation('relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     '''
     model.add(Conv2D(32, (3, 3)))
     model.add(Activation('relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))

     model.add(Conv2D(64, (3, 3)))
     model.add(Activation('relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     '''
     model.add(Flatten())
     model.add(Dense(64))
     model.add(Activation('relu'))
     model.add(Dropout(0.5))
     model.add(Dense(8))
     return model

def train(model, testdata, testlabel, traindata, trainlabel):

     # model.compile里的参数loss就是损失函数(目标函数)
     model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(), optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])

     # 开始训练， show_accuracy在每次迭代后显示正确率 。  batch_size是每次带入训练的样本数目 ， nb_epoch 是迭代次数
     model.fit(traindata, trainlabel, batch_size=10, epochs=20,
               validation_data=(testdata, testlabel))
     # 设置测试评估参数，用测试集样本
     model.evaluate(testdata, testlabel, batch_size=2, verbose=1)

def save(model, file_path=FILE_PATH):
     print('Model Saved.')
     model.save_weights(file_path)

def predict(model,image):

     img = image.resize((1, 512, 512, 3))
     img = image.astype('float32')
     img /= 255
     #归一化
     result = model.predict(img)
     result = result*1000+10

     print(result)
     return result




if __name__ == '__main__':


     model = __chessre__()
     '''
     testdata, testlabel = __data_label__(testpath)
     traindata, trainlabel = __data_label__(trainpath)
     train(model,testdata, testlabel, traindata, trainlabel)
     test_loss, test_acc = model.evaluate(testdata, testlabel)

     model.save_weights('saveweight.h5')
     print('test_acc:', test_acc)
     print('test_loss:', test_loss)

'''
     #model.load_weights('savemodell.h5')
     img = []
     path = "pic5_2.jpg"
     image = load_img(path)
     img.append(img_to_array(image))
     img_data = np.array(img)
     rects = predict(model,img_data)
     print(rects)
