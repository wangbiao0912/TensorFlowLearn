#coding:utf-8
import cv2
import sys
import dlib
from PIL import Image
from keras.models import load_model
import numpy as np
import chineseText

detector = dlib.get_frontal_face_detector()  #使用默认的人类识别器模型


img = cv2.imread("img/gather.png")
face_classifier = cv2.CascadeClassifier(
    "/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/Deno/人脸识别/神经网络/获取照片/haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))

gender_classifier = load_model(
    "./simple_CNN.81-0.96.hdf5")
gender_labels = {0: '女', 1: '男'}
color = (255, 255, 255)



def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)#该方法是写入打开时视频框的名称
    # 捕捉摄像头
    cap = cv2.VideoCapture(camera_idx)#camera_idx 的参数是0代表是打开笔记本的内置摄像头，也可以写上自己录制的视频路径
    while cap.isOpened():#判断摄像头是否打开，打开的话就是返回的是True
        ok, frame = cap.read()#读取一帧数据，该方法返回两个参数，第一个参数是布尔值，frame就是每一帧的图像，是个三维矩阵，当输入的是一个是视频文件，读完ok==flase
        if not ok:#如果读取帧数不是正确的则ok就是Flase则该语句就会执行
            break
        # 显示图像

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        for face in dets:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, 0)
            face = face / 255.0
            gender_label_arg = np.argmax(gender_classifier.predict(face))
            gender = gender_labels[gender_label_arg]
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)

        cv2.putText(frame, gender, x + h, y, color, 30)

        cv2.imshow(window_name, frame)#该方法就是现实该图像
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):#q退出视频
            break
            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    CatchUsbVideo("娱乐", 0)