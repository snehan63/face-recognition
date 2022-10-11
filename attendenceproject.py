import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path= 'images attendence'
images =[]
classNames= []
mylist =os.listdir(path)
print(mylist)
for cls in mylist:
    curImg = cv2.imread(F'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)
def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
encodelistknown = findencodings(images)
print('encoding complete')
cap = cv2.VideoCapture(0)
while True:
    sucess, img =cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs,faceCurFrame)
    for encodeFace,faceloc in zip(encodesCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis = face_recognition.face_distance(encodelistknown,encodeFace)
        #print(facedis)
        matchIndex = np.argmin(facedis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        markAttendance(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

