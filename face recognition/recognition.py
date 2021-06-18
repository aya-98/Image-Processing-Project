import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition


 
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)

#print(myList)
for cl in myList:

    curImg = face_recognition.load_image_file(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])




# func to extract features from image and encode the face
def findEncodings(images):

    encodeList = []
    for img in images:

        
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



# function to record recognized images in csv file   
 
def markAttendance(name ):

    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}' +'\n')
 

encodeListKnown = findEncodings(images)

testdir= "test"
 
for filename in os.listdir(testdir):

    img = face_recognition.load_image_file(f'{testdir}/{filename}')

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted_image = 255 - gray_image

    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

    inverted_blurred = 255 - blurred

    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)


 
    facesCurFrame = face_recognition.face_locations(img)

    encodesCurFrame = face_recognition.face_encodings(img,facesCurFrame)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1,x2,y2,x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2+22),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+10,y2+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

            cv2.rectangle( pencil_sketch,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle( pencil_sketch,(x1,y2+22),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText( pencil_sketch,name,(x1+10,y2+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

           
            
            
            markAttendance(name )

        else:
            name = 'Unknown'
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1,x2,y2,x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2),(x2,y2+22),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+10,y2+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

            cv2.rectangle( pencil_sketch,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle( pencil_sketch,(x1,y2+22),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText( pencil_sketch,name,(x1+10,y2+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)


    cv2.imshow('original',img)

    cv2.imshow('sketched',pencil_sketch)
    
    cv2.waitKey(5000) 




