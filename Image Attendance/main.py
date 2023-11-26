import cv2
import face_recognition

img1 = face_recognition.load_image_file('image/babar.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('image/babar test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


facelocation = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(facelocation[3],facelocation[0]), (facelocation[1],facelocation[2]), (255,0,255),2)


facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]), (facelocTest[1],facelocTest[2]), (255,0,255),2)

results = face_recognition.compare_faces([encode1],encodeTest)
faceDis = face_recognition.face_distance([encode1],encodeTest)
print(results, faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('noman',img1)
cv2.imshow('test',imgTest)
cv2.waitKey(0)
