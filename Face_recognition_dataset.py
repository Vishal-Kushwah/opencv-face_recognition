import cv2 
import numpy as np
fc = cv2.CascadeClassifier("C:\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml")

def face_extractor(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = fc.detectMultiScale( gray, scaleFactor = 1.3, minNeighbors = 5)
	print(faces)
	if faces is ():
		return None

	for(x,y,w,h) in faces:
		cropped_faces = img[y:y+h,x:x+w]
	return cropped_faces

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
count = 0

while 1:
	ret,frame =cap.read()
	if face_extractor(frame) is not None:
		count+=1
		face = cv2.resize(face_extractor(frame),(200,200))
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		file_name_path = 'C:/Users/Vishal Kushwah/Desktop/faces/user'+str(count)+'.jpg'
		cv2.imwrite(file_name_path,face)

		cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
		cv2.imshow('Face Cropper',face)
	else:
		print("Face not Found!!!")
		pass
	if cv2.waitKey(1)==13 or count==50:
		break
cap.release()
cv2.destroyAllWindows()
print('Collecting Image Complete!!!')