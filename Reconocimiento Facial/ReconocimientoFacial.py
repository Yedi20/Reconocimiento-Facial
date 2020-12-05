import cv2
import os

dataPath = 'C:/Users/SAE/Desktop/Reconocimiento Facial/data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

#Leyend0 modelo Engein faces
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('modeloEingenFace.xml')

#Leyend0 modelo FisherFace
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer.read('modeloFisherFace.xml')

#cap = cv2.VideoCapture('Img y Vid/YediTest0.mp4')
cap = cv2.VideoCapture('Imgenes y Video de prueba/bailarinaTest0.MP4')
#C:\Users\SAE\Desktop\Reconocimiento Facial\Imgenes y Video de prueba

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#cap = cv2. VideoCapture ( 0 )
while True:
	ret, frame = cap.read()
	if ret== False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame= gray.copy()

	faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150), interpolation= cv2.INTER_CUBIC)
		resulto = face_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(resulto),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
	
	# EigenFaces
	if resulto[1] < 700:
		cv2.putText(frame,'{}'.format(imagePaths[resulto[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
	else:
		cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
	''' 
	if result[1] < 5700:
		cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
	else:
		cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    '''
	cv2.imshow('frame', frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()

#python ReconocimientoFacial.py