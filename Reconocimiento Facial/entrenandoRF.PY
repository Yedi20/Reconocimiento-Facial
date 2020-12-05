import cv2
import os
import numpy as np

dataPath = 'C:/Users/SAE/Desktop/Reconocimiento Facial/data'    
#Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
#v2.imshow('image',image)
#cv2.waitKey(10)
    label = label + 1
print('labels= ',labels)
print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
#cv2.destroyAllWindows()
#python entrenandoRF.py
face_recognizer = cv2.face.EigenFaceRecognizer_create()

#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#entrenando el reconocedor de rosotos
print('Entrenando...')
face_recognizer.train(facesData, np.array(labels))

#almacenado los modelos obtendios
face_recognizer.write('modeloEingenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
print("Modelo almacenado")