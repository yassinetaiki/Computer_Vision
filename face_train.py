import cv2 as cv 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import random

#afficher liste des element qui se trouve dans train
lien=r"C:\Users\Admin\Desktop\ComputerVision\OpenCV\Face_Recognisation\Faces\train"
lien_2=r"C:\Users\Admin\Desktop\ComputerVision\OpenCV\Face_Recognisation\Faces\val"
#Liste Label train
list_label_train=[]
for dossier in os.listdir(lien):
    list_label_train.append(dossier)

#Liste Label test
list_label_test=[]
for dossier in os.listdir(lien_2):
    list_label_test.append(dossier)


features=[]
labels=[]

def recadrer_image(img):
    img_NB=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    classifeur=cv.CascadeClassifier('hare_face.xml')
    rect_visage=classifeur.detectMultiScale(img_NB,scaleFactor=1.1,minNeighbors=1)

    for x,y,w,h in rect_visage:
        visage_recadrer=img_NB[y:y+h,x:x+w]

    return visage_recadrer

def create_train(lien,list_label):
    for col in list_label:
        path=os.path.join(lien,col) 
        label=list_label.index(col)
        
        for img in os.listdir(path):
            image_path=os.path.join(path,img)
            
            mon_image=cv.imread(image_path)
            img_NiveauGris=cv.cvtColor(mon_image,cv.COLOR_BGR2GRAY)
            classifier=cv.CascadeClassifier('hare_face.xml')
            rectangle_image=classifier.detectMultiScale(img_NiveauGris,scaleFactor=1.1,minNeighbors=1)

            for x,y,w,h in rectangle_image:
                visage_recadrer=img_NiveauGris[y:y+h,x:x+w]
                features.append(visage_recadrer)
                labels.append(label)
    return features,labels ,visage_recadrer 

test_labels=[]
test_features=[]



#training 

features,labels,visage_recadrer_train=create_train(lien,list_label_train)
featurestest,labelstest,visage_recadrer_test=create_train(lien_2,list_label_test)
         
variables_train=np.array(features,dtype="object")
categories_train=np.array(labels)
variables_test=np.array(featurestest,dtype="object")
categories_test=np.array(labelstest)

classifier_visage=cv.face.LBPHFaceRecognizer_create()
classifier_visage.train(variables_train,categories_train)


prediction,confidence=classifier_visage.predict(visage_recadrer_test)
print("label predite",list_label_test[prediction],"avec une confidence de ",confidence)
classifier_visage.save('face_trained.yml')
np.save('features.npy',variables_train)
np.save('labels.npy',categories_test)

#prediction

images=[]
for col in os.listdir(lien_2):
    path=os.path.join(lien_2,col)
    for img in os.listdir(path):
        path_img=os.path.join(path,img)
        image=cv.imread(path_img)
        images.append(image)

random.shuffle(images)
plt.figure(figsize=(20,10))
for col in range(1,8):
    visage_recadrer=recadrer_image(images[col])
    prediction,confidence=classifier_visage.predict(visage_recadrer)
    plt.subplot(2,4,col)
    images[col]=cv.cvtColor(images[col],cv.COLOR_BGR2RGB)
    plt.imshow(images[col])
    plt.xlabel(" ")
    plt.title(list_label_test[prediction])
    plt.tight_layout()
plt.show()

cv.waitKey(0)

