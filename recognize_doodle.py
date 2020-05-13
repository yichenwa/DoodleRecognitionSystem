import cv2
import os
from ml_recognize_doodle import *
from sklearn.preprocessing import MinMaxScaler
class_name = ['apple', 'bicycle', 'bird', 'book', 'car',
            'cat', 'clock', 'face', 'key', 'sun']

def load_images_from_album(path):
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            cv2.imshow('',img)
            cv2.waitKey(0)


def show_mostSimilar_image(l,images):
    result_index = l.index(max(l))
    print("The similarity is : ", max(l))
    result_image = images[result_index]
    return result_image

def input_img(m):
    for i in range(0,len(m)):
        for j in range(0,len(m[0])):
            m[i,j] = 255 - m[i,j]
    return m

def compareImages(img, model):
    print("Start")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    cv2.imshow('28*28',img)
    img = np.array(img)
    #print(img)
    #print(img.dtype)
    img = img.astype('float32')
    #print("+++++++++")
    #print(img)
    #print(img.dtype)
    img = input_img(img)
    img = img / 255.0
    scaler = MinMaxScaler()
    scaler.fit(img)
    normalized = scaler.transform(img)
    img = scaler.inverse_transform(normalized)
    #img = preprocessing.scale(img)
    #img = preprocessing.normalize(img, norm='l2')
    #print(img)
    cnn = getResult_CNN(model, img)
    print("I think this is :", class_name[cnn])
    result_path = 'album/'+class_name[cnn]
    load_images_from_album(result_path)
