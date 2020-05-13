import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
def getModel_CNN(train_images, train_labels, test_images, test_labels):
    image_vector_size = 28 * 28
    train_images = train_images.reshape(train_images.shape[0], image_vector_size)
    test_images = test_images.reshape(test_images.shape[0], image_vector_size)
    num_classes = 10
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    train_images = train_images.reshape(len(train_images), 28, 28, 1)
    test_images = test_images.reshape(len(test_images), 28, 28, 1)
    model = keras.Sequential([
        keras.layers.Conv2D(28, (1, 1), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(56, (1, 1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(56, (1, 1), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(56, activation='sigmoid'),
        keras.layers.Dense(10)
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit( train_images, train_labels, epochs=10, batch_size=64)
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=False)
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')
    return model


def getResult_CNN(model, img):
    test = [img]
    test = np.array(test)
    test = test.reshape(1, 28, 28, 1)

    pred = model.predict(test)
    #print("PP = ", test[0].shape, test[0],test[0].dtype)
    #print("pred=", pred)
    pred_index = np.argmax(pred[0])
    return int(pred_index)

def getModel_SVM(train_images, train_labels, test_images, test_labels):
    print("I am strat")
    model = svm.SVC(kernel='linear')
    model.fit(train_images,train_labels)
    test_pred = model.predict(test_images)
    print("Accuracy: ",metrics.accuracy_score(test_labels,test_pred))
    return model

def getResult_SVM(model, img):
    test = [img]
    test = np.array(test)
    test = test.reshape(1, 784)
    test = preprocessing.normalize(test, norm='l2')
    pred = model.predict(test)
    print("SVM pred=", pred)
    pred_index = np.argmax(pred[0])
    return int(pred_index)