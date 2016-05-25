from keras.models import Sequential
from keras.layers import Convolution2D
from keras.optimizers import Adam
import numpy as np
import cv2

#setup model
def set_model(structure):
    model = Sequential()
    if structure=="SR":
        model.add(Convolution2D(128, 9, 9, activation='relu', input_shape=(3, 32, 32)))
        model.add(Convolution2D(64, 5, 5, activation='relu'))
        model.add(Convolution2D(3, 5, 5))
        model.load_weights("weights/SR.h5")
        model.compile(loss='mse', optimizer=Adam())

    elif structure=="DB":
        model.add(Convolution2D(64, 9, 9, activation='relu', input_shape=(3, 32, 32)))
        model.add(Convolution2D(32, 5, 5, activation='relu'))
        model.add(Convolution2D(3, 5, 5))
        model.load_weights("weights/DB.h5")
        model.compile(loss='mse', optimizer=Adam())

    elif structure == "DN":
        model.add(Convolution2D(64, 7, 7, activation='relu', input_shape=(3, 32, 32)))
        model.add(Convolution2D(32, 7, 7, activation='relu'))
        model.add(Convolution2D(3, 5, 5))
        model.load_weights("weights/DN.h5")
        model.compile(loss='mse', optimizer=Adam())
    else:
        print("No model is loaded.")

    return model

def predict_batch(model, testdata_path, num):
    test = np.load(testdata_path)
    cv2.imwrite("before.png", test[num]*255)
    trans_test = test.transpose(0,3,1,2)
    output = model.predict(trans_test)
    visualize = output.transpose(0,2,3,1)
    cv2.imwrite("after.png", visualize[num]*255)

def predict(model, path, mode=None):
    img = cv2.imread(path).astype("float32")
    shape = np.array(img.shape)
    if mode=="SR":
        img = cv2.resize(img, tuple(np.flipud(shape[:2])*2), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("test_cubic.png", img)
    if mode=="DN":
        np.random.seed()
        ga_num = np.random.normal(scale=10, size=img.shape)
        img+=ga_num
        cv2.imwrite("test_noise.png", img)
    if mode=="DB":
        img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=0.8, sigmaY=0.8)
        cv2.imwrite("test_blur.png", img)
    shape = list(img.shape)
    shape.insert(0, -1)
    img = img.reshape(tuple(shape))
    img = (img.transpose(0, 3, 1, 2))/255
    out = model.predict(img)
    cv2.imwrite("test_after.png", out[0].transpose(1, 2, 0)*255)

model = set_model("SR")
predict(model, "test.jpg")
# predict_batch(model, "train_data/numeric_data/test_pic.npy", 4)