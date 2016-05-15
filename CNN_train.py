from keras.models import Sequential
from keras.layers import Convolution2D
from keras.optimizers import Adam
from skimage.util import crop
from skimage import io
import numpy as np

#load data
##some setup for target set, including crop the target data as target size
target_set = np.load('train_data/numeric_data/highres_data.npy')
crop_target_list = []
for i in range(target_set.shape[0]):
    crop_target_list.append(crop(target_set[i], ((8,8),(8,8),(0,0))))
crop_target = io.concatenate_images(crop_target_list)
y_train = crop_target.transpose(0,3,1,2)
print(y_train.shape)

#setup model
model1 = Sequential()
model2 = Sequential()
model3 = Sequential()
def trainDB(model):
    train_set = np.load('train_data/numeric_data/blur_data.npy')
    X_train = train_set.transpose(0, 3, 1, 2)
    print(X_train.shape)
    model.add(Convolution2D(64, 9, 9, activation='relu', input_shape=(3, 32, 32), init='he_normal'))
    model.add(Convolution2D(32, 5, 5, activation='relu', init='he_normal'))
    model.add(Convolution2D(3, 5, 5, init='he_normal'))
    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=16, validation_split=0.1, shuffle=True, nb_epoch=100, verbose=2)
    model.save_weights('weights/DB.h5', overwrite=True)

def trainSR(model):
    train_set = np.load('train_data/numeric_data/lowres_data.npy')
    X_train = train_set.transpose(0, 3, 1, 2)
    print(X_train.shape)
    model.add(Convolution2D(128, 9, 9, activation='relu', input_shape=(3, 32, 32), init='he_normal'))
    model.add(Convolution2D(64, 5, 5, activation='relu', init='he_normal'))
    model.add(Convolution2D(3, 5, 5, init='he_normal'))
    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=16, validation_split=0.1, shuffle=True, nb_epoch=150, verbose=2)
    model.save_weights('weights/SR.h5', overwrite=True)

def trainDN(model):
    train_set = np.load('train_data/numeric_data/noise_data.npy')
    X_train = train_set.transpose(0, 3, 1, 2)
    print(X_train.shape)
    model.add(Convolution2D(64, 7, 7, activation='relu', input_shape=(3, 32, 32), init='he_normal'))
    model.add(Convolution2D(32, 7, 7, activation='relu', init='he_normal'))
    model.add(Convolution2D(3, 5, 5, init='he_normal'))
    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=16, validation_split=0.1, shuffle=True, nb_epoch=100, verbose=2)
    model.save_weights('weights/DN.h5', overwrite=True)

# trainDN(model1)
# trainDB(model2)
trainSR(model3)