import numpy as np
import cv2
from skimage import io

PIC_SIZE = 256
PIC_CUT = 8

def noise(data, sigma):
    for i in range(len(data)):
        np.random.seed()
        ga_num = np.random.normal(scale=sigma, size=data[i].shape)
        data[i] = data[i] + ga_num
    noise_arr = io.concatenate_images(data)/255
    print(noise_arr.shape)
    np.save("train_data/numeric_data/noise_data", noise_arr)

def blur(data, with_resize = False):
    for i in range(len(data)):
        data[i] = cv2.GaussianBlur(data[i], ksize=(0,0), sigmaX=0.8, sigmaY=0.8)
        if with_resize == True:
            data[i] = cv2.resize(data[i], (16,16))
            data[i] = cv2.resize(data[i], (32,32), interpolation = cv2.INTER_CUBIC)
    blur_arr = io.concatenate_images(data)/255
    print(blur_arr.shape)
    np.save("train_data/numeric_data/blur_data", blur_arr)

def lowres(data):
    for i in range(len(data)):
        data[i] = cv2.resize(data[i], (16, 16))
        data[i] = cv2.resize(data[i], (32, 32), interpolation=cv2.INTER_CUBIC)
    lowres_arr = io.concatenate_images(data)/255
    print(lowres_arr.shape)
    np.save("train_data/numeric_data/lowres_data", lowres_arr)

#load image
path = "train_data/imnet/"
prefix = 'Places2_test_'
all_pics = []
for i in range(1000):
    pic_index = str(i+1).zfill(8)
    img = cv2.imread(path+prefix+pic_index+'.jpg')
    all_pics.append(img)
print(all_pics[0].shape)

#crop image as certain size
data = []
for i in range(len(all_pics)):
    for j in range(PIC_CUT):
        for k in range(PIC_CUT):
            data.append(all_pics[i][(PIC_SIZE//PIC_CUT)*j:(PIC_SIZE//PIC_CUT)*(j+1),
                        (PIC_SIZE//PIC_CUT)*k:(PIC_SIZE//PIC_CUT)*(k+1),:])
del all_pics

#Target data
highres_arr = io.concatenate_images(data)/255
print(highres_arr.shape)
np.save("train_data/numeric_data/highres_data", highres_arr)
del highres_arr

#Training data
blur(data)
noise(data, 10)
lowres(data)