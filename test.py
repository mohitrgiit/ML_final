import numpy as np
import cv2
from skimage import io

path = "train_data/imnet/"
prefix = 'Places2_test_'

blur = False
rescale = True
noise = False
data=[]
for i in range(5000,5010):
    pic_index = str(i+1).zfill(8)
    img = cv2.imread(path+prefix+pic_index+'.jpg')
    if blur:
        img = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=0.8, sigmaY=0.8)
    if rescale:
        img = cv2.resize(img, (128, 128))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    if noise:
        np.random.seed()
        ga_num = np.random.normal(scale=10, size=img.shape)
        img = img + ga_num
    data.append(img/255)

np.save('train_data/numeric_data/test_pic', io.concatenate_images(data))