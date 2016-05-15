from sklearn.metrics import mean_squared_error
from skimage.util import crop
import cv2
import numpy as np

class psnr_evaluate:
    def __init__(self, standard, test):
        self.std = cv2.imread(standard)
        self.test = cv2.imread(test)

    def evaluate(self):
        if self.std.shape != self.test.shape:
            print("please crop images!")
        mse = mean_squared_error(self.test.flatten(), self.std.flatten())
        PSNR = 10*np.log10((255**2)/(mse/3))
        return PSNR

    def fit_crop(self, size):
        std_H = self.std.shape[0]
        std_W = self.std.shape[1]
        test_H = self.test.shape[0]
        test_W = self.test.shape[1]
        self.std = crop(self.std,
                        ((int((std_H-size)/2), int((std_H-size)/2)), (int((std_W-size)/2), int((std_W-size)/2)),(0,0)))
        self.test = crop(self.test,
                         ((int((test_H-size)/2), int((test_H-size)/2)), (int((test_W-size)/2), int((test_W-size)/2)),
                          (0,0)))

eval1 = psnr_evaluate("origin.jpg", "before.png")
eval1.fit_crop(128)
print(eval1.evaluate())
eval2 = psnr_evaluate("origin.jpg", "after.png")
eval2.fit_crop(128)
print(eval2.evaluate())