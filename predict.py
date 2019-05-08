import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from datetime import datetime

from dataset import DataSet
from model import Model
from label import Label

class Predict:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict_in_rect(self):
        image = self.model.image
        thresh = self.model.img_bin()
        rects = self.model.img_rects()
        clf = self.dataset.SVM()
        chars = []
        romanjis = []

        for r in rects:
            (x,y,w,h) = r
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
            roi = thresh[y:y+h,x:x+w]
            #roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),block_norm="L2-Hys")
            nbr = clf.predict(np.array([roi_hog_fd], np.float32))
            label = Label()
            char, romanji = label.search(int(nbr[0]))
            chars.append(char)
            romanjis.append(romanji)
        return chars, romanjis

"""model = Model("../test1.jpg")
dataset = DataSet()
predict = Predict(model, dataset)
predict_img, chars, romanjis = predict.predict_in_rect()
for char in chars:
    print(char)"""
