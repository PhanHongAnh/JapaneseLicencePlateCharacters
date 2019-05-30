import cv2
import numpy as np
import pickle
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from datetime import datetime

from model import Model
from label import Label

class Recognize:
    def __init__(self, model):
        self.model = model
        self.number_clf = pickle.load(open("Digits_SVC.pickle", "rb"))
        self.hira_clf = pickle.load(open("Hiragana_SVC.pickle", "rb"))

    def recognize_all(self):
        chars = []
        final_rects, rects_kanji, rects_num1, rects_hira, rects_num2 = self.model.img_rects()
        # for r in rects_kanji:
        #     char = self.recognize_in_rect(r, 3)
        #     chars.append(char)
        for r in rects_num1:
            char = self.recognize_in_rect(r, 1)
            chars.append(char)
        for r in rects_hira:
            char = self.recognize_in_rect(r, 2)
            chars.append(char)
        for r in rects_num2:
            char = self.recognize_in_rect(r, 1)
            chars.append(char)
        return chars

    def hog_in_rect(self, rect):
        thresh = self.model.img_bin()
        x, y, w, h = rect
        cv2.rectangle(self.model.image,(x,y),(x+w,y+h),(0,255,0),3)
        roi = thresh[y:y+h,x:x+w]
        #roi = np.pad(roi,(3,3),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        im,thresh = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        roi_hog_fd = hog(thresh, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),block_norm="L2-Hys")
        #cv2.imshow("img", thresh)
        #cv2.waitKey(0)
        return roi_hog_fd

    def recognize_in_rect(self, rect, choice):
        char = ''
        roi_hog_fd = self.hog_in_rect(rect)
        if (choice == 1):
            nbr = self.number_clf.predict(np.array([roi_hog_fd], np.float32))
            char = str(int(nbr[0]))
        elif (choice == 2):
            nbr = self.hira_clf.predict(np.array([roi_hog_fd], np.float32))
            label = Label()
            char = label.search(int(nbr[0]))
        elif (choice == 3):
            nbr = self.number_clf.predict(np.array([roi_hog_fd], np.float32))
            label = Label()
            char = label.search(int(nbr[0]))
        return char

model = Model("bien1.png")
recognize = Recognize(model)
print(recognize.recognize_all())
#recognize.hog_in_rect(rects_num2[0])
