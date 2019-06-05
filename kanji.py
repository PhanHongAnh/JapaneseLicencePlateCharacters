import cv2
import numpy as np
import pickle
from skimage.feature import hog
from skimage import data, exposure
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class Kanji:
    def __init__(self):
        X_train = np.ndarray((150, 28, 28), dtype = np.uint8)
        X_test = np.ndarray((50, 28, 28), dtype = np.uint8)
        y_train = np.zeros((15), dtype = np.uint8)
        y_test = np.zeros((5), dtype = np.uint8)

        for i in range(1,10):
            y = np.full((15), i, dtype = np.uint8)
            y_train = np.append(y_train, y)
        self.y_train = y_train

        for i in range(1,10):
            y = np.full((5), i, dtype = np.uint8)
            y_test = np.append(y_test, y)
        self.y_test = y_test

        fn = ''
        k = 0
        for i in range(0,10):
            for j in range(1,16):
                fn = str(i) + '_' + str(j)
                x = self.img_bin(fn)
                X_train[k] = x
                k = k+1;
        self.X_train = X_train

        k = 0
        for i in range(0,10):
            for j in range(1,6):
                fn = str(i) + '_' + str(j)
                x = self.img_bin(fn)
                X_test[k] = x
                k = k+1;
        self.X_test = X_test

    def img_bin(self, fn):
        img = cv2.imread("../sample/kmnist/Train/" + fn + ".png")
        im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)
        im,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh

    def X_train_feature(self):
        X_train_feature = []
        for i in range(len(self.X_train)):
            feature = hog(self.X_train[i],orientations=9,pixels_per_cell=(4,4),cells_per_block=(2,2),block_norm="L2-Hys")
            X_train_feature.append(feature)
        X_train_feature = np.array(X_train_feature,dtype = np.float32)
        return X_train_feature

    def X_test_feature(self):
        X_test_feature = []
        for i in range(len(self.X_test)):
            feature = hog(self.X_test[i],orientations=9,pixels_per_cell=(4,4),cells_per_block=(2,2),block_norm="L2-Hys")
            X_test_feature.append(feature)
        X_test_feature = np.array(X_test_feature,dtype=np.float32)
        return X_test_feature

    def SVM(self):
        C = 6.19231438919562
        clf = SVC(C=C, kernel='linear')
        clf.fit(self.X_train_feature(),self.y_train)
        with open('Kanji_SVC'+'.pickle', mode='wb') as f:
            pickle.dump(clf,f)
        return clf

    def print_accuracy(self):
        clf = self.SVM()
        y_pred = clf.predict(self.X_train_feature())
        acc_train=accuracy_score(self.y_train, y_pred)
        print ('accuracy on training data',acc_train)

        y_pred_test = clf.predict(self.X_test_feature())
        acc_test=accuracy_score(self.y_test, y_pred_test)
        print ('accuracy on test data',acc_test)

        print(classification_report(self.y_train, y_pred))
        print(classification_report(self.y_test, y_pred_test))

kanji = Kanji()
kanji.print_accuracy()
