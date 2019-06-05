import cv2
import numpy as np
import pickle
from skimage.feature import hog
from skimage import data, exposure
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class DataSet:
    def __init__(self):
        X_train = np.load('../sample/kmnist/k49-train-imgs.npz')['arr_0']
        y_train = np.load('../sample/kmnist/k49-train-labels.npz')['arr_0']
        X_test = np.load('../sample/kmnist/k49-test-imgs.npz')['arr_0']
        y_test = np.load('../sample/kmnist/k49-test-labels.npz')['arr_0']

        X_train_bin = self.img_bin(X_train)
        X_test_bin = self.img_bin(X_test)

        self.X_train, self.y_train = self.devidedData(X_train_bin, y_train, 1000)
        self.X_test, self.y_test = self.devidedData(X_test_bin, y_test, 260)

    def devidedData(self, X, y, number):
        classes = np.zeros((41,number))
        k = 0
        for i in range(0, 47):
            if (i != 3 and i !=4 and i != 11 and i != 28 and i != 44 and i != 45):
                ii = np.where(y == i)[0]
                classes[k] = ii[0:number]
                k = k+1
        classes = classes.reshape(41*number)
        X_l = []
        y_l = []
        for i in classes:
            X_l.append(X[int(i)])
            y_l.append(y[int(i)])
        X_rs = np.array(X_l).reshape(41*number, 28, 28)
        y_rs = np.array(y_l).reshape(41*number)
        return X_rs, y_rs

    def img_bin(self, array):
        bin_arr = []
        for img in array:
            im,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            bin_arr.append(thresh)
        return bin_arr

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
        C = 0.24700244468265697
        clf = SVC(C=C, kernel='linear')
        clf.fit(self.X_train_feature(),self.y_train)
        with open('Hiragana_SVC'+'.pickle', mode='wb') as f:
            pickle.dump(clf,f)
        return clf

    def print_accuracy(self):
        clf = self.SVM()
        y_pred = clf.predict(self.X_train_feature())
        acc_train=accuracy_score(self.y_train, y_pred)
        print ('accuracy on training data(not class average)',acc_train)

        y_pred_test = clf.predict(self.X_test_feature())
        acc_test=accuracy_score(self.y_test, y_pred_test)
        print ('accuracy on test data(not class average)',acc_test)

        print(classification_report(self.y_train, y_pred))
        print(classification_report(self.y_test, y_pred_test))

dataset = DataSet()
print(dataset.y_train.dtype)
