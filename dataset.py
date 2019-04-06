import pickle
import cv2
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class DataSet:
    def __init__(self):
        X_train = np.load('../sample/kmnist/k49-train-imgs.npz')['arr_0']
        y_train = np.load('../sample/kmnist/k49-train-labels.npz')['arr_0']
        X_test = np.load('../sample/kmnist/k49-test-imgs.npz')['arr_0']
        y_test = np.load('../sample/kmnist/k49-test-labels.npz')['arr_0']

        X_train_bin = self.img_bin(X_train)
        X_test_bin = self.img_bin(X_test)

        self.X_train, self.y_train = self.devidedData(X_train_bin, y_train)
        self.X_test = X_test_bin
        self.y_test = y_test

    def devidedData(self, X, y):
        number = 100
        classes = np.zeros((49,number))
        for i in range(0, 49):
            ii = np.where(y == i)[0]
            classes[i] = ii[0:number]
        classes = classes.reshape(49*number)
        X_l = []
        y_l = []
        for i in classes:
            X_l.append(X[int(i)])
            y_l.append(y[int(i)])
        X_rs = np.array(X_l).reshape(49*number, 28, 28)
        y_rs = np.array(y_l).reshape(49*number)
        return X_rs, y_rs

    def img_bin(self, array):
        bin_arr = []
        for img in array:
            image_blurred = cv2.GaussianBlur(img, (3, 3), 0)
            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(image_blurred,kernel,iterations = 1)

            im,thresh = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
        C = 4.152705171689228
        gamma = 0.006783091541660457
        clf = SVC(C=C, gamma = gamma, kernel='rbf')
        clf.fit(self.X_train_feature(),self.y_train)
        return clf

    def print_accuracy(self):
        clf = self.SVM()
        y_pred = clf.predict(self.X_train_feature())
        acc_train=accuracy_score(self.y_train, y_pred)
        print ('accuracy on training data(not class average)',acc_train)

        y_pred_test = clf.predict(self.X_test_feature())
        acc_test=accuracy_score(self.y_test, y_pred_test)
        print ('accuracy on test data(not class average)',acc_test)

        accuracy_train = 0
        accuracy_test = 0
        for i in range(0,49):
            y_index_train = np.where(self.y_train==i)
            y_index_test = np.where(self.y_test==i)
            acc_train=accuracy_score(self.y_train[y_index_train], y_pred[y_index_train])
            acc_test=accuracy_score(self.y_test[y_index_test], y_pred_test[y_index_test])
            accuracy_train+=acc_train/49
            accuracy_test+=acc_test/49
        print ('accuracy on training data(class averaged)',accuracy_train)
        print ('accuracy on test data(class averaged)',accuracy_test)

dataset = DataSet()
dataset.print_accuracy()
"""print(dataset.y_train[0])
image = dataset.X_train[0]
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()"""
