import cv2
import numpy as np

class Model:
    def __init__(self, img_link):
        ori_img = cv2.imread(img_link)
        self.image = ori_img

    def img_bin(self):
        im_gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(image_blurred,kernel,iterations = 1)

        im,thresh = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh

    def img_rects(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        erosion = cv2.erode(self.img_bin(), kernel, iterations = 1)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        dilation = cv2.dilate(erosion, kernel1, iterations = 1)

        contours,hierachy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        return rects

    def print_rects(self):
        for r in self.img_rects():
            x,y,w,h = r
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("image",self.image)
        cv2.waitKey(0)
