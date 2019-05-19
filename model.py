import cv2
import numpy as np
from PIL import Image
from datetime import datetime

class Model:
    def __init__(self, img_link):
        ori_img = cv2.imread(img_link)
        self.image = ori_img

    def img_bin(self):
        im_gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(image_blurred,kernel,iterations = 1)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

        im,thresh = cv2.threshold(opening,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh

    def img_rects(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        erosion = cv2.erode(self.img_bin(), kernel, iterations = 1)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilation = cv2.dilate(erosion, kernel1, iterations = 1)

        contours,hierachy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]

        height = self.image.shape[0]
        width = self.image.shape[1]
        rects_kanji = []
        rects_num1 = []
        rects_hira = []
        rects_num2 = []

        for r in rects:
            (x,y,w,h) = r
            if (w <= 25 or h <= 25):
                rects.remove(r)

        for r in rects:
            (x,y,w,h) = r
            if y < height/3:
                if (x > width/5 and x < width*0.4):
                    rects_kanji.append(r)
                elif (x > width*0.4 and x < width - width/4):
                    rects_num1.append(r)
            else:
                if (x < width/5):
                    rects_hira.append(r)
                else:
                    rects_num2.append(r)
        rects_kanji.sort()
        rects_num1.sort()
        rects_hira.sort() #phai gop rects_hira vao
        rects_num2.sort()
        for r in rects_num2:
            (x,y,w,h) = r
            if (h < 40):
                rects_num2.remove(r)

        final_rects = rects_kanji + rects_num1 + rects_hira +rects_num2
        return final_rects, rects_kanji, rects_num1, rects_hira, rects_num2

    def print_rects(self):
        rects, rects_kanji, rects_num1, rects_hira, rects_num2 = self.img_rects()
        for r in rects:
            x,y,w,h = r
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(0,255,0),2)
        filename = self.file_name()
        cv2.imwrite(filename,self.image)
        rects_img = Image.open(filename)
        return rects_img

    def file_name(self):
        now = datetime.now()
        date_time = now.strftime("%m%d%Y%H%M%S")
        fn = "Recognized/" + date_time + ".png"
        return fn
