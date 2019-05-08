import cv2
from PIL import Image, ImageTk
from tkinter import Frame, Tk, BOTH, Text, Label, Button, messagebox
from tkinter.filedialog import Open

from dataset import DataSet
from model import Model
from predict import Predict

class GUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("Japanese Licence Plate's Characters Recognize")
        self.pack(fill=BOTH, expand=1)

        f1 = Frame(self, bd=2, relief="ridge", width=600, height=500)
        f2 = Frame(self, width=600, height=100)
        f3 = Frame(self, bd=2, relief="ridge", width=300, height=500)
        f4 = Frame(self, width=300, height=100)

        f1.grid(row=0, column=0, sticky="nsew")
        f2.grid(row=1, column=0, sticky="nsew")
        f3.grid(row=0, column=1, sticky="nsew")
        f4.grid(row=1, column=1, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        input_img = Label(f1, text="Licence Plate Image", font=("Courier", 26))
        input_img.place(x=100, y=20)

        options = Label(f2, text="Options", font=("Courier", 20))
        options.place(x=100,y=10)

        open_bt = Button(f2, text="Open Image", bd=2, relief="raised",
            command=self.onOpen)
        open_bt.place(x=10, y=50)

        recognize_bt = Button(f2, text="Recognize", bd=2, relief="raised",
            command=self.onRecognize)
        recognize_bt.place(x=150, y=50)

        totext = Label(f3, text="Recognized\n Characters",
            font=("Courier", 20), justify="center")
        totext.place(height=100, width=296)

        self.text_frame = Text(f3, bd=2)
        self.text_frame.place(x=0,y=100)

        self.img_dir = ''
        self.pack()

    def onOpen(self):
        ftypes = [('Image files', '*.jpg *.png *.jpeg *.JPG'), ('All files', '*')]
        dlg = Open(self, filetypes = ftypes)
        self.img_dir = dlg.show()
        if self.img_dir != '':
            self.image = self.openImage(self.img_dir)
            label = Label(self, image=self.image)
            label.grid(row=0, column=0)

            self.text_frame.delete(1.0,'end')

    def openImage(self, filename):
        img = self.resize(Image.open(filename))
        return img

    def resize(self, img):
        width, height = img.size
        ratio = height/width
        if (width > 590):
            resized_img = img.resize((590, int(590*ratio)), Image.ANTIALIAS)
        else:
            resized_img = img

        width, height = resized_img.size
        if (height > 400):
            resized_img = resized_img.resize((int(400/ratio), 400), Image.ANTIALIAS)
        imageTk = ImageTk.PhotoImage(resized_img)
        return imageTk

    def onRecognize(self):
        if self.img_dir != '':
            model = Model(self.img_dir)
            dataset = DataSet()
            predict = Predict(model, dataset)

            self.rects_img = self.resize(model.print_rects())
            label = Label(self, image=self.rects_img)
            label.grid(row=0, column=0)

            chars, romanjis = predict.predict_in_rect()
            for romanji in romanjis:
                self.text_frame.insert("end", chars)
        else:
            messagebox.showwarning("Warning","Please input image")

window = Tk()
window.geometry("900x600+30+30")
gui = GUI(window)
window.mainloop()
