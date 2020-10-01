import os
import tkinter as tk  # for graphical interface
from tkinter import simpledialog

import PIL.Image
import PIL.ImageTk
from Computer_Vision import camera
import cv2 as cv
from Computer_Vision import model


class App:

    def __init__(self, window=tk.Tk(), window_title="Computer Vision- Camera Classifier"):

        self.window = window
        self.window_title = window_title

        self.counters = [0, 0]  # for the img indexing

        self.model = model.Model()

        self.auto_predict = False

        self.camera = camera.Camera()

        self.init_gui()

        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:",
                                                    parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:",
                                                    parent=self.window)

        self.head_label = tk.Label(self.window, text="Computer Vision- Camera Classifier:", font=("Helvetica", 18))
        self.head_label.grid(row=0, column=1)

        self.class_label = tk.Label(self.window, text="-- blank --", font=("Helvetica", 16))
        self.class_label.grid(row=1, column=1)

        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.grid(row=2, column=1)

        self.error_label = tk.Label(self.window, text="", font=("Helvetica", 16), foreground='red')
        self.error_label.grid(row=3, column=1)

        self.btn_class_one = tk.Button(self.window, text=f'{self.classname_one} - quantity: 0', width=30,
                                       font=("Helvetica", 12), command=lambda: self.save_for_class(1))
        self.btn_class_one.grid(row=5, column=0)

        self.btn_class_two = tk.Button(self.window, text=f'{self.classname_two} - quantity: 0', width=30,
                                       font=("Helvetica", 12), command=lambda: self.save_for_class(2))
        self.btn_class_two.grid(row=5, column=5)

        self.btn_toggleauto = tk.Button(self.window, text="Toggle Auto Prediction", width=60, font=("Helvetica", 12),
                                        command=self.auto_predict_toggle)
        self.btn_toggleauto.grid(row=4, column=1)

        self.btn_train = tk.Button(self.window, text="Train Model", width=60, font=("Helvetica", 12),
                                   command=lambda: self.model.train_model() if self.counters[0] >= 50 and self.counters[
                                       1] >= 50 else False)
        self.btn_train.grid(row=5, column=1)

        self.btn_predict = tk.Button(self.window, text="Predcit", width=60, font=("Helvetica", 12),
                                     command=self.predict)
        self.btn_predict.grid(row=6, column=1)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("img"):
            os.mkdir("img")
        if not os.path.exists("img/1"):
            os.mkdir("img/1")
        if not os.path.exists("img/2"):
            os.mkdir("img/2")

        # pre-processing. reduce the amount of data in order to reduce the training time.
        img_path = f'img/{class_num}/frame{self.counters[class_num - 1]}.jpg'
        # cv.imwrite(img_path, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        cv.imwrite(img_path, frame)
        img = PIL.Image.open(img_path)
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save(img_path)

        self.counters[class_num - 1] += 1
        # update the quantity
        self.btn_class_one.config(text=f'{self.classname_one} - quantity:{self.counters[0]}')
        self.btn_class_two.config(text=f'{self.classname_two} - quantity:{self.counters[1]}')

    def update(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        if self.counters[0] >= 50 and self.counters[1] >= 50:
            frame = self.camera.get_frame()
            prediction = self.model.predict(frame)

            self.error_label.config(text='')
            self.show_prediction(prediction)
        else:
            self.error_label.config(text='Error: You need to provide at least 50 images of each class')

    def show_prediction(self, prediction):
        if prediction <= 0.4:
            self.class_label.config(text=self.classname_one)
            return self.classname_one
        elif prediction >= 0.6:
            self.class_label.config(text=self.classname_two)
            return self.classname_two
        else:
            self.class_label.config(text='I have no idea')
            return 'I have no idea'

