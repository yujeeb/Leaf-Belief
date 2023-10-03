from tkinter import*
from tkinter import messagebox
import cv2
import tkinter as tk
import os
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import tensorflow as tf
import numpy as np

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.cap = cv2.VideoCapture(self.video_source)
        
        # Create a button for capturing the image
        self.capture_button = tk.Button(window, text="Upload", fg="white", bg="blue", width=25, command=self.upload)
        self.capture_button.grid(row=0, column=0)
        
        self.capture_button = tk.Button(window, text="Capture", fg="white", bg="blue", width=25, command=self.capture)
        self.capture_button.grid(row=0, column=1)
        
        self.capture_button = tk.Button(window, text="Evaluate", fg="white", bg="green", width=25, command=self.evaluate)
        self.capture_button.grid(row=1, column=0)

        self.exit = tk.Button(window, text="Exit", width=25, fg="white", bg="red", command=self.exit_cap)
        self.exit.grid(row=1, column=1)

        # Create a label for displaying the webcam stream
        self.lmain = tk.Label(window)
        self.lmain.grid(row=2, column=0, columnspan=2)

        self.update()

        self.window.mainloop()

    def exit_cap(self):
        try:
            self.window.destroy()
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as es:
            messagebox.showerror("Error", f"Due to:{str(es)}", parent=self.window)

    def update(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Crop the frame to a square
                frame = frame[0: frame.shape[0],
                        int((frame.shape[1] - frame.shape[0]) / 2): int((frame.shape[1] - frame.shape[0]) / 2) +
                                                                    frame.shape[0]]

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lmain.imgtk = imgtk
                self.lmain.configure(image=imgtk)
            self.window.after(10, self.update)
        except Exception as es:
            messagebox.showerror("Error", f"Due to:{str(es)}", parent=self.window)

    def capture(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Crop the frame to a square
                frame = frame[0: frame.shape[0],
                        int((frame.shape[1] - frame.shape[0]) / 2): int((frame.shape[1] - frame.shape[0]) / 2) +
                                                                    frame.shape[0]]
                frame = cv2.resize(frame, (256, 256))
                frame = frame[0:256, 0:256]

                file_name = "Image.jpg"
                if not os.path.exists("Images"):
                    os.makedirs("Images")
                cv2.imwrite("Images/" + file_name, frame)
                print("Captured " + file_name)
                l = Label(root, text = "Captured Image")
            else:
                print("Error: Could not capture image.")
        except Exception as es:
            messagebox.showerror("Error", f"Due to:{str(es)}", parent=self.window)

    def upload(self):
        try:
            f_types = [('Jpg Files', '*.jpg')]   # type of files to select 
            filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
            frame = cv2.imread(filename[0])
            frame = cv2.resize(frame, (256, 256))
            frame = frame[0:256, 0:256]
            file_name = "Image.jpg"
            if not os.path.exists("Images"):
                os.makedirs("Images")
            cv2.imwrite("Images/" + file_name, frame)
            print("Uploaded " + file_name)
            l = Label(root, text = "Uploaded Image")
        except Exception as es:
            messagebox.showerror("Error", f"Due to:{str(es)}", parent=self.window)
        
    def evaluate(self):
        model = tf.keras.models.load_model("tomatoes_95.39.h5")
        filepath = "Images/Image.jpg"
        img = tf.keras.utils.load_img(
            filepath, target_size=(256, 256)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        prediction = model.predict(img_array)
        print(prediction)
        print("PREDICTION COMPLETED!")
        pred_index = np.argmax(prediction)
        print("Prediction Index: ", pred_index)
        classes = ['Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold', 'Septoria leaf spot',
                   'Spider mites Two-spotted spider mite', 'Target Spot', 'Tomato Yellow Leaf Curl Virus',
                   'Tomato mosaic virus', 'healthy']
        print(classes[pred_index])

if __name__ == "__main__":
        root = Tk()
        obj = App(root, "Leaf Belief")
        root.mainloop()
