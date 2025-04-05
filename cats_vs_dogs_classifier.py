import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore


class App(ctk.CTk):
    def __init__(self, title, size):
        # main setup
        super().__init__()
        self.title(title)

        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(size[0],size[1])
        self.maxsize(size[0],size[1])

        ctk.set_appearance_mode('dark')

        # create the menu bar
        self.create_menubar()

        # create the selection and result frames
        self.selection_frame = Selection_Frame(self, self.on_select_pressed)
        # create the frame where the image will be displayed
        self.image_frame = Image_Frame(self)
        
        # Load the model
        self.model = tf.keras.models.load_model('cats_dogs_cnn_model.h5') # type: ignore

        # run
        self.mainloop()

    # Load and preprocess the image to be fed to the model for prediction
    def load_and_preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(160, 160))  # Resize
        img_array = image.img_to_array(img)  # Convert to NumPy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array

    # Function to open file dialog, load image and predict the class
    # Returns the prediction value
    def open_image(self) -> float:
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)

            img.thumbnail((640, 800))  # Resize while keeping aspect ratio
            img_tk = ImageTk.PhotoImage(img)
            self.image_frame.display_image(img_tk)

            # Preprocess the image and make prediction
            img_array = self.load_and_preprocess_image(file_path)
            prediction = self.model.predict(img_array)

            return prediction[0][0]  # Get the prediction value
        else:
            return 2.0  # return a value that is not valid for a prediction

    def create_menubar(self):        
        self.menu = tk.Menu(self)

        # sub menu file
        self.file_menu = tk.Menu(self.menu, tearoff=False)
        self.file_menu.add_command(label='Exit', command=self.on_exit_pressed)
        self.menu.add_cascade(label='File', menu=self.file_menu)

        # sub menu appearance mode
        self.help_menu = tk.Menu(self.menu, tearoff=False)
        self.help_menu.add_command(label='Light', command=lambda: ctk.set_appearance_mode('light'))
        self.help_menu.add_command(label='Dark', command=lambda: ctk.set_appearance_mode('dark'))
        self.menu.add_cascade(label='Appearance', menu=self.help_menu)

        self.configure(menu=self.menu)

    # called when Select button is pressed
    def on_select_pressed(self):
        prediction = self.open_image()
        self.selection_frame.display_prediction_result(prediction)
    
    # called when File/Exit is selected
    def on_exit_pressed(self):
        self.quit()         # Stop the event loop
        self.destroy()      # Destroy the window
            

class Selection_Frame(ctk.CTkFrame):
    def __init__(self, parent, on_select):
        super().__init__(parent, border_width=1)
        self.place(relx=0, y=0, relwidth=0.33, relheight=1)
        self.parent = parent
        
        # create the widgets
        self.select_btn = ctk.CTkButton(self, text="Select an image", font=("", 25, "bold"), command=on_select)
        self.select_btn.grid(row=0, column=0, sticky='nswe', padx=45, pady=30)
        self.result_txt_lbl = ctk.CTkLabel(self, text="Is it a cat or a dog? ", anchor="center", font=("", 25, "bold"))
        self.result_txt_lbl.grid(row=1, column=0, sticky='ew', padx=4)
        self.result_lbl = ctk.CTkLabel(self, text=" ", anchor="center", padx=5)
        self.result_lbl.grid(row=2, column=0, sticky='nswe')
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)  

        img = Image.open("appImages/cat_or_dog.jpeg")
        img.thumbnail((240, 240))  # Resize while keeping aspect ratio
        img_dog_tk = ImageTk.PhotoImage(img)
        self.result_lbl.configure(image=img_dog_tk)
        self.result_lbl.image = img_dog_tk # type: ignore

           
    def display_prediction_result(self, prediction: float):
        img = Image.open("appImages/dog.jpeg")
        img.thumbnail((240, 240))  # Resize while keeping aspect ratio
        img_dog_tk = ImageTk.PhotoImage(img)

        img = Image.open("appImages/cat.jpeg")
        img.thumbnail((240, 240))  # Resize while keeping aspect ratio
        img_cat_tk = ImageTk.PhotoImage(img)

        # Interpret the result
        if prediction <= 1.0:
            if prediction > 0.5:
                confidence = (prediction -0.5) / 0.5
                self.result_lbl.configure(image=img_dog_tk)
                self.result_lbl.image = img_dog_tk # type: ignore
                self.result_txt_lbl.configure(text="It's a dog " + str(round(confidence*100, 2)) + "% sure")
            else:
                confidence = (0.5 - prediction) / 0.5
                self.result_lbl.configure(image=img_cat_tk)
                self.result_lbl.image = img_cat_tk # type: ignore
                self.result_txt_lbl.configure(text="It's a cat "+ str(round(confidence*100, 2)) + "% sure")
            
            
class Image_Frame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, border_width=1)
        self.place(relx=0.33, y=0, relwidth=0.67, relheight=1)

        # create the widgets
        self.image_lbl = ctk.CTkLabel(self, text="", anchor="center")
        self.image_lbl.grid(row=0, column=0, sticky='nswe', padx=15, pady=15)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)  

        img = Image.open("appImages/cats_and_dogs.jpeg")
        img.thumbnail((640, 640))  # Resize while keeping aspect ratio
        img_cats_and_dogs_tk = ImageTk.PhotoImage(img)
        self.image_lbl.configure(image=img_cats_and_dogs_tk)
        self.image_lbl.image = img_cats_and_dogs_tk # type: ignore
      
    def display_image(self, img_tk):
        self.image_lbl.configure(image=img_tk)
        self.image_lbl.image = img_tk # type: ignore
        
        


App('Cats vs Dogs CNN', (1000, 700))