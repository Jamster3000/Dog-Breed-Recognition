import customtkinter
import tkinter as tk
from PIL import Image
import os
import numpy as np
from tkinter import messagebox
import threading
import time

import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from Recognizer import im_size, preprocess_input

DOG_BREEDS = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 
    'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 
    'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 
    'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 
    'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 
    'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 
    'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 
    'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 
    'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 
    'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 
    'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 
    'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 
    'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 
    'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 
    'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 
    'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 
    'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 
    'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 
    'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 
    'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 
    'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 
    'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier'
]

class GuiApp:
    def __init__(self, root):
        root.title("Dog Breed Recognizer")
        customtkinter.set_appearance_mode("system") 
        customtkinter.set_default_color_theme("green")

        screenWidth = 600
        screenHeight = 500
        root.geometry(f"{screenWidth}x{screenHeight}")

        self.current_file_path = None
        self.image_label = None

        self.imageFrame = customtkinter.CTkFrame(root, width=300, height=300, border_width=5, border_color="#2CC985")
        self.resultFrame = customtkinter.CTkFrame(root, width=260, height=300, border_width=5, border_color="#2CC985")
        
        #Process image button
        self.processButton = customtkinter.CTkButton(
            root, 
            text="Recognize Breed", 
            command=self.processImageButton, 
            width=575, 
            height=40,
            fg_color="#2CC985",
            hover_color="#25A06E"
        )
        
        #choose image button
        self.fileChooserButton = customtkinter.CTkButton(
            root, 
            text="Choose Image", 
            command=self.openFileDialogue, 
            width=575, 
            height=40,
            fg_color="#4284f5",
            hover_color="#3464C5"
        )  

        #Preidction result labels
        self.resultLabel = customtkinter.CTkLabel(
            self.resultFrame, 
            text="Prediction Results", 
            font=("Arial", 18, "bold"),
            text_color="#2CC985"
        )
        
        #breed label
        self.breedLabel = customtkinter.CTkLabel(
            self.resultFrame, 
            text="", 
            font=("Arial", 16)
        )
        
        #confidence label
        self.confidenceLabel = customtkinter.CTkLabel(
            self.resultFrame, 
            text="", 
            font=("Arial", 14)
        )
        
        # processing indicator label
        self.processingLabel = customtkinter.CTkLabel(
            self.resultFrame,
            text="",
            font=("Arial", 14),
            text_color="#FF9A3C"
        )

        self.imageFrame.place(x=10, y=95)
        self.resultFrame.place(x=325, y=95)
        self.fileChooserButton.place(x=10, y=15)
        self.processButton.place(x=10, y=440)
        
        self.resultLabel.place(x=10, y=20)
        self.breedLabel.place(x=10, y=70)
        self.confidenceLabel.place(x=10, y=110)
        self.processingLabel.place(x=10, y=190)
        
        #create a new thread to load the model in the background.
        self.model_initialized = False
        self.initProgressLabel = customtkinter.CTkLabel(
            root,
            text="Loading model... Please wait",
            font=("Arial", 14),
            text_color="#FF9A3C"
        )
        self.initProgressLabel.place(x=230, y=60)
        
        #actually intiate the model thread
        threading.Thread(target=self.initializeModel, daemon=True).start()

    def initializeModel(self):
        """Initialize the model in a background thread"""
        try:
            self.predictor = DogBreedPredictor()
            self.model_initialized = True
            self.initProgressLabel.after(0, self.initProgressLabel.destroy)
        except Exception as e:
            self.initProgressLabel.after(0, lambda: self.initProgressLabel.configure(
                text=f"Error loading model. {e}",
                text_color="#E74C3C"
            ))
    
    def openFileDialogue(self):
        """Open file dialog to select an image"""
        filePath = tk.filedialog.askopenfilename(
            title="Select a dog image", 
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
    
        if filePath:
            self.current_file_path = filePath
            
            #Clear the results from previous image
            self.breedLabel.configure(text="")
            self.confidenceLabel.configure(text="")
            
            #remove the previous image
            if hasattr(self, 'inputtedImageLabel') and self.inputtedImageLabel is not None:
                self.inputtedImageLabel.destroy()
            
            openImageFile = Image.open(filePath)
            imageWidth, imageHeight = openImageFile.size
        
            frameWidth = 290 
            frameHeight = 290 
        
            scaleWidth = frameWidth / imageWidth
            scaleHeight = frameHeight / imageHeight
            scaleFactor = min(scaleWidth, scaleHeight)
        
            newWidth = int(imageWidth * scaleFactor)
            newHeight = int(imageHeight * scaleFactor)
        
            self.inputtedImage = customtkinter.CTkImage(
                light_image=openImageFile, 
                size=(newWidth, newHeight)
            )
            
            self.inputtedImageLabel = customtkinter.CTkLabel(
                self.imageFrame, 
                image=self.inputtedImage, 
                text=""
            )
            
            self.inputtedImageLabel.place(relx=0.5, rely=0.5, anchor='center')
            
    def update_processing_status(self, processing=True):
        """Update the processing status indicator"""
        if processing:
            self.processingLabel.configure(text="Processing image...")
            self.processButton.configure(state="disabled")
        else:
            self.processingLabel.configure(text="")
            self.processButton.configure(state="normal")
            
    def processImageButton(self):
        """Process the selected image to recognize the dog breed"""
        if not hasattr(self, 'current_file_path') or self.current_file_path is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        if not self.model_initialized:
            messagebox.showwarning("Model Loading", "Model is still loading. Please wait.")
            return
            
        self.update_processing_status(True)
        
        #Process the image in a separate thread to avoid blocking the UI
        threading.Thread(target=self.process_image_thread, daemon=True).start()
            
    def process_image_thread(self):
        """Process the image in a background thread"""
        try:
            breed_name, confidence = self.predictor.predict(self.current_file_path)
            
            self.breedLabel.after(0, lambda: self.breedLabel.configure(
                text=f"Breed: {breed_name}"
            ))
            
            self.confidenceLabel.after(0, lambda: self.confidenceLabel.configure(
                text=f"Confidence: {confidence:.2f}%"
            ))
            
        except Exception as e:
            self.breedLabel.after(0, lambda: messagebox.showerror(
                "Prediction Error", 
                f"Error during prediction: {str(e)}"
            ))
        finally:
            # Always reset the processing status
            self.breedLabel.after(0, lambda: self.update_processing_status(False))


#The actual dog breed indicator
class DogBreedPredictor:
    def __init__(self):
        self.model_loaded = False
        
        # Load model
        self.model = load_model('dogRecognizer.keras')
        
        #Warm up the model with a dummy prediction to reduce first prediction latency
        dummy_input = np.zeros((1, im_size, im_size, 3), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        
        self.model_loaded = True
            
    def predict(self, image_path):
        if not self.model_loaded:
            return "Model not loaded", 0.0
    
        #load and process the image
        img = Image.open(image_path)
        img = img.resize((im_size, im_size))
        img_array = preprocess_input(np.expand_dims(np.array(img).astype(np.float32), axis=0))
    
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
    
        #Anomaly detection to check if it is a dog or not
        featureExtractor = Model(inputs=self.model.input, outputs=last_conv_layer.output)
        newImageFeatures = featureExtractor.predict(img_array, verbose=0)
    
        newImageFeatures = np.mean(newImageFeatures, axis=(1, 2)).flatten()
    
        # Calculate distance for anomaly detection
        try:
            if not hasattr(self, 'meanFeatures') or self.meanFeatures is None:
                self.meanFeatures = np.load('mean_features.npy')
                self.varianceFeatures = np.load('variance_features.npy')
            
            diff = newImageFeatures - self.meanFeatures
            distance = np.sqrt(np.sum((diff ** 2) / self.varianceFeatures))
        
            anomaly_threshold = 45.0
        
            if distance > anomaly_threshold:
                print(f"This image is not a dog. Distance: {distance:.2f}")
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
    
        if distance < anomaly_threshold:
            #predict breed
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_index] * 100
    
            if predicted_class_index < len(DOG_BREEDS):
                breed_name = DOG_BREEDS[predicted_class_index]
                breed_name = breed_name.replace('_', ' ').title()
            else:
                breed_name = f"Unknown Breed"
    
            return breed_name, confidence
        else:
            return "Not a Dog", 0.00

    
root = customtkinter.CTk()

application = GuiApp(root)

root.mainloop()
