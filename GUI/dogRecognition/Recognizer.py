import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tqdm import tqdm
import collections
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from PIL import Image

accuracy = None
dog = None
results = None

# Define settings
num_breeds = 120
im_size = 224
batch_size = 64
encoder = LabelEncoder()

CLASS_NAME = []

class TeachModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepareImages(self):
        """
        Prepares the images for training the model - mimicking dbi.py approach
        """
        startTime = time.time()
        print("Preparing the dataset of images.")
        
        # Load labels exactly like dbi.py
        df_labels = pd.read_csv('dog-breed-identification/labels.csv')
        train_file = 'dog-breed-identification/train/'
        
        # Select breeds using the same method as dbi.py
        breed_dict = list(df_labels['breed'].value_counts().keys()) 
        new_list = sorted(breed_dict, reverse=True)[:num_breeds] 
        df_labels = df_labels.query('breed in @new_list').copy()
        df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")
        
        # Create image arrays exactly like dbi.py
        train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')
        
        print(f"Loading {len(df_labels)} images...")
        for i, img_id in enumerate(df_labels['img_file']):
            if i % 100 == 0:
                print(f"Processed {i}/{len(df_labels)} images")
            img = Image.open(train_file+img_id)
            img = img.resize((im_size, im_size))
            img_array = preprocess_input(np.expand_dims(np.array(img).astype(np.float32), axis=0))
            train_x[i] = img_array
        
        # Create labels and split data
        train_y = encoder.fit_transform(df_labels["breed"].values)
        
        # Store the breed list for prediction
        global CLASS_NAME
        CLASS_NAME = list(encoder.classes_)
        print(f"Selected {len(CLASS_NAME)} breeds")
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow(
            x_train, 
            y_train, 
            batch_size=batch_size
        )
        
        test_datagen = ImageDataGenerator()
        
        test_generator = test_datagen.flow(
            x_test, 
            y_test, 
            batch_size=batch_size
        )
        
        print('====================')
        print(f"Time taken to prepare images: {(time.time() - startTime) / 60:.2f} minutes")
        print('====================')
        
        return train_generator, test_generator, CLASS_NAME

    def createModel(self):
        """
        Creates the model and prepares it for training
        """
        startTime = time.time()
        print("Creating model")
        
        baseModel = ResNet50V2(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))
        for layer in baseModel.layers:
            layer.trainable = False
        
        # Use the exact same architecture as dbi.py
        x = baseModel.output
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        predictions = Dense(num_breeds, activation='softmax')(x)
        
        model = Model(inputs=baseModel.input, outputs=predictions)
        
        # Use the exact same optimizer and loss as dbi.py
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )
        
        model.summary()
        
        print('====================')
        print(f"Time taken to create model: {(time.time() - startTime) / 60:.2f} minutes")
        print('====================')
        
        return model
    
    def trainModel(self, model, train_generator, test_generator, epochs=20):
        """
        Trains the model and saves it
        """
        global accuracy
        startTime = time.time()
        print("Training Model")
        
        # Callbacks matching dbi.py
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=test_generator.n // batch_size,
            callbacks=[reduce_lr, early_stop]
        )
        
        # Evaluate and save
        score = model.evaluate(test_generator)
        accuracy = round(score[1] * 100, 2)
        print(f'Accuracy over the test set: {accuracy}%')
        
        model.save("dogRecognizer.keras")
        
        # Extract features for anomaly detection
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        featureExtractor = Model(inputs=model.input, outputs=last_conv_layer.output)
        
        # Use batches for feature extraction
        features_dataset = test_generator
        dogFeatures = []
        
        # Reset the generator to the beginning
        features_dataset.reset()
        
        num_batches = len(features_dataset)
        print(f"Processing {num_batches} batches for feature extraction...")

        for i in tqdm(range(num_batches)):
            batch_images, _ = next(features_dataset) 
            batch_features = featureExtractor.predict(batch_images)
            batch_features = np.mean(batch_features, axis=(1, 2))
            dogFeatures.append(batch_features)
        
        if len(dogFeatures) > 0:
            try:
                dogFeatures = np.concatenate(dogFeatures, axis=0)
        
                meanFeatures = np.mean(dogFeatures, axis=0)
                varianceFeatures = np.var(dogFeatures, axis=0)
        
                np.save('mean_features.npy', meanFeatures)
                np.save('variance_features.npy', varianceFeatures)
                print(f"Saved features with shape: {dogFeatures.shape}")
            except Exception as e:
                print(f"Error concatenating features: {e}")
                print(f"Feature shapes: {[f.shape for f in dogFeatures]}")
        else:
            print("No features were extracted!")
        
        print('====================')
        print(f"Time taken to train the model: {(time.time() - startTime) / 60:.2f} minutes")
        print('====================')

class DogPrediction:
    """
    For predicting what the breed of dog is (or if the image is not a dog)
    """
    
    def __init__(self, class_names):
        """
        Loads the model file and the mean and variance features
        """
        self.model = self.loadModel()
        self.meanFeatures, self.varianceFeatures = self.loadMeanVariance()
        self.CLASS_NAME = class_names
        self.dog = "dog"
        
    def loadModel(self):
        return load_model('dogRecognizer.keras')
        
    def loadMeanVariance(self):
        meanFeatures = np.load('mean_features.npy')
        varianceFeatures = np.load('variance_features.npy')
        return meanFeatures, varianceFeatures
    
    def predict(self, image_path):  
        startTime = time.time()
    
        # Use exactly the same preprocessing as in dbi.py for prediction
        img = Image.open(image_path)
        img = img.resize((im_size, im_size))
        img_array = preprocess_input(np.expand_dims(np.array(img).astype(np.float32), axis=0))
    
        # Find the last convolutional layer for anomaly detection
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
            
        # Anomaly detection
        featureExtractor = Model(inputs=self.model.input, outputs=last_conv_layer.output)
        newImageFeatures = featureExtractor.predict(img_array)
    
        # Apply global average pooling to the features
        newImageFeatures = np.mean(newImageFeatures, axis=(1, 2)).flatten()
    
        diff = newImageFeatures - self.meanFeatures
        distance = np.sqrt(np.sum((diff ** 2) / self.varianceFeatures))
    
        anomaly_threshold = 16.00
        self.dog = "dog"  # Set the dog status in the instance
        if distance > anomaly_threshold:
            self.dog = "not dog"  # Update the dog status in the instance
            print(f"This image is not a dog. Distance: {distance:.2f}")
        
        # Predict the breed using the same approach as dbi.py
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.CLASS_NAME[predicted_class_index]
    
        print('====================')
        print(f"Time taken to predict dog: {(time.time() - startTime) / 60:.2f} minutes")
        print('====================')
    
        return predicted_class

def getResults(class_names):   
    global results
    # Load model and make predictions
    prediction = DogPrediction(class_names)
    
    test_images = [
        "Test Images/leonberg.jpg",
        "Test Images/husk.jpg",
        "Test Images/deerhound.jpg",
        "Test Images/germanshep.png",
        "Test Images/yorkshire.jpg",
        "Test Images/appl.jpg",
        "Test Images/cat.jpg",
        "Test Images/fox.jpg",
        "Test Images/wolf.png"
    ]
    
    for img_path in test_images:
        try:
            results = prediction.predict(img_path)
            print('==============================')
            print(f"Image: {img_path}, Predicted Breed: {results}")
        except Exception as e:
            print(f"Error with {img_path}: {e}")

def write_data(epoch, batch_size):
    global accuracy, dog, results
    with open("results.txt", "a") as f:
        f.write(
            f"Accuracy: {accuracy} | Epoch: {epoch} | Batch Size: {batch_size} | "
            f"Dog: {dog} | Results: {results}\n"
        )

def createModel():
    epochs = 20
    
    teachModel = TeachModel()
    train_generator, test_generator, class_names = teachModel.prepareImages()
    model = teachModel.createModel()
    teachModel.trainModel(model, train_generator, test_generator, epochs=epochs)
            
    getResults(class_names)
    write_data(epochs, batch_size)

#if __name__ == "__main__":
#    createModel()

