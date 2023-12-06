import cv2
from tensorflow.keras.models import load_model
import numpy as np


class CharRecognizer:
    def __init__(self):
        # Load the character recognition Neural Network 
        model_file = 'braille_ocr_model.h5'
        self.model = load_model(model_file)
    
    def recognize(self, image):
        # Scale pixel values between 0 and 1
        x = image  / 255.0
        
        # Add a channel dimension
        x = np.expand_dims(x, axis=-1)  
        x = np.expand_dims(x, axis=0)
        
        # Use model to classify the character 
        pred = self.model(x, training=False)
        
        # Return the prediction as an integer representing the bits of the character
        return self.__pred_to_int(pred)
        
    # Returns integer representation of character from a model prediction 
    def __pred_to_int(self, pred):
        # Convert prediction to binary multi-hot array
        pred = pred.numpy()[0]
        pred = pred.round()
        pred = pred.astype('int')
        
        # Convert multi-hot array to integer
        pred_int = 0
        for i in range(len(pred)):
            pred_int += pow(2, i) * pred[i]
            
        return int(pred_int)