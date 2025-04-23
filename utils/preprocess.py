import cv2
import numpy as np

def preprocess_frame(frame, img_height, img_width):
    #Resize to match training input 
    frame = cv2.resize(frame, (img_width, img_height))
    
    # Convert to greyscale for webcam input
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Normalize and reshape for the model input
    frame = frame.reshape(1, img_height, img_width, 1) / 255.0
    return frame