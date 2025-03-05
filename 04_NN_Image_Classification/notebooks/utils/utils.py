import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def load_images_from_folder(folder, target_size, classes_to_use=['all']):
    images = []
    labels = []
    
    # Loop through each class folder
    for class_name in os.listdir(folder):
        # Skip if not in classes_to_use
        if classes_to_use != ['all'] and class_name not in classes_to_use:
            continue
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Loop through each image in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Load and preprocess the image
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                
                # Add to our lists
                images.append(img_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)