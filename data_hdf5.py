'''
This script creates an HDF5 file to store all images and their corresponding labels in a single dataset. 
The file is structured to facilitate efficient training for various machine learning and deep learning models.
Images are resized, normalized, and compressed for compact storage.
Why hdf5? 
HDF5 supports chunked storage and built-in compression, 
which can significantly reduce the file size without impacting performance. 
This is particularly beneficial for large datasets like mine, with 50,000 images, as it improves storage efficiency and data transfer speeds.



- Input: 
  1. A CSV file containing image IDs and labels.
  2. A directory with the image files.

- Output: 
  A single HDF5 file containing preprocessed images and labels.
'''

import h5py
import cv2
import numpy as np
import pandas as pd
import os

# Paths
csv_path = "/Users/francescomorandi/Downloads/cifar-10/trainLabels.csv"
image_folder = "/Users/francescomorandi/Downloads/cifar-10/train"

# Load the CSV file
df = pd.read_csv(csv_path)

# Create HDF5 file
with h5py.File("dataset.h5", "w") as hdf:
    img_data = []  # List for images
    labels = []    # List for labels

    for _, row in df.iterrows():
        # Image path construction
        img_path = os.path.join(image_folder, str(row['id']) + ".png")  # Adjust extension if needed
        
        # Load and preprocess the image
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))  # Resize to standard dimensions
            img = img / 255.0  # Normalize pixel values
            img_data.append(img)  # Append image data to the list
            labels.append(row['label'])  # Append label to the list
        else:
            print(f"Warning: Image not found or could not be loaded: {img_path}")
    
    # Save images and labels to HDF5
    hdf.create_dataset("images", data=np.array(img_data), compression="gzip")  # Save images
    dt = h5py.string_dtype(encoding='utf-8')  # Define string data type for labels
    hdf.create_dataset("labels", data=np.array(labels, dtype=dt), compression="gzip")  # Save labels as strings

print("HDF5 dataset successfully created.")
