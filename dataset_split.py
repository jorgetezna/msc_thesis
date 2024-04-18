import os
import shutil
from glob import glob
import random

# Set the dataset directory here
dataset_dir = '/Users/jorgemartinez/thesis_retinanet/vehicle-detection-from-satellite-images-data-set'  # Update this to your dataset directory

# Make sure to replace 'jpg' with the correct image file extension if it's different
image_paths = glob(os.path.join(dataset_dir, '*.jpg'))
random.shuffle(image_paths)

# Calculate split sizes
total_images = len(image_paths)
train_split = int(0.7 * total_images)
val_split = int(0.9 * total_images)  # 70% for train, 20% for validation, and the rest for test

# Create train, valid, test directories
train_dir = os.path.join(dataset_dir, 'train')
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to move images and corresponding XMLs
def move_files(start_index, end_index, destination):
    for i in range(start_index, end_index):
        image_path = image_paths[i]
        xml_path = image_path.replace('.jpg', '.xml')  # Replace with actual image extension if not jpg

        # Move files
        shutil.move(image_path, destination)
        shutil.move(xml_path, destination)

# Move files to respective directories
move_files(0, train_split, train_dir)
move_files(train_split, val_split, valid_dir)
move_files(val_split, total_images, test_dir)

print('Dataset successfully split and moved into train, valid, and test folders.')
