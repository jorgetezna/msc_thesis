import os
import shutil

# Define paths
source_dir = '/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/4.12_cars/color_combined/cutouts'
destination_dir = '/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/4.12_cars/color_combined/cutouts_classification'
car_dir = os.path.join(destination_dir, 'car')
no_car_dir = os.path.join(destination_dir, 'no_car')

# Create destination directories if they don't exist
os.makedirs(car_dir, exist_ok=True)
os.makedirs(no_car_dir, exist_ok=True)

# List all files in the source directory
all_files = os.listdir(source_dir)

# Filter out image files (assuming .jpg format)
image_files = [file for file in all_files if file.endswith('.jpg')]

# Process each image file
for image_file in image_files:
    image_path = os.path.join(source_dir, image_file)
    annotation_path = os.path.splitext(image_path)[0] + '.txt'
    
    # Check if the annotation file is empty or not
    if os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0:
        # If the annotation file is not empty, it's a 'car' image
        dest_path = os.path.join(car_dir, image_file)
    else:
        # If the annotation file is empty, it's a 'no_car' image
        dest_path = os.path.join(no_car_dir, image_file)
    
    # Copy the image to the appropriate directory
    shutil.copy(image_path, dest_path)

print("Dataset transformation complete.")