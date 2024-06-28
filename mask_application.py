import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Function to apply mask to a single image
def apply_mask(image_path, mask_points, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a mask with the same dimensions as the image, initialized to black
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw the polygon on the mask
    points = np.array(mask_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    # Create a 3-channel mask for the color image
    mask_3ch = cv2.merge([mask, mask, mask])

    # Apply the mask to the image, turning everything outside the mask black
    masked_image = cv2.bitwise_and(image, mask_3ch)

    # Save the resulting image
    cv2.imwrite(output_path, masked_image)

# Function to parse the XML mask file
def parse_mask(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract the polygon points from the XML
    polygon = root.find(".//polygon")
    points = polygon.attrib['points']
    points = [tuple(map(float, point.split(','))) for point in points.split(';')]
    
    return points

# Main script to process all images in the folder
def process_folder(input_folder, output_folder, mask_points):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            apply_mask(image_path, mask_points, output_path)
            print(f"Processed {file_name}")

# Set the path to the mask file and parse it
mask_path = "/Users/jorgemartinez/Downloads/mask_west.xml"
mask_points = parse_mask(mask_path)

# Set the path to the folder containing images and the output folder
input_folder = "/Users/jorgemartinez/thesis_retinanet/datasets/west_easy_cars_no_padding_yolo/all/test_new"
output_folder = "/Users/jorgemartinez/thesis_retinanet/datasets/west_easy_cars_no_padding_yolo/all/masked"

# Process the images
process_folder(input_folder, output_folder, mask_points)

