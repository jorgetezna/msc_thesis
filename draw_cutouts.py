import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import glob

def parse_xml_for_polylines(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    polylines = []
    for polyline in root.findall('.//polyline'):
        points_str = polyline.get('points')
        points = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
        polylines.append(points)
    return polylines

def slide_window_along_polyline(points, window_size, step_size, image_width, image_height):
    windows = []
    window_width, window_height = window_size
    for i in range(len(points) - 1):
        start_point = np.array(points[i])
        end_point = np.array(points[i+1])
        direction = end_point - start_point
        distance = np.linalg.norm(direction)
        direction /= distance

        num_steps = int(distance / step_size) + 1
        for step in range(num_steps + 1):
            center_point = start_point + step * step_size * direction
            top_left = np.maximum(center_point - np.array([window_width / 2, window_height / 2]), 0)
            bottom_right = np.minimum(top_left + np.array([window_width, window_height]), np.array([image_width, image_height]))
            windows.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    return windows

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        boxes = [list(map(float, line.strip().split())) for line in file.readlines()]
    return boxes

def process_images_in_folder(folder_path, xml_path, output_folder, window_size=(60, 60), step_size=30):
    polylines = parse_xml_for_polylines(xml_path)
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))  # Assuming images are in JPEG format

    for image_path in image_files:
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue  # Skip files that cannot be read as images
        image_height, image_width = original_image.shape[:2]
        annotation_path = image_path.replace('.jpg', '.txt')
        boxes = load_annotations(annotation_path)
        
        for points in polylines:
            windows = slide_window_along_polyline(points, window_size, step_size, image_width, image_height)
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            for i, (x1, y1, x2, y2) in enumerate(windows):
                sub_image = original_image[int(y1):int(y2), int(x1):int(x2)]
                window_filename = f"{base_filename}_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.jpg"
                window_path = os.path.join(output_folder, window_filename)
                cv2.imwrite(window_path, sub_image)

                # Save annotations if present
                annotation_file_path = window_path.replace('.jpg', '.txt')
                with open(annotation_file_path, 'w') as file:
                    for box in boxes:
                        class_id, x_center, y_center, width, height = box
                        # Convert from relative to absolute coordinates
                        abs_x_center = x_center * image_width
                        abs_y_center = y_center * image_height
                        abs_width = width * image_width
                        abs_height = height * image_height

                        box_x_min = max(abs_x_center - abs_width / 2, x1)
                        box_y_min = max(abs_y_center - abs_height / 2, y1)
                        box_x_max = min(abs_x_center + abs_width / 2, x2)
                        box_y_max = min(abs_y_center + abs_height / 2, y2)

                        if box_x_min < x2 and box_x_max > x1 and box_y_min < y2 and box_y_max > y1:
                            norm_x_center = ((box_x_min + box_x_max) / 2 - x1) / (x2 - x1)
                            norm_y_center = ((box_y_min + box_y_max) / 2 - y1) / (y2 - y1)
                            norm_width = (box_x_max - box_x_min) / (x2 - x1)
                            norm_height = (box_y_max - box_y_min) / (y2 - y1)
                            file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

# Example usage

process_images_in_folder(
    folder_path='/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/94.13_cars/color_combined3/ok',
    xml_path='/Users/jorgemartinez/thesis_retinanet/datasets/mask94_13.xml',
    output_folder='/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/94.13_cars/color_combined3/cutouts',
    window_size=(60, 60),
    step_size=30
)


