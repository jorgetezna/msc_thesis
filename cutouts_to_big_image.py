import os
import re
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN

def load_model(weights_path):
    """Load the trained YOLOv8 model."""
    model = YOLO(weights_path)
    return model

def parse_window_filename(filename):
    """Parse the window filename to extract the coordinates of the window in the larger image."""
    pattern = r'_(\d+)_(\d+)_(\d+)_(\d+)\.jpg$'
    match = re.search(pattern, filename)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return x1, y1, x2, y2
    else:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")

def convert_coordinates(window_coords, box):
    """Convert box coordinates from window to larger image coordinate system."""
    x1, y1, x2, y2 = window_coords
    box_x1, box_y1, box_x2, box_y2 = box
    new_x1 = x1 + box_x1
    new_y1 = y1 + box_y1
    new_x2 = x1 + box_x2
    new_y2 = y1 + box_y2
    return [new_x1, new_y1, new_x2, new_y2]

def run_inference(model, image_path, conf_threshold=0.7):
    """Run inference on a single window."""
    image = cv2.imread(image_path)
    results = model(image)
    
    if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'boxes'):
        # Extract boxes from the first result
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        # Filter out low-confidence predictions
        high_conf_indices = scores > conf_threshold
        boxes = boxes[high_conf_indices]
        scores = scores[high_conf_indices]
        return np.hstack((boxes, scores[:, np.newaxis])).tolist()
    return []

def load_ground_truth(ground_truth_path, image_size):
    """Load ground truth bounding boxes from a YOLO format .txt file."""
    boxes = []
    h, w = image_size
    try:
        with open(ground_truth_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    boxes.append([x1, y1, x2, y2])
    except FileNotFoundError:
        # No ground truth file means no objects in the image
        pass
    return boxes

def cluster_and_merge_boxes(boxes, eps=30, min_samples=1):
    """Cluster and merge bounding boxes using DBSCAN."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    centers = np.vstack((x_centers, y_centers)).T

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_

    merged_boxes = []
    for label in set(labels):
        if label == -1:
            continue
        label_boxes = boxes[labels == label]
        x1 = np.min(label_boxes[:, 0])
        y1 = np.min(label_boxes[:, 1])
        x2 = np.max(label_boxes[:, 2])
        y2 = np.max(label_boxes[:, 3])
        merged_boxes.append([x1, y1, x2, y2])

    return merged_boxes

def aggregate_detections(image_detections):
    """Aggregate detections for each larger image and apply clustering to merge boxes."""
    all_boxes = []
    for window_coords, detections in image_detections.items():
        for detection in detections:
            x1, y1, x2, y2, score = detection
            box = convert_coordinates(window_coords, [x1, y1, x2, y2])
            all_boxes.append([*box, score])

    if all_boxes:
        all_boxes = np.array(all_boxes)
        final_boxes = cluster_and_merge_boxes(all_boxes)
        return final_boxes
    return []

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def compute_precision_recall(detections, ground_truths, iou_threshold=0.2):
    """Compute precision and recall for a single IoU threshold."""
    tp, fp, fn = 0, 0, 0
    matched_gts = []

    for detection in detections:
        best_iou = 0
        best_gt = None
        for gt in ground_truths:
            iou = compute_iou(detection[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        if best_iou >= iou_threshold and best_gt not in matched_gts:
            tp += 1
            matched_gts.append(best_gt)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall, tp

def compute_map(detections, ground_truths, iou_threshold=0.2):
    """Compute mean Average Precision (mAP)"""
    precisions = []
    recalls = []

    for i in range(len(detections)):
        precision, recall, _ = compute_precision_recall(detections[:i+1], ground_truths, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_indices = np.argsort(recalls)
    precisions = precisions[sorted_indices]
    recalls = recalls[sorted_indices]

    ap = np.trapz(precisions, recalls)
    return ap

def visualize_detections(image_path, detection_boxes, ground_truth_boxes):
    """Visualize the detections and ground truth on the larger image."""
    image = cv2.imread(image_path)
    
    # Draw detection boxes in green
    detection_image = image.copy()
    for box in detection_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw ground truth boxes in blue
    gt_image = image.copy()
    for box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(gt_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for ground truth

    # Plot the images side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Detections')
    axs[1].axis('off')
    
    plt.show()

# Paths to weights, windows folder, and large images folder
weights_path = '/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/yolov8_runs/cutouts_new/weights/best.pt'
windows_folder = '/Users/jorgemartinez/thesis_retinanet/datasets/west_easy_cars_no_padding_yolo/test_cutouts_new'
large_images_folder = '/Users/jorgemartinez/thesis_retinanet/datasets/west_easy_cars_no_padding_yolo/all/test_new'

# Load the model
model = load_model(weights_path)

# Dictionary to hold detections for each large image
image_detections = defaultdict(list)

# Run inference on each window
for window_file in Path(windows_folder).rglob('*.jpg'):
    try:
        window_coords = parse_window_filename(window_file.name)
        if window_coords:
            detections = run_inference(model, str(window_file))
            image_name = '_'.join(window_file.name.split('_')[:-4]) + '.jpg'
            image_detections[image_name].append((window_coords, detections))
    except ValueError as e:
        print(e)

# Aggregate detections and apply clustering for each large image
total_precisions = []
total_recalls = []
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
total_ground_truth_cars = 0
all_detections = []
all_ground_truths = []

for image_name, window_info_detections in image_detections.items():
    detections_dict = defaultdict(list)
    for window_coords, detections in window_info_detections:
        detections_dict[window_coords].extend(detections)
    
    final_boxes = aggregate_detections(detections_dict)
    
    image_path = str(Path(large_images_folder) / image_name)
    ground_truth_path = str(Path(large_images_folder) / image_name.replace('.jpg', '.txt'))
    image_size = cv2.imread(image_path).shape[:2]
    ground_truth_boxes = load_ground_truth(ground_truth_path, image_size)
    
    all_detections.extend([[*box, 1.0] for box in final_boxes])  # Assigning a dummy confidence of 1.0
    all_ground_truths.extend(ground_truth_boxes)
    
    precision, recall, true_positives = compute_precision_recall([[*box, 1.0] for box in final_boxes], ground_truth_boxes.copy(), iou_threshold=0.2)
    total_precisions.append(precision)
    total_recalls.append(recall)
    
    total_true_positives += true_positives
    total_false_positives += len(final_boxes) - true_positives
    total_false_negatives += len(ground_truth_boxes) - true_positives
    total_ground_truth_cars += len(ground_truth_boxes)
    
    visualize_detections(image_path, final_boxes, ground_truth_boxes)

mean_precision = np.mean(total_precisions)
mean_recall = np.mean(total_recalls)

# Compute average precision for IoU threshold 0.5
average_precision = compute_map(all_detections, all_ground_truths.copy(), iou_threshold=0.2)

# Calculate F1 Score
f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0

print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"Total cars correctly detected: {total_true_positives}/{total_ground_truth_cars}")
print(f"Detection ratio: {total_true_positives / total_ground_truth_cars:.2f}")
print(f"Average Precision (AP) @0.5: {average_precision:.4f}")
print(f"False Positives: {total_false_positives}")
