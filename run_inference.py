import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO('/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/yolov8_runs/kaiming_new2_public3/weights/best.pt')

iou_threshold = 0.2  # IoU threshold for mAP@0.5 calculation
score_threshold = 0.2  # Confidence score threshold for detections

def run_yolo_on_frame(frame):
    """Run YOLO on a single frame and return detections."""
    results = model(frame)
    boxes = results[0].boxes.xyxy if len(results) > 0 else np.empty((0, 4))
    scores = results[0].boxes.conf if len(results) > 0 else np.empty((0,))
    return boxes.cpu().numpy(), scores.cpu().numpy()

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes."""
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

def match_detections_to_ground_truth(detections, ground_truths, iou_threshold):
    """Match detections to ground truth and count true positives."""
    tp = 0
    matched = []
    
    for detection in detections:
        detection_box = detection[:4]  # Only take the bounding box coordinates
        for gt in ground_truths:
            if gt not in matched and compute_iou(detection_box, gt) >= iou_threshold:
                tp += 1
                matched.append(gt)
                break
    
    fp = len(detections) - tp
    fn = len(ground_truths) - tp
    
    return tp, fp, fn

def compute_average_precision(detections, ground_truths, iou_threshold):
    """Compute average precision for a single IoU threshold."""
    precisions = []
    recalls = []
    thresholds = np.arange(0, 1.1, 0.1)
    
    for t in thresholds:
        tp, fp, fn = match_detections_to_ground_truth([d for d in detections if d[4] >= t], ground_truths, iou_threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_indices = np.argsort(recalls)
    precisions = precisions[sorted_indices]
    recalls = recalls[sorted_indices]

    ap = np.trapz(precisions, recalls)
    return ap

def draw_boxes(image, boxes, color=(0, 255, 0), label=''):
    """Draw bounding boxes on the image."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def main(images_folder):
    image_files = sorted(Path(images_folder).glob('*.jpg'))  # Assuming images are in .jpg format
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_ground_truth_cars = 0
    all_detections = []
    all_ground_truths = []

    for image_file in image_files:
        frame = cv2.imread(str(image_file))

        # Run YOLO on the current frame
        boxes, scores = run_yolo_on_frame(frame)

        # Filter detections using confidence score threshold
        high_conf_indices = scores > score_threshold
        filtered_boxes = boxes[high_conf_indices]

        # Load ground truth
        ground_truth_file = image_file.with_suffix('.txt')
        image_size = frame.shape[:2]
        ground_truth_boxes = load_ground_truth(ground_truth_file, image_size)

        # Store detections and ground truths
        all_detections.extend([list(box) + [score] for box, score in zip(filtered_boxes, scores[high_conf_indices])])
        all_ground_truths.extend(ground_truth_boxes)

        # Match detections to ground truth
        tp, fp, fn = match_detections_to_ground_truth(filtered_boxes.tolist(), ground_truth_boxes, iou_threshold)

        total_true_positives += tp
        total_false_positives += fp
        total_false_negatives += fn
        total_ground_truth_cars += len(ground_truth_boxes)

        # Draw predictions and ground truths
        frame_with_predictions = frame.copy()
        draw_boxes(frame_with_predictions, filtered_boxes, color=(0, 0, 255), label='Pred')

        frame_with_ground_truth = frame.copy()
        draw_boxes(frame_with_ground_truth, ground_truth_boxes, color=(0, 255, 0), label='GT')

        # Display images separately
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(frame_with_ground_truth, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(frame_with_predictions, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Predictions')
        axes[1].axis('off')

        plt.suptitle(f'Comparison for {image_file.name}')
        plt.show()

    # Compute average precision for IoU threshold 0.5
    average_precision = compute_average_precision(all_detections, all_ground_truths, iou_threshold)

    # Calculate precision, recall, and F1 score
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Total cars correctly detected: {total_true_positives}/{total_ground_truth_cars}")
    print(f"Detection ratio: {total_true_positives / total_ground_truth_cars:.2f}")
    print(f"Average Precision (AP) @0.5: {average_precision:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"False Positives: {total_false_positives}")

if __name__ == "__main__":
    images_folder = '/Users/jorgemartinez/thesis_retinanet/datasets/west_easy_cars_no_padding_yolo/all/test_new'
    main(images_folder)
