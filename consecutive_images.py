import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO('/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/yolov8_runs/public3_new/weights/best.pt')

score_threshold = 0.1  # Confidence score threshold for detections
motion_threshold = 0.001  # Extremely low motion threshold to make detection as sensitive as possible
iou_threshold = 0.2  # IoU threshold for mAP calculation

def run_yolo_on_frame(frame):
    """Run YOLO on a single frame and return detections."""
    results = model(frame)
    boxes = results[0].boxes.xyxy if len(results) > 0 else np.empty((0, 4))
    scores = results[0].boxes.conf if len(results) > 0 else np.empty((0,))
    return boxes.cpu().numpy(), scores.cpu().numpy()

def draw_boxes(image, boxes, color=(0, 255, 0), label=''):
    """Draw bounding boxes on the image."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def compute_optical_flow(prev_frame, current_frame):
    """Compute dense optical flow using Farneback method."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 1, 3, 1, 3, 0.5, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag

def apply_nms(detections, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to filter overlapping boxes."""
    if len(detections) == 0:
        return []
    
    boxes = np.array(detections)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep].tolist()

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
                    boxes.append((x1, y1, x2, y2))  # Convert to tuple
    except FileNotFoundError:
        # No ground truth file means no objects in the image
        pass
    return boxes

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

def match_detections_to_ground_truth(detections, ground_truths, iou_threshold):
    """Match detections to ground truth and count true positives."""
    tp = 0
    matched = set()
    
    for detection in detections:
        detection_box = tuple(detection[:4])
        for gt in ground_truths:
            if gt not in matched and compute_iou(detection_box, gt) >= iou_threshold:
                tp += 1
                matched.add(gt)
                break
    
    fp = len(detections) - tp
    fn = len(ground_truths) - tp
    
    return tp, fp, fn

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

def main(images_folder):
    image_files = sorted(Path(images_folder).glob('*.jpg'))  # Assuming images are in .jpg format

    previous_frame = None
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_ground_truth_cars = 0
    all_detections = []
    all_ground_truths = []

    for image_file in image_files:
        frame = cv2.imread(str(image_file))

        # Step 1: Load ground truth
        ground_truth_file = image_file.with_suffix('.txt')
        image_size = frame.shape[:2]
        ground_truth_boxes = load_ground_truth(ground_truth_file, image_size)
        total_ground_truth_cars += len(ground_truth_boxes)
        all_ground_truths.extend(ground_truth_boxes)

        # Step 2: Run YOLO on the current frame
        boxes, scores = run_yolo_on_frame(frame)

        # Filter detections using confidence score threshold
        high_conf_indices = scores > score_threshold
        filtered_boxes = boxes[high_conf_indices]
        filtered_scores = scores[high_conf_indices]

        # Combine boxes with scores for NMS
        detections_with_scores = [list(box) + [score] for box, score in zip(filtered_boxes, filtered_scores)]

        # Step 3: Compute optical flow if there is a previous frame
        if previous_frame is not None:
            motion_mag = compute_optical_flow(previous_frame, frame)

            # Add any detections from optical flow if they match moving objects
            for i, det in enumerate(filtered_boxes):
                x1, y1, x2, y2 = map(int, det[:4])
                if np.mean(motion_mag[y1:y2, x1:x2]) > motion_threshold:
                    detections_with_scores.append(list(det) + [filtered_scores[i]])

        # Step 4: Apply NMS to filter overlapping boxes
        final_boxes = apply_nms(detections_with_scores)
        all_detections.extend(final_boxes)

        # Step 5: Match detections to ground truth
        tp, fp, fn = match_detections_to_ground_truth(final_boxes, ground_truth_boxes, iou_threshold)
        total_true_positives += tp
        total_false_positives += fp
        total_false_negatives += fn

        # Draw ground truth and predictions
        frame_with_ground_truth = frame.copy()
        frame_with_predictions = frame.copy()
        draw_boxes(frame_with_ground_truth, ground_truth_boxes, color=(0, 255, 0), label='GT')
        draw_boxes(frame_with_predictions, final_boxes, color=(0, 0, 255), label='Pred')

        # Display images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(frame_with_ground_truth, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(frame_with_predictions, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Predictions')
        axes[1].axis('off')

        plt.suptitle(f'Comparison for {image_file.name}')
        plt.show()

        # Update previous frame
        previous_frame = frame

    # Compute average precision for IoU threshold 0.5
    average_precision = compute_map(all_detections, all_ground_truths.copy(), iou_threshold=0.2)

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
