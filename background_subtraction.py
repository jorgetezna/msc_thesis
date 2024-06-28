import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append((filename, img))
    return images

def load_ground_truth(ground_truth_path, image_size):
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
        pass
    return boxes

def get_contour_detections(mask, thresh=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > thresh:
            detections.append([x, y, x + w, y + h, area])
    return np.array(detections)

def compute_iou(box1, box2):
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

    if union_area == 0:
        return 0
    return inter_area / union_area

def match_detections_to_ground_truth(detections, ground_truths, iou_threshold=0.0):
    tp = 0
    fp = 0
    fn = 0

    detected_gt = [0] * len(ground_truths)

    for detection in detections:
        detection_box = detection[:4]
        match_found = False
        for i, gt in enumerate(ground_truths):
            if compute_iou(detection_box, gt) > iou_threshold:
                tp += 1
                detected_gt[i] = 1
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = len(ground_truths) - sum(detected_gt)
    return tp, fp, fn

def compute_average_precision(detections, ground_truths, iou_threshold=0.5):
    precisions = []
    recalls = []
    thresholds = np.arange(0, 1.1, 0.1)

    for t in thresholds:
        filtered_detections = [d for d in detections if len(d) >= 5 and d[4] >= t]
        tp, fp, fn = match_detections_to_ground_truth(filtered_detections, ground_truths, iou_threshold)

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

def evaluate_performance(detections, ground_truths):
    tp, fp, fn = match_detections_to_ground_truth(detections, ground_truths, iou_threshold=0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, tp, fp, fn, 0

def cluster_and_merge_boxes(boxes, eps=30, min_samples=1):
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

def main():
    image_folder = '/Users/jorgemartinez/thesis_retinanet/datasets/west_easy_cars_no_padding_yolo/all/test_new'
    image_files = sorted(Path(image_folder).glob('*.jpg'))  # Load all images in the folder

    back_sub = cv2.createBackgroundSubtractorMOG2()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_aps = []

    for i in range(1, len(image_files)):
        image_file1 = image_files[i - 1]
        image_file2 = image_files[i]
        filename1 = str(image_file1)
        filename2 = str(image_file2)
        print(f"Processing {filename1} and {filename2}...")

        image1 = cv2.imread(filename1)
        image2 = cv2.imread(filename2)

        if image1 is None or image2 is None:
            print(f"Could not open one of the images: {filename1} or {filename2}")
            continue

        # Apply background subtraction
        fg_mask1 = back_sub.apply(image1)
        fg_mask2 = back_sub.apply(image2)
        mask = cv2.bitwise_and(fg_mask1, fg_mask2)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        detections = get_contour_detections(mask, thresh=500)

        # Separate bboxes and scores
        if len(detections) == 0:
            print(f"No detections found for {filename2}")
            continue

        bboxes = detections[:, :4]
        scores = detections[:, -1]

        clustered_detections = cluster_and_merge_boxes(detections, eps=30, min_samples=1)

        # Load ground truth
        ground_truth_file = str(image_file2).replace('.jpg', '.txt')
        ground_truth_boxes = load_ground_truth(ground_truth_file, image2.shape[:2])

        print(f"Detections: {detections}")
        print(f"Ground Truth: {ground_truth_boxes}")
        print(f"Clustered Detections: {clustered_detections}")

        # Calculate metrics
        precision, recall, f1, tp, fp, fn, tn = evaluate_performance(
            clustered_detections,
            ground_truth_boxes
        )
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        ap = compute_average_precision(detections, ground_truth_boxes, iou_threshold=0.5)
        all_aps.append(ap)

        # Draw bounding boxes on the image
        for box in clustered_detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for box in ground_truth_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image2, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the mask and the original image with detections
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.title(f'Detections for {filename2}')
        plt.show()

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}, Total TN: {total_tn}")
    print(f"Average Precision: {np.mean(all_precisions):.4f}")
    print(f"Average Recall: {np.mean(all_recalls):.4f}")
    print(f"Average F1 Score: {np.mean(all_f1s):.4f}")
    print(f"Average AP@0.5: {np.mean(all_aps):.4f}")

if __name__ == "__main__":
    main()
