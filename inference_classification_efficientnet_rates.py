import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from shutil import copy2

def save_image(image_path, output_dir, image_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    copy2(image_path, os.path.join(output_dir, image_name))

"""def load_ground_truth_annotations(annotation_dir):
    annotations = {}
    for file_name in os.listdir(annotation_dir):
        if file_name.endswith(".txt"):
            image_name = file_name.replace(".txt", ".jpg")
            with open(os.path.join(annotation_dir, file_name), 'r') as file:
                lines = file.readlines()
                bboxes = [list(map(float, line.strip().split()[1:])) for line in lines]
                annotations[image_name] = bboxes
    return annotations"""

def main():
    # Set the data directories
    test_dir = '/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/cutouts_night/test_classify'
    #annotation_dir = '/Users/jorgemartinez/thesis_retinanet/datasets/annotation_all'
    
    # Define output directories for classified images
    output_dirs = {
        'TP': '/Users/jorgemartinez/thesis_retinanet/datasets/night/TP',
        'TN': '/Users/jorgemartinez/thesis_retinanet/datasets/night/TN',
        'FP': '/Users/jorgemartinez/thesis_retinanet/datasets/night/FP',
        'FN': '/Users/jorgemartinez/thesis_retinanet/datasets/night/FN'
    }

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure all images are resized to 64x64
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(test_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Print dataset details for debugging
    print(f'Number of classes: {len(test_dataset.classes)}')
    print(f'Classes: {test_dataset.classes}')
    print(f'Class to index mapping: {test_dataset.class_to_idx}')
    print(f'Number of test images: {len(test_dataset)}')

    # Verify dataset distribution
    class_counts = {class_name: 0 for class_name in test_dataset.classes}
    for _, label in test_dataset.samples:
        class_counts[test_dataset.classes[label]] += 1
    print(f'Dataset distribution: {class_counts}')

    # Load the trained model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, len(test_dataset.classes))
    model.load_state_dict(torch.load('/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/yolov8_runs/classification/efficientnet_classifier_night_b0.pth', map_location=torch.device('cpu')))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference
    all_preds = []
    all_labels = []
    image_paths = [s[0] for s in test_dataset.samples]  # Extract image paths from the dataset samples

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Print sample predictions and labels for debugging
            print(f'Sample predictions: {preds.cpu().numpy()}')
            print(f'Sample labels: {labels.cpu().numpy()}')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Print the confusion matrix
    print(f'Confusion Matrix:\n{conf_matrix}')

    # Extract metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    # Print metrics
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')

    # Optionally, calculate rates
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

    # Additional checks
    print(f'Total predictions: {len(all_preds)}')
    print(f'Total labels: {len(all_labels)}')
    print(f'Unique predictions: {np.unique(all_preds, return_counts=True)}')
    print(f'Unique labels: {np.unique(all_labels, return_counts=True)}')

    """# Load ground truth annotations
    annotations = load_ground_truth_annotations(annotation_dir)"""

    # Save images to respective directories based on manual TP, TN, FP, FN
    car_index = test_dataset.class_to_idx['car']  # Assuming 'car' is the class name for images with cars

    for i, (pred, label, path) in enumerate(zip(all_preds, all_labels, image_paths)):
        image_name = os.path.basename(path).split('_')[0] + '.jpg'
        if pred == car_index and label == car_index:
            save_image(path, output_dirs['TP'], os.path.basename(path))
        elif pred != car_index and label != car_index:
            save_image(path, output_dirs['TN'], os.path.basename(path))
        elif pred == car_index and label != car_index:
            save_image(path, output_dirs['FP'], os.path.basename(path))
        elif pred != car_index and label == car_index:
            save_image(path, output_dirs['FN'], os.path.basename(path))

    # Manually count TP, TN, FP, FN
    tp_manual = sum((np.array(all_preds) == car_index) & (np.array(all_labels) == car_index))
    tn_manual = sum((np.array(all_preds) != car_index) & (np.array(all_labels) != car_index))
    fp_manual = sum((np.array(all_preds) == car_index) & (np.array(all_labels) != car_index))
    fn_manual = sum((np.array(all_preds) != car_index) & (np.array(all_labels) == car_index))

    print(f'Manual True Positives (TP): {tp_manual}')
    print(f'Manual True Negatives (TN): {tn_manual}')
    print(f'Manual False Positives (FP): {fp_manual}')
    print(f'Manual False Negatives (FN): {fn_manual}')

if __name__ == "__main__":
    main()
