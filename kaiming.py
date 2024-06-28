import torch
import os
import random
from torch import nn
from ultralytics import YOLO
import numpy as np

NUM_THREADS = 8

def safe_seed_worker(worker_id):
    """Override the default `seed_worker` function with a safer version."""
    try:
        os.sched_setaffinity(0, range(NUM_THREADS))
    except (AttributeError, OSError) as e:
        print(f"Warning: Unable to set CPU affinity ({e}).")

    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def kaiming_initialization(layer):
    """Applies Kaiming initialization to the given layer."""
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

def initialize_yolov8_with_kaiming(model_name='yolov8n.yaml'):
    """Initializes a YOLOv8 model with Kaiming initialization."""
    # Load the model architecture
    model = YOLO(model_name)

    # Apply Kaiming initialization
    model.model.apply(kaiming_initialization)

    return model

import ultralytics.data.build as data_build
data_build.seed_worker = safe_seed_worker

#pretrained_weights = 'yolov8n.pt'

yolo_model = initialize_yolov8_with_kaiming('yolov8n.yaml')
#yolo_model = YOLO(pretrained_weights)
print(yolo_model.model)

from torch.optim.lr_scheduler import StepLR
optimizer = torch.optim.Adam(yolo_model.model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

yolo_model.train(
    data = '/zhome/fa/a/163291/msc_thesis/data/Vehicle_detection_from_satellite.v1i.yolov5pytorch/data.yaml',
    epochs = 500,
    patience = 50,
    lr0 = 0.001,
    lrf = 0.01,
    weight_decay = 0.0005,
    optimizer = 'Adam',
    imgsz = 60,
    workers = 4,
    batch = 64,
    project = '/zhome/fa/a/163291/msc_thesis/yolov8/runs',
    name = 'kaiming_new2_public1',
    exist_ok = True
)

/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/yolov5/kaiming.py