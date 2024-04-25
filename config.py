import torch

BATCH_SIZE = 16 # Increase / decrease according to GPU memeory.
RESIZE_TO = 950 # Resize the image for training and transforms.
NUM_EPOCHS = 1000 # Number of epochs to train for.
NUM_WORKERS = 8 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = 'data/vehicle-detection-from-satellite-images-data-set/train'
# Validation images and XML files directory.
VALID_DIR = 'data/vehicle-detection-from-satellite-images-data-set/valid'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'car'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'

