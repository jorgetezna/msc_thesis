import torch

BATCH_SIZE = 1 # Increase / decrease according to GPU memeory.
RESIZE_TO = 2688 # Resize the image for training and transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 1 # Number of parallel workers for data loading.

DEVICE = torch.device('mps') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = '/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/combined/color/train/overfit'
# Validation images and XML files directory.
VALID_DIR = '/Users/jorgemartinez/thesis_retinanet/RetinaNet_Custom_dataset/data/combined/color/train/overfit'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'car'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = True

# Location to save model and plots.
OUT_DIR = 'outputs'

