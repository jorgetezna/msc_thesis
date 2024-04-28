import torchvision
import torch

from functools import partial
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator

#def create_model(num_classes=91, checkpoint_path = None):
def create_model(num_classes=91):
    # Load the pretrained RetinaNet model
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained=True, weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)


    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (48,), (64,), (96,)),  # A tuple of tuples, each corresponds to a feature map level
        aspect_ratios=((0.5, 1.0, 1.5, 2.0),) * 5   # Repeat the aspect ratio tuple for each feature map level
    )

    # Apply the custom anchor generator to the model
    model.anchor_generator = anchor_generator

    # Number of anchors calculated from the custom anchor generator
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    # Replace the classification and regression heads
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,  # Typically 256 for RetinaNet heads
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    model.head.regression_head = RetinaNetRegressionHead(
        in_channels=256,
        num_anchors=num_anchors,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )


    """if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)"""
    return model

if __name__ == '__main__':
    model = create_model(1)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
