import torchvision
import torch

from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

import torchvision.models.detection
import torchvision.models as models
from torchvision.models.detection.retinanet import RetinaNet, retinanet_resnet50_fpn, RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


#def create_model(num_classes=91, checkpoint_path = None):
def create_model(num_classes):
    # Load a pre-trained ResNet-34 model
    backbone = resnet_fpn_backbone('resnet34', pretrained=True, trainable_layers=3,
                                   returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    
    # Define the anchor generator for RetinaNet
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # Define the head of the RetinaNet using the number of expected features from the FPN and the number of classes
    head = RetinaNetHead(
        in_channels=256,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
        num_classes=num_classes)
    
    # Create the RetinaNet model
    model = RetinaNet(backbone, num_classes=num_classes,
                      anchor_generator=anchor_generator)
    
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
