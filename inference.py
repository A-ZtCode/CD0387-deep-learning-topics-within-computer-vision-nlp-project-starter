import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models
import nvgpu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def Net():
    # Use ResNet18 to match the training model
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_ftrs = model.fc.in_features
    # Replace the final layer to output 133 classes
    model.fc = nn.Linear(num_ftrs, 133)
    return model


def model_fn(model_dir):
    device = "cpu"
    logger.info(f"Device: {device}")

    model = net(device)

    logger.info("Loading model weights")

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()

    return model