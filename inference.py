import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def net():
    num_classes = 133
    model = models.resnet50(pretrained=False)  # Avoid downloading weights
    for param in model.parameters():
        param.requires_grad = False
    num_inputs = model.fc.in_features
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    model.eval()  # Ensure evaluation mode
    return model

def predict_fn(input_data, model):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return {'predicted_class': predicted_class, 'probabilities': probabilities.tolist()}
