import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse

from PIL import ImageFile
from tqdm import tqdm

from smdebug.pytorch import get_hook
from smdebug.pytorch import modes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

ImageFile.LOAD_TRUNCATED_IMAGES = True

hook = get_hook(create_if_not_exists=True)
logger.info(f"Hook initialised: {hook}")

# Global device variable (set later in main)
DEVICE = torch.device("cpu")

def test(model, test_loader, criterion):
    """
    Evaluate the model on test data.
    """
    model.eval()
    hook.set_mode(modes.EVAL)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def validate(model, valid_loader, criterion):
    """
    Validate the model on validation data.
    """
    model.eval()
    hook.set_mode(modes.EVAL)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Validating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train(model, train_loader, valid_loader, criterion, optimizer, epochs):
    """
    Train the model on training data and evaluate on validation data after each epoch.
    """
    model.train()
    hook.set_mode(modes.TRAIN)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} training loss: {avg_loss:.4f}")
        
        # Run validation after each epoch
        valid_loss, valid_acc = validate(model, valid_loader, criterion)
        logger.info(f"Epoch {epoch+1}/{epochs} validation loss: {valid_loss:.4f}, validation accuracy: {valid_acc:.2f}%")
    return model

def net(num_classes, freeze_pretrained=True):
    """
    Initialise and fine-tune a pretrained model.
    By default, the pretrained layers are frozen and only the final fully connected layer is trained.
    """
    model = models.resnet18(pretrained=True)
    if freeze_pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_data_loaders(train_dir, valid_dir, test_dir, batch_size):
    """
    Create data loaders for training, validation and testing.
    """
    # Training data transformations include augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Validation and test data use centre cropping for consistency
    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def main(args):
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")
    
    # Create data loaders for training, validation and testing
    train_loader, valid_loader, test_loader = create_data_loaders(args.train_data,
                                                                  args.validation_data,
                                                                  args.test_data,
                                                                  args.batch_size)
    
    # Initialise the model for fine-tuning
    model = net(num_classes=args.num_classes, freeze_pretrained=args.freeze_pretrained)
    hook.register_module(model)
    model = model.to(DEVICE)
    
    # Set up loss function and optimiser.
    # Only parameters that require gradients are updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Train the model, with validation after each epoch
    model = train(model, train_loader, valid_loader, criterion, optimizer, args.epochs)
    
    # Evaluate on test data
    test_loss, test_acc = test(model, test_loader, criterion)
    
    # Save the trained model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=133, help='Number of classes')
    parser.add_argument('--freeze-pretrained', type=bool, default=True,
                        help='Freeze pretrained layers (only train the final layer)')
    parser.add_argument('--train-data', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data/train'),
                        help='Directory for training data')
    parser.add_argument('--validation-data', type=str,
                        default=os.environ.get('SM_CHANNEL_VALIDATION', 'data/validation'),
                        help='Directory for validation data')
    parser.add_argument('--test-data', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST', 'data/test'),
                        help='Directory for testing data')
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', 'model'),
                        help='Directory where the model will be saved')
    
    args = parser.parse_args()
    
    main(args)
