import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import logging
import sys
import time

from PIL import ImageFile

# Configure PIL to handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, batch_size):
    """
    Evaluates the performance of the model on the test dataset.

    This function sets the model to evaluation mode, disables gradient computation,
    and iterates over the test data loader to compute the loss and accuracy.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function to evaluate the model.
        device (torch.device): Device on which to perform computations.
        batch_size (int): Number of samples per batch.

    Returns:
        None
    """
    logger.info("Testing started!")
    model.eval()
    running_losses = []
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Accumulate loss and correct predictions
            running_losses.append(loss.item())
            running_corrects += torch.sum(preds == labels.data)

    # Calculate average loss and accuracy
    total_loss = sum(running_losses) / len(running_losses)
    total_acc = running_corrects.double().item() / len(test_loader.dataset)

    logger.info(f"Testing Loss: {total_loss:.4f}")
    logger.info(f"Testing Accuracy: {total_acc:.4f}")


def train(model, train_loader, validation_loader, criterion, optimizer, device, batch_size, hook=None):
    """
    Trains the neural network model using the provided training and validation data loaders.

    This function handles the training loop, including forward and backward passes,
    loss computation, optimizer steps, and validation. It also implements early stopping
    based on validation loss improvements.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        validation_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device on which to perform computations.
        batch_size (int): Number of samples per batch.
        hook (callable, optional): Optional debugging or profiling hook.

    Returns:
        torch.nn.Module: The trained model with updated parameters.
    """
    logger.info("Training started!")
    epochs = 2  # Only two epochs for tuning job
    best_loss = float('inf')
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            dataset_length = len(image_dataset[phase].dataset)
            if phase == 'train':
                model.train()
                grad_enabled = True
            else:
                model.eval()
                grad_enabled = False

            running_losses = []
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                # Move inputs and labels to the specified device
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(grad_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                _, preds = torch.max(outputs, 1)

                # Accumulate loss and correct predictions
                running_losses.append(loss.item())
                running_corrects += torch.sum(preds == labels.data)

                processed_images_count = batch_idx * batch_size + len(inputs)
                logger.info(
                    f"{phase} epoch: {epoch} [{processed_images_count}/{dataset_length} "
                    f"({100.0 * processed_images_count / dataset_length:.0f}%)]\tLoss: {loss.item():.6f}"
                )

            # Calculate average loss and accuracy for the epoch
            epoch_loss = sum(running_losses) / len(running_losses)
            epoch_acc = running_corrects.double().item() / len(image_dataset[phase].dataset)

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1  # Early Stopping if validation loss gets worse

            logger.info(
                f"{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, best validation loss: {best_loss:.4f}"
            )

        if loss_counter == 1:  # Early Stopping
            logger.info("Early stopping triggered.")
            break

    return model


def net():
    """
    Initializes and returns a pre-trained ResNet-50 model customized for the specific task.

    This function loads a ResNet-50 model pre-trained on ImageNet, freezes its parameters to prevent
    them from being updated during training, and replaces the final fully connected layer to match
    the desired output size.

    Returns:
        torch.nn.Module: The customized ResNet-50 model.
    """
    output_size = 133
    model = models.resnet50(pretrained=True)

    # Freeze all parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, output_size)
    )
    return model


def create_data_loaders(data, batch_size):
    """
    Creates and returns data loaders for training, testing, and validation datasets.

    This function applies appropriate transformations to the datasets and initializes
    DataLoaders with the specified batch size.

    Args:
        data (str): Root directory containing 'train', 'test', and 'valid' subdirectories.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: A tuple containing DataLoaders for training, testing, and validation datasets.
    """
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    """
    The main function orchestrates the training and testing of the neural network model.

    It initializes the model, configures the device for computation, sets up the loss function
    and optimizer, creates data loaders, and manages the training and testing processes.
    Finally, it saves the trained model to the specified directory.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing hyperparameters and paths.

    Returns:
        None
    """
    # Initialize the model
    model = net()

    # Configure device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multiple_gpus_exist = torch.cuda.device_count() > 1
    if multiple_gpus_exist:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    logger.info(f"Transferring Model to Device {device}")
    model = model.to(device)

    # Set up loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(
        model.module.fc.parameters() if multiple_gpus_exist else model.fc.parameters(),
        lr=args.learning_rate
    )

    # Create data loaders
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)

    # Train the model
    start_time = time.time()
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args.batch_size)
    logger.info(f"Training time: {round(time.time() - start_time, 2)} seconds.")

    # Test the model
    start_time = time.time()
    test(model, test_loader, loss_criterion, device, args.batch_size)
    logger.info(f"Testing time: {round(time.time() - start_time, 2)} seconds.")

    # Save the trained model
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network model.")

    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training and evaluation.')

    # Directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'),
                        help='Path to the training data directory.')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'),
                        help='Directory to save the trained model.')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'),
                        help='Directory for output data.')

    args = parser.parse_args()

    main(args)
