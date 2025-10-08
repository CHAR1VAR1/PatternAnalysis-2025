import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders
from modules import AlzheimersClassifier
import matplotlib.pyplot as plt

def train_model(root_dir, epochs=10, batch_size=16, lr=1e-4, device='cuda'):
    """
    Train and evaluate a deep learning model for Alzheimer's disease classification.

    This function performs supervised training of a model using the given training
    data, evaluates it on the test data at each epoch, and reports progress.

    Args:
        root_dir (str): Path to the ADNI dataset root directory.
        epochs (int, optional): Number of training epochs. Default is 10.
        batch_size (int, optional): Number of samples per training batch. Default is 16.
        lr (float, optional): Learning rate for the optimiser. Default is 1e-4.
        device (str, optional): Device to run training on ('cuda' or 'cpu'). Default is "cuda".

    Returns:
        None

    Attributes:
        - Saves the best model's state_dict to "best_model.pth" in the current directory.
        - Generates and saves two plots:
            - "loss_curve.png" for training and validation loss
            - "val_acc_curve.png" for validation accuracy
        - Prints training/validation progress and best validation accuracy to the console.
    """
    train_loader, test_loader = get_dataloaders(root_dir, batch_size=batch_size)

    # Define model, loss, optimiser
    model = AlzheimersClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.5, patience=2, verbose=True)

    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0.0
    epochs_no_improve = 0
    patience = 3

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = correct / total
            val_loss /= len(test_loader)

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Update scheduler
            scheduler.step(val_acc)
            current_lr = optimiser.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            # Save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_model.pth")
                print("New best model saved!")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print("\n Early stopping triggered!")
                break

    # Plot training curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig("val_acc_curve.png")

    print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    root_dir = "/home/groups/comp3710/ADNI"
    train_model(root_dir, epochs=10, batch_size=16, lr=1e-4)