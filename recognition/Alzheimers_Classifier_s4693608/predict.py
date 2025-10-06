import torch
from torchvision import transforms
from PIL import Image
from modules import AlzheimersClassifier

MODEL_PATH = "best_model.pth"

def load_model(model_path=MODEL_PATH):
    """
    This function initializes an instance of the AlzheimersClassifier model,
    loads the trained weights from the specified checkpoint file, and sets
    the model to evaluation mode on the appropriate device (CPU or GPU).

    Args:
        model_path (str, optional): Path to the saved model checkpoint (.pth file).
                                   Defaults to the value of MODEL_PATH.

    Returns:
        AlzheimersClassifier:
            A ConvNeXt-based PyTorch model ready for inference.
    """

def predict_image(image_path, model):
    """
    The function loads a grayscale MRI slice, applies the same preprocessing
    transformations used during training (resize, tensor conversion, normalization),
    and performs a forward pass through the trained model to obtain the predicted
    class and associated confidence score.

    Args:
        image_path (str): Path to the input image (e.g., .jpeg slice).
        model (torch.nn.Module): Trained AlzheimersClassifier model instance.

    Returns:
        tuple:
            - label (str): Predicted class label ("AD" or "NC").
            - confidence (float): Model confidence for the predicted class,
                                  between 0.0 and 1.0.
    """