import torch
from torchvision import transforms
from PIL import Image
from modules import AlzheimersClassifier

def load_model(model_path="best_model.pth"):
    """
    This function initializes an instance of the AlzheimersClassifier model,
    loads the trained weights from the specified checkpoint file, and sets
    the model to evaluation mode on the appropriate device (CPU or GPU).

    Args:
        model_path (str, optional): Path to the saved model checkpoint (.pth file).
                                   Defaults to the value of "best_model.pth".

    Returns:
        AlzheimersClassifier:
            A ConvNeXt-based PyTorch model ready for inference.
    """
    model = AlzheimersClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.to("cuda")
    model.eval()
    return model

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
    image = Image.open(image_path)

    # Apply transform to image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    class_names = ["NC", "AD"]
    label = class_names[pred.item()]
    print(f"Prediction: {label} ({conf.item() * 100:.2f}% confidence)")
    return label, conf.item()

if __name__ == "__main__":
    model = load_model()
    test_image_path = "/home/groups/comp3710/ADNI/AD_NC/test/AD/388206_78.jpeg"
    predict_image(test_image_path, model)