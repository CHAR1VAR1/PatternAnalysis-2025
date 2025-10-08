import os
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
            - prediction (int): Predicted class label value (1 for "AD" or 0 for "NC").
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
    return pred.item(), conf.item()

def evaluate_folder(model, folder_path, label):
    """
    This function iterates through all `.jpeg` files in the specified folder,
    performs inference using the provided model, and counts how many predictions
    match the expected label.

    Args:
        model : The trained PyTorch model used for prediction.
        folder_path (str): Path to the folder containing `.jpeg` images to evaluate.
        label (int): The ground truth label for all images in this folder

    Returns:
        correct (int): Number of correctly classified images.
        total (int): Total number of images evaluated.
        conf_sum (float): Current sum of confidence of classifcation of images in the folder.
    """
    correct = total = conf_sum = 0
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        pred, conf = predict_image(fpath, model)
        total += 1
        correct += int(pred == label)
        conf_sum += conf
    return correct, total, conf_sum

if __name__ == "__main__":
    model = load_model()

    ad_path = "/home/groups/comp3710/ADNI/AD_NC/test/AD"
    nc_path = "/home/groups/comp3710/ADNI/AD_NC/test/NC"
    
    ad_correct, ad_total, ad_conf_sum = evaluate_folder(model, ad_path, label = 1)
    nc_correct, nc_total, nc_conf_sum = evaluate_folder(model, nc_path, label = 0)

    total_correct = ad_correct + nc_correct
    total_images = ad_total + nc_total
    accuracy = total_correct / total_images

    print(f"\n ----- Evaluation Results -----")
    print(f"AD: {ad_correct}/{ad_total} correct ({ad_correct / ad_total * 100:.2f}%)")
    print(f"NC: {nc_correct}/{nc_total} correct ({nc_correct / nc_total * 100:.2f}%)")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"\n ----- Average Confidence Results -----")
    print(f"AD: {ad_conf_sum / ad_total * 100:.2f}%")
    print(f"NC: {nc_conf_sum / nc_total * 100:.2f}%")
    print(f"Total: {(ad_conf_sum + nc_conf_sum) / total_images * 100:.2f}%")