import os
import torch
from torchvision import transforms
from PIL import Image
from modules import AlzheimersClassifier
from collections import defaultdict
import numpy as np

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

def predict_slice(image_path, model):
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
        transforms.Normalize(mean=[0.1159], std=[0.2199])
    ])
    image = transform(image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return probs

def aggregate_patient_predictions(patient_probs):
    """
    Given a list of [NC_prob, AD_prob] arrays for one patient,
    average them and return predicted label + confidence.

    Args:
        patient_probs (dict): probabilities of AD case for all slices of one patient.

    Returns:
        tuple:
            - label (str): Predicted class label value (1 for "AD" or 0 for "NC").
            - confidence (float): Model confidence for the predicted class,
              between 0.0 and 1.0.
            - mean_probs (int): mean probability of AD case across aggregated slices
              for a patient.
    """
    mean_probs = np.mean(patient_probs, axis=0)
    label = np.argmax(mean_probs)
    confidence = mean_probs[label]

    return label, confidence, mean_probs

if __name__ == "__main__":
    model = load_model()
    root_dir = "/home/groups/comp3710/ADNI/AD_NC/test"
    class_names = ["NC", "AD"]

    # Group all slices by patient id
    patient_slices = defaultdict(list)
    for cls in ["AD", "NC"]:
        folder = os.path.join(root_dir, cls)
        for fname in os.listdir(folder):
            patient_id = fname.split('_')[0]
            patient_slices[patient_id].append(os.path.join(folder, fname))

    # Predict each slice and aggregate
    patient_results = {}
    for pid, slice_paths in patient_slices.items():
        slice_probs = [predict_slice(p, model) for p in slice_paths]
        label, conf, mean_probs = aggregate_patient_predictions(slice_probs)
        patient_results[pid] = (label, conf, mean_probs)

    # Evaluate patient accuracy
    correct, total = 0, 0
    for pid, (label, conf, probs) in patient_results.items():
        true_label = 1 if any("AD/" in p for p in patient_slices[pid]) else 0
        total += 1
        correct += int(label == true_label)

        print(f"Patient {pid}: Predicted {class_names[label]} ({conf * 100:.2f}%)"
              f" | True: {class_names[true_label]}")

    print(f"\nPatient Prediction Accuracy: {100 * correct / total:.2f}%")