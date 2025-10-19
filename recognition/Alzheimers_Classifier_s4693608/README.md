# Alzheimer’s Disease MRI Classifier — COMP3710 Task 8

This project implements a deep learning classifier to distinguish between Alzheimer’s Disease (AD) and Normal Control (NC) using 2D MRI brain slices from the ADNI dataset.  
The model is based on ConvNeXt, trained using PyTorch, and makes predictions on the slice-level and then aggregates through these predictions and averages them to make patient-level predictions.
The model was trained and tested on UQ's rangpur, achieving a final patient prediction accuracy of 80.22%.

Developed for COMP3710 — Pattern Recognition and Analysis, University of Queensland by Christian Vever - s4693608.

## Dataset Structure

/home/groups/comp3710/ADNI/
├── AD_NC/
│   ├── train/
│   │   ├── AD/
│   │   │   ├── `<patientID>_<slice>.jpeg`
│   │   └── NC/
│   │       ├── `<patientID>_<slice>.jpeg`
│   ├── test/
│   │   ├── AD/
│   │   └── NC/
└── meta_data_with_label.json

All images are greyscale JPEGs of size (256 x 240), and are 2D slices of MRI brain scans form the ADNI dataset.
Filenames follow `<patientID>_<slice>.jpeg` naming convention.

## Requirements

### System Requirments

* Python 3.10 +
* Parallel processing using GPU (optional but recommended)

### Dependency Requirements

* torch
* torchvision
* timm
* pillow
* numpy

## Training Model Parameters

* Model: ConvNeXt (pretrained on ImageNet)
* Epochs: 30
* Batch Size: 32 (optimal tested number of batches)
* Learning Rate: 1e-4
* Scheduler: OneCycleLR (allows for increased learnin rate over first few epochs before decreasing)
* Dropout: Enabled in classifier head (set to 0.5; prevents overfitting)
* Loss: Weighted CrossEntropyLoss (weigth = [2.0, 1.0] to focus more on AD cases)
* Augmentation: Random rotation +/- 10 degrees and horizontal flip (to force the classifier to focus more on general features)
* Normalisation: mean = 0.1159, std = 0.2199 (the calculated mean and std of the dataset)

The script saves the best checkpoint to `best_model.pth`.

## Running Scripts and Outputs

Scripts were run on rangpur using sbatcb.
sbatch runner for train.py:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --job-name=train
#SBATCH --time=04:00:00
#SBATCH -o train.out
#SBATCH -e train.err

conda activate torch
python train.py
```

The train.py script outputs (per epoch):
* The current epoch number (out of total epochs)
* The training loss for this epoch
* The validation loss for this epoch
* The validation accuracy for this epoch
* The current learning rate
Example output:
```
Epoch 7/30 | Train Loss: 0.0914 | Val Loss: 0.6345 | Val Acc: 0.7551 | LR: 0.000447
```
Once all epochs have been run, the script outputs the best validation accuracy.

sbatch runner for predict.py:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=predict
#SBATCH -o predict.out
#SBATCH -e predict.err

conda activate torch
python predict.py
```

The predict.py script outputs:
* The patiend id
* The predicted case (AD or NC)
* The true case (AD or NC)
* The patient prediction accuracy across all slices
Example output:
```
Patient 389298: Predicted AD (99.99%) | True: AD
```
Once all patient id's have been tested, the script outputs the overall patient prediction accuracy.