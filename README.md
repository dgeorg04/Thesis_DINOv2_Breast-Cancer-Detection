
# Thesis: Breast Cancer Detection Using DINOv2 on Thermal Images

#Dimitriana Georgiou - Department of Computer Science-UCY, 2025.

This repository contains the full code and dataset used in my undergraduate thesis:

**"Leveraging Vision Transformers for Early Breast Cancer Detection: A Study on Thermal Infrared Imaging with the use of DINOv2."**

The project explores the use of **frozen DINOv2 ViT-S/14 features** from thermal breast images to classify patients as **Healthy** or **Unhealthy** using multiple classifier architectures (Linear and MLPs). The experiments prioritize **high sensitivity (≥ 0.90)** and are evaluated using **5-fold stratified cross-validation**.

---
******* DOWNLOAD DATASET ******** 
Due to GitHub’s file size limitations, the full dataset and model checkpoint are hosted on Google Drive.
🔗 https://drive.google.com/file/d/1Dv3M4hG10TV8VrU7HoR2WSuG1OUI-igx/view?usp=sharing
After downloading, extract the contents and place them in your Google Drive in the appropriate folders as expected by the Colab notebook.


## Dataset Overview

###`newdb/`  
This folder contains the **original dataset**, which is a combination of two public datasets:
- **DMR-IR**  
- **TIBS-DB**

Images are organized into two subfolders:
- `Healthy/`
- `Unhealthy/`

---

###`DINOV2_transformed_images/`  
This folder includes the **colorized, resized, and preprocessed** versions of the thermal images used for feature extraction with DINOv2. It follows the same structure:
- `Healthy/`
- `Unhealthy/`

---

###`newdbBalanced/`  
This folder contains a **balanced subset** of the dataset used in additional experiments, where the number of Healthy and Unhealthy images is equalized to test model robustness.
This was solely used for experimental purposes, and it is not included in the code. 
---

## Code and Model Setup

All code is contained in a **single Colab notebook**:
- `Fine_Tuned_DINOv2_Classification.ipynb`

The notebook was run using **Google Colab Pro** to access higher resource limits and prevent mid-execution terminations, especially when handling the full dataset.  
Two runtime types were used based on availability and stability:
- **A100 GPU**
- **L4 GPU**

---

### How to Run the Notebook
After importing the notebook in Google Drive, open the notebook in Google Colab and run the following cell first to mount your Google Drive:

```python
# Mount the drive content
from google.colab import drive
drive.mount('/content/drive')

### Model Used
This project uses the DINOv2 ViT-S/14 model with optional NeCo weights.

To load the base DINOv2 model run the following cell:

'import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')'

To load the pretrained NeCo checkpoint run the following cell: 

'model.load_state_dict(torch.load('path/to/neco_on_dinov2r_vit14_model.ckpt'), strict=False)'

###What the Code Does
The notebook performs the following steps in the correspondind cells (check the 1st comment- title) :

Preprocessing of thermal images (colorization, resizing, normalization)

Feature extraction using frozen DINOv2 ViT-S/14 [CLS] token

Stratified 5-Fold splitting of the dataset (with a fixed final test set)

Training and evaluation of three classifier heads:

1. Linear Classification Head (fully frozen backbone)
2. MLP Classification Head (fully frozen backbone)
3. MLP Classification Head (Partial Fine-Tuning / block 11 unfrozen) 

### Split_Folds Folder
The folder Split_Folds/ contains:

1. folder SPLITS/:
Subfolders for each fold, containing images of the following sets:

Train/

Validate-Test/

Test/

 2. .pt files for each fold:
Pre-extracted DINOv2 features for training the classifiers.


###Summary Steps
Import the notebook (Fine_Tuned_DINOv2_Classification.ipynb) to your Google Drive (included in GitHub repository)

Import the model checkpoint used (neco_on_dinov2r_vit14_model.ckpt) into Drive (use link https://drive.google.com/file/d/1Dv3M4hG10TV8VrU7HoR2WSuG1OUI-igx/view?usp=sharing) 

Import the dataset folder newdb/ into Drive (use link https://drive.google.com/file/d/1Dv3M4hG10TV8VrU7HoR2WSuG1OUI-igx/view?usp=sharing) 

Mount the Drive in Colab using cell:

from google.colab import drive
drive.mount('/content/drive')
Check that the dataset and model exist in the correct Drive paths.

Run the initial cells of the notebook to perform image preprocessing and dataset splitting.

Run the remaining cells to train and evaluate the three classifier heads and obtain the final results.

Note:
Some code cells in the notebook are commented out. These are not required for running the notebook. They were left in place to preserve the development history and experimental steps taken while building the thesis.


Contact
Dimitriana Georgiou
Department of Computer Science,  University of Cyprus
Email: dgeorg04@ucy.ac.cy / Personal Email: DemiGeo3@gmail.com



