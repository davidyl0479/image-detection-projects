# Face Mask Classification (Transfer Learning)

This project implements a **face mask classification system** using **transfer learning with ResNet18** and PyTorch.

Faces are dynamically cropped using bounding boxes from **PASCAL VOC annotations**, and classified into:
- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

## Key Features
- ResNet18 pretrained on ImageNet (TorchVision multi-weight API)
- Custom Dataset with on-the-fly face cropping
- Train / validation / test split
- Quantitative evaluation (accuracy, F1, confusion matrix)
- Qualitative error analysis
- Clean separation of data, model, and experimentation logic

## Structure
face-mask-tl/
├── data_utils.py # Dataset, preprocessing, dataloaders
├── model_utils.py # Model, training, evaluation
├── main.ipynb # End-to-end pipeline and analysis
└── artifacts/ # Local-only (models, history) – ignored by Git


## Dataset
[Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Notes
- Trained models and artifacts are intentionally **not committed** to GitHub.
- The notebook is designed to run on **Google Colab (GPU)**.