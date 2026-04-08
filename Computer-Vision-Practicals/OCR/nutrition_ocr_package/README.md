
# Nutrition Label OCR Pipeline
=====================================

## Folder Structure
nutrition_ocr_package/
├── best.pt                  ← Trained YOLO model weights
├── data.yaml                ← Class names & dataset config
├── runs/
│   └── exp1/run1/
│       └── weights/
│           └── best.pt      ← Same weights (original location)
├── test_images/             ← Sample test images
│   └── *.jpg
└── README.md                ← This file

## Inference Pipeline (applied at inference time)
==================================================

### Step 1 — Image Enhancement (applied before YOLO)
  1. Denoise   : cv2.fastNlMeansDenoisingColored  h=7, hColor=7
  2. CLAHE     : clipLimit=2.0, tileGridSize=(8,8) on L channel (LAB space)
  3. Sharpen   : Unsharp Mask — GaussianBlur 3x3, strength=0.5

### Step 2 — Object Detection (YOLO)
  - Model     : YOLOv8 custom trained on nutrition labels
  - Weights   : best.pt
  - Conf      : 0.25
  - Input     : Enhanced image

### Step 3 — ROI Crop
  - Crop each detected bounding box from the image
  - Clamp coordinates to image bounds

### Step 4 — docTR OCR
  - det_arch              : db_resnet50
  - reco_arch             : crnn_vgg16_bn
  - detect_orientation    : True   (auto detects rotation)
  - straighten_pages      : True   (auto corrects rotation)
  - assume_straight_pages : False

## Install Requirements
=======================
  pip install ultralytics
  pip install opencv-python
  pip install Pillow numpy matplotlib
  pip install python-doctr[torch]
  pip install torch torchvision

## Quick Inference Code
=======================
  from ultralytics import YOLO
  model = YOLO("best.pt")
  results = model.predict(source="your_image.jpg", conf=0.25)

## Training Info
================
  Dataset     : Labels-1-augmented-enhanced
  Augmentation: Rotation 0/90/180/270° + Denoise + CLAHE + Sharpen
  Splits      : train / valid / test
  GPU         : T4 (Google Colab)
