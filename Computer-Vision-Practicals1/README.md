# Computer Vision Practical New

End-to-end computer vision workspace covering:
- Image Classification (EfficientNet-B3, ViT)
- Semantic Segmentation (U-Net, SegFormer)
- Object Detection (YOLO on VisDrone)
- OCR Pipeline (YOLO + docTR + Streamlit + MinerU)

This README is structured for GitHub browsing, with a compact project map, folder-by-folder analysis, model/output inventory, and runtime notes.

## Quick Navigation
- `Classification/`: satellite image classification experiments and metrics
- `Image_segmentation/`: road damage segmentation inference and outputs
- `Object_Detection/`: VisDrone object detection training artifacts
- `OCR/`: nutrition label detection + OCR app and pipeline
- `tasks-to-done.ipynb`: planned next experiments and dataset links

## Project Structure (Accessibility-Friendly)
Note: Large media-heavy folders (many images/video frames) are intentionally summarized for readability.

```text
computer-vision-practical-new/
|- tasks-to-done.ipynb
|- Classification/
|  |- classification-using-vit.ipynb
|  |- EfficientNet-B3/
|  |  |- efficientnet_b3_full.pth
|  |  |- training_metrics.csv
|  |  |- confusion_matrix.csv
|  |  |- classification_report.csv
|  |- VIT/
|  |  |- vit_full.pth
|  |  |- training_metrics.csv
|  |  |- confusion_matrix.csv
|  |  |- classification_report.csv
|- Image_segmentation/
|  |- inference.py
|  |- inference_segformer.py
|  |- README.txt
|  |- model/unet_model.pth
|  |- logs/unet_experiment/version_0/metrics.csv
|  |- logs/unet_experiment/version_0/checkpoints/*.ckpt
|  |- data_sample/
|  |- prediction_outputs/
|  |- code/
|  |- SegFormer/
|  |  |- segmentation-using-segformer.ipynb
|  |  |- README.txt
|  |  |- model/unet_model.pth
|  |  |- logs/segformer/version_*/
|  |  |- code/
|  |  |- data_sample/
|- Object_Detection/
|  |- requirements.txt
|  |- config/visdrone.yaml
|  |- notebook/visdrone-object-detection-yolov26.ipynb
|  |- models/best.pt
|  |- results/results.csv
|  |- results/predictions/
|  |- .git/ (nested git repo)
|- OCR/
|  |- min.py
|  |- OCR_PIPELINE.ipynb
|  |- nutrition_ocr_package/
|  |  |- README.md
|  |  |- data.yaml
|  |  |- best.pt
|  |  |- runs/detect/exp1/run1/
|  |  |  |- args.yaml
|  |  |  |- results.csv
|  |  |  |- weights/*.pt
|  |  |- test_images/
|  |  |- user_interface/
|  |  |  |- app.py
|  |  |  |- models.py
|  |  |  |- pipeline.py
```

## File Inventory Snapshot
Counts detected while scanning this folder:
- `*.ipynb`: 5
- `*.py`: 6
- `*.pt`: 14
- `*.pth`: 4
- `*.ckpt`: 3
- `*.csv`: 10
- `*.yaml`: 3
- `*.txt`: 3
- `*.md`: 1

## Full Folder Analysis

### 1) `Classification/`
Purpose:
- Multi-class image classification benchmarking between EfficientNet-B3 and ViT.

Key files:
- `Classification/classification-using-vit.ipynb`: Kaggle notebook for training/evaluation.
- `Classification/EfficientNet-B3/training_metrics.csv`: epoch-wise loss/accuracy.
- `Classification/VIT/training_metrics.csv`: epoch-wise loss/accuracy.
- `Classification/*/*.pth`: trained model weights.

Observed metrics summary:
- EfficientNet-B3 (`Classification/EfficientNet-B3/training_metrics.csv`):
  - Train Acc: `80.76 -> 88.62`
  - Val Acc: `89.39 -> 91.69`
- ViT (`Classification/VIT/training_metrics.csv`):
  - Train Acc: `92.74 -> 99.46`
  - Val Acc: `94.44 -> 97.52`


### 2) `Image_segmentation/`
Purpose:
- Road damage segmentation (background/pothole/crack/manhole) with U-Net and SegFormer.

Key files:
- `Image_segmentation/inference.py`: U-Net inference on video.
- `Image_segmentation/inference_segformer.py`: SegFormer inference on video.
- `Image_segmentation/model/unet_model.pth`: U-Net weights.
- `Image_segmentation/logs/unet_experiment/version_0/metrics.csv`: training/validation logs.
- `Image_segmentation/SegFormer/segmentation-using-segformer.ipynb`: SegFormer training notebook.

Implementation notes:
- Both inference scripts do GPU check via `torch.cuda.is_available()`.
- Both scripts overlay class masks and live class-percentage stats on frames.
- SegFormer script uses `transformers` (`SegformerForSemanticSegmentation`, `SegformerImageProcessor`).


### 3) `Object_Detection/`
Purpose:
- Drone scene object detection using YOLO with VisDrone classes.

Key files:
- `Object_Detection/notebook/visdrone-object-detection-yolov26.ipynb`: training notebook.
- `Object_Detection/config/visdrone.yaml`: 10-class mapping + Kaggle dataset path.
- `Object_Detection/models/best.pt`: trained detector.
- `Object_Detection/results/results.csv`: full epoch metrics.

Dataset/classes (`Object_Detection/config/visdrone.yaml`):
- `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`
- `path: /kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset`

Training progression (`Object_Detection/results/results.csv`):
- `metrics/mAP50(B)` improves from about `0.1146` (epoch 1) to about `0.3343` (epoch 50).

Notes:
- `Object_Detection/requirements.txt` is an environment export (very large, includes many packages not strictly required for this project).
- Contains nested `.git/` history.


### 4) `OCR/`
Purpose:
- Nutrition-label detection and text extraction pipeline.

Key files:
- `OCR/OCR_PIPELINE.ipynb`: OCR experimentation notebook and pipeline workflow.
- `OCR/nutrition_ocr_package/user_interface/app.py`: Streamlit app.
- `OCR/nutrition_ocr_package/user_interface/models.py`: cached YOLO + docTR loaders.
- `OCR/nutrition_ocr_package/user_interface/pipeline.py`: preprocessing + OCR pipeline.
- `OCR/nutrition_ocr_package/best.pt`: YOLO detector weights.
- `OCR/nutrition_ocr_package/runs/detect/exp1/run1/results.csv`: training metrics.
- `OCR/min.py`: MinerU quick OCR script.

Pipeline (from `pipeline.py` and `README.md`):
- Detect label ROI with YOLO.
- Preprocess ROI: denoise + CLAHE + sharpening.
- OCR with docTR (`db_resnet50` + `crnn_vgg16_bn`) and orientation correction.
- Optional MinerU extraction script (`OCR/min.py`).


### 5) Root utility files
- `tasks-to-done.ipynb`: roadmap notebook referencing multiple Kaggle datasets/tasks.
- `3.8.0`, `4.9.0`: version-marker files (kept as-is).

## Runtime Environment Evidence (Kaggle/Colab/GPU)
Explicit indicators found while analyzing files:
- Kaggle notebook metadata and paths in:
  - `Classification/classification-using-vit.ipynb`
  - `Object_Detection/notebook/visdrone-object-detection-yolov26.ipynb`
  - `Object_Detection/config/visdrone.yaml`
- Colab/GPU signals in:
  - `Image_segmentation/SegFormer/segmentation-using-segformer.ipynb`
  - `OCR/nutrition_ocr_package/README.md` (T4 mention)
  - `Object_Detection/requirements.txt` (`google-colab`, CUDA packages)
- Direct CUDA runtime checks in:
  - `Image_segmentation/inference.py`
  - `Image_segmentation/inference_segformer.py`

Summary:
- Some tasks were run on Kaggle GPU environments.
- Some tasks were run on Colab GPU environments.
- Local folder stores exported artifacts (weights, logs, metrics, predictions).

## Libraries Used Across This Folder
The consolidated dependency list is in root `requirements.txt`.
Sources considered:
- All `.py` files in this folder
- All code cells in all `.ipynb` files in this folder

## Repro Steps (Quick)
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run by module:
   - Segmentation inference: `Image_segmentation/inference.py` or `Image_segmentation/inference_segformer.py`
   - OCR app: `streamlit run OCR/nutrition_ocr_package/user_interface/app.py`
   - Notebooks: open `.ipynb` files and adjust dataset/model paths if needed

