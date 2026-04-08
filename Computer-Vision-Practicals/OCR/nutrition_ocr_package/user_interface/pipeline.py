import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from dataclasses import dataclass


@dataclass
class BoxResult:
    box_idx: int
    cls_id: int
    yolo_conf: float
    roi_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    straightened_rgb: np.ndarray
    ocr_text: str
    ocr_conf: float
    angle: int


def enhance_roi(img_bgr: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    return cv2.addWeighted(clahe_img, 1.5, blurred, -0.5, 0)


def run_doctr(roi_rgb: np.ndarray, doctr_model):
    h, w = roi_rgb.shape[:2]
    if h < 32 or w < 32:
        scale = max(32 / h, 32 / w)
        roi_rgb = np.array(
            Image.fromarray(roi_rgb).resize(
                (max(32, int(w * scale)), max(32, int(h * scale))), Image.BICUBIC
            )
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        Image.fromarray(roi_rgb).save(tmp_path)

    try:
        from doctr.io import DocumentFile
        doc = DocumentFile.from_images(tmp_path)
        result = doctr_model(doc)

        page_orientation = result.pages[0].orientation
        angle = (
            page_orientation.get("value", 0)
            if isinstance(page_orientation, dict)
            else 0
        )

        straightened_rgb = (
            np.array(Image.fromarray(roi_rgb).rotate(angle, expand=True))
            if angle != 0
            else roi_rgb.copy()
        )

        lines_text, confidences = [], []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = []
                    for word in line.words:
                        words.append(word.value)
                        confidences.append(word.confidence)
                    lines_text.append(" ".join(words))

        ocr_text = "\n".join(lines_text) if lines_text else "— no text detected —"
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    finally:
        os.remove(tmp_path)

    return ocr_text, avg_conf, angle, straightened_rgb


def run_pipeline(image_bgr: np.ndarray, yolo_model, doctr_model, conf: float = 0.25):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_bgr.shape[:2]

    results = yolo_model.predict(source=image_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes

    full_vis = image_rgb.copy()
    box_results = []

    if boxes is None or len(boxes) == 0:
        return full_vis, []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(full_vis, (x1, y1), (x2, y2), (34, 197, 94), 3)
        label = f"{float(box.conf[0]):.2f}"
        cv2.putText(full_vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        roi_bgr = image_bgr[y1:y2, x1:x2]
        roi_rgb = image_rgb[y1:y2, x1:x2]

        enhanced_bgr = enhance_roi(roi_bgr)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        ocr_text, ocr_conf, angle, straightened_rgb = run_doctr(enhanced_rgb, doctr_model)

        box_results.append(BoxResult(
            box_idx=i + 1,
            cls_id=int(box.cls[0]),
            yolo_conf=float(box.conf[0]),
            roi_rgb=roi_rgb,
            enhanced_rgb=enhanced_rgb,
            straightened_rgb=straightened_rgb,
            ocr_text=ocr_text,
            ocr_conf=ocr_conf,
            angle=angle,
        ))

    return full_vis, box_results