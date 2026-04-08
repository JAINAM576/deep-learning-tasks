import streamlit as st
from ultralytics import YOLO
from doctr.models import ocr_predictor


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo(weights_path: str):
    return YOLO(weights_path)


@st.cache_resource(show_spinner="Loading docTR OCR model...")
def load_doctr():
    return ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=False,
        detect_orientation=True,
        straighten_pages=True,
    )