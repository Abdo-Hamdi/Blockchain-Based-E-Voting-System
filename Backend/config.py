"""
Configuration Module

This module contains all configuration settings, model initialization, and constants
for the blockchain-based e-voting system backend.

@title Configuration Settings
@version 1.0.0
@description Centralized configuration for face verification and ID processing
"""

import os
import warnings
import torch
import logging
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, request, jsonify
import tempfile
import matplotlib.pyplot as plt
import cv2
import easyocr
import tensorflow as tf
from skimage.feature import local_binary_pattern

# Logging Configuration

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy third-party loggers
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('easyocr').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Suppress TensorFlow info logs but keep warnings and errors
tf.get_logger().setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR

# Keep application logs visible
logging.getLogger('__main__').setLevel(logging.INFO)
logging.getLogger('data_extraction').setLevel(logging.INFO)
logging.getLogger('face_auth').setLevel(logging.INFO)

# Device Configuration

# @notice Device selection for model inference
# @dev Automatically selects CUDA if available, otherwise falls back to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# File and Image Processing

# @notice Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# @notice Minimum image size for processing
MIN_IMAGE_SIZE = 100

# @notice Target image size for face embedding extraction
IMAGE_SIZE = 160

# @notice ID number class identifier in YOLO model
ID_NUMBER_CLASS = 6

# Face Verification Settings

# @notice Similarity threshold for face matching
# @dev Faces with similarity >= this threshold are considered a match
SIMILARITY_THRESHOLD = 0.85

# Fake Detection Settings

# @notice Threshold for determining real vs fake IDs
# @dev Scores above this threshold indicate a real ID
REAL_THRESHOLD = 0.75

# @notice JPEG quality for Error Level Analysis (ELA)
ELA_QUALITY = 85

# @notice Scale factor for ELA difference enhancement
ELA_SCALE_FACTOR = 15

# @notice Local Binary Pattern (LBP) parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# Flask Server Settings

# @notice Flask server host configuration
HOST = '0.0.0.0'

# @notice Flask server port configuration
PORT = 5000

# @notice Flask debug mode setting
DEBUG = False

# @notice Swagger API documentation configuration
SWAGGER = {
    'title': 'Face Authentication API',
    'uiversion': 3,
    'specs_route': '/apidocs/'
}

# OCR and Text Processing

# @notice Digit mapping for Arabic/Persian to English conversion
# @dev Maps various digit representations to English digits
DIGIT_MAP = {
    # Arabic-Indic digits
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    # Persian digits
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    # English digits (passthrough)
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
}

# Model Initialization

# @notice Get project directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# @notice Initialize MTCNN face detector
# @dev Configured for optimal face detection in ID verification scenarios
mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=14,
    device=DEVICE,
    selection_method='center_weighted_size',
    min_face_size=40
)

# @notice Initialize InceptionResnetV1 face embedding model
# @dev Pre-trained on VGGFace2 dataset for face recognition
INCEPTION_MODEL = InceptionResnetV1(
    classify=False, 
    pretrained='vggface2'
).to(DEVICE).eval()

# @notice Initialize YOLO model for ID detection in images
# @dev Detects ID card boundaries within uploaded images
YOLO_MODEL_1 = YOLO(
    os.path.join(project_root, "Models", "ID Detection & Field Extraction", "detect_id_inside_image.pt")
)

# @notice Initialize YOLO model for field extraction from ID cards
# @dev Detects specific fields (like ID number) within cropped ID cards
YOLO_MODEL_2 = YOLO(
    os.path.join(project_root, "Models", "ID Detection & Field Extraction", "detect_field_inside_id.pt")
)

# @notice Initialize fake detection model
# @dev TensorFlow model for distinguishing real vs fake ID cards
FAKE_MODEL = tf.keras.models.load_model(
    os.path.join(project_root, "Models", "ID Fake Detection", "Fake_model_best.keras"), 
    compile=False
)

# @notice Initialize EasyOCR reader
# @dev Configured for Arabic and English text recognition
READER = easyocr.Reader(['ar', 'en'])

# Configuration Validation

def validate_configuration():
    """
    @notice Validate that all required models and configurations are properly loaded
    @return True if all validations pass, False otherwise
    @dev Performs basic validation of model initialization and configuration
    """
    try:
        # Validate device availability
        if DEVICE == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA specified but not available, falling back to CPU")
        
        # Validate model paths exist
        model_paths = [
            os.path.join(project_root, "Models", "ID Detection & Field Extraction", "detect_id_inside_image.pt"),
            os.path.join(project_root, "Models", "ID Detection & Field Extraction", "detect_field_inside_id.pt"),
            os.path.join(project_root, "Models", "ID Fake Detection", "Fake_model_best.keras")
        ]
        
        for path in model_paths:
            if not os.path.exists(path):
                logging.error(f"Model file not found: {path}")
                return False
        
        # Validate configuration values
        if not (0.0 <= SIMILARITY_THRESHOLD <= 1.0):
            logging.error(f"Invalid SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
            return False
        
        if not (0.0 <= REAL_THRESHOLD <= 1.0):
            logging.error(f"Invalid REAL_THRESHOLD: {REAL_THRESHOLD}")
            return False
        
        logging.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        return False

def get_model_info():
    """
    @notice Get information about loaded models
    @return Dictionary containing model information
    @dev Provides runtime information about model status and device usage
    """
    return {
        "device": DEVICE,
        "face_model": {
            "name": "InceptionResnetV1",
            "pretrained": "vggface2",
            "device": DEVICE
        },
        "id_detection_model": {
            "type": "YOLO",
            "task": "ID card detection"
        },
        "field_extraction_model": {
            "type": "YOLO", 
            "task": "Field extraction"
        },
        "fake_detection_model": {
            "type": "TensorFlow/Keras",
            "task": "Fake ID detection"
        },
        "ocr_reader": {
            "type": "EasyOCR",
            "languages": ["ar", "en"]
        },
        "thresholds": {
            "face_similarity": SIMILARITY_THRESHOLD,
            "fake_detection": REAL_THRESHOLD
        }
    }

# Startup Validation

# Validate configuration on module import
if __name__ != "__main__":
    config_valid = validate_configuration()
    if not config_valid:
        logging.error("❌ Configuration validation failed - some features may not work properly")
    else:
        logging.info("🚀 Configuration loaded successfully")
        model_info = get_model_info()
        logging.info(f"📊 Using device: {model_info['device']}")
        logging.info(f"🎯 Face similarity threshold: {model_info['thresholds']['face_similarity']}")
        logging.info(f"🔍 Fake detection threshold: {model_info['thresholds']['fake_detection']}")