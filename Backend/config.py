import os
import warnings
import torch
import logging
import numpy as np
from typing import Optional
from ultralytics import YOLO
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, request, jsonify
import tempfile
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from skimage.feature import local_binary_pattern
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
SIMILARITY_THRESHOLD = 0.7
REAL_THRESHOLD = 0.75
MIN_IMAGE_SIZE = 40
ID_NUMBER_CLASS = 6
IMAGE_SIZE = 160
ELA_QUALITY = 85
ELA_SCALE_FACTOR = 15
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False
SWAGGER = {
    'title': 'Face Authentication API',
    'uiversion': 3,
    'specs_route': '/apidocs/'
}
mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=14,
    device=DEVICE,
    selection_method='center_weighted_size',
    min_face_size=MIN_IMAGE_SIZE
)
INCEPTION_MODEL = InceptionResnetV1(classify=False, pretrained='vggface2').to(DEVICE).eval()
YOLO_MODEL_1 = YOLO(r"Models\ID Detection & Field Extraction\detect_id_inside_image.pt")
YOLO_MODEL_2 = YOLO(r"Models\ID Detection & Field Extraction\detect_field_inside_id.pt")
FAKE_MODEL = tf.keras.models.load_model(r"Models\ID Fake Detection\Fake_model_best.keras", compile=False)