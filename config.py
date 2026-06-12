#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Central configuration for the Ishara sign language recognition system.
Defines all dataset, model, log paths, and default threshold values.
"""
import os
import logging

# Base directories
DATA_DIR = 'arabic_data'
MODEL_DIR = 'arabic_model'
USERS_DIR = os.path.join(DATA_DIR, 'users')
LOG_DIR = 'logs'
REPORTS_DIR = 'reports'

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Dataset paths
TRAIN_CSV = os.path.join(DATA_DIR, 'arabic_keypoints.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test_keypoints.csv')
LABELS_CSV = os.path.join(DATA_DIR, 'arabic_labels.csv')
SETTINGS_JSON = os.path.join(DATA_DIR, 'settings.json')
POSES_JS = os.path.join('static', 'poses.js')

# Model paths
# The production TFLite model currently used in server.py
PRODUCTION_MODEL_TFLITE = os.path.join(MODEL_DIR, 'arabic_sign_model_2026-05-22_95.96.tflite')

# Versioned model locations
MODEL_PATH_H5 = os.path.join(MODEL_DIR, 'arabic_sign_model.h5')
MODEL_PATH_TFLITE = os.path.join(MODEL_DIR, 'arabic_sign_model.tflite')

# Pointer files to latest models (Phase 5)
LATEST_MODEL_KERAS = os.path.join(MODEL_DIR, 'latest.keras')
LATEST_MODEL_TFLITE = os.path.join(MODEL_DIR, 'latest.tflite')

# Log paths (Phase 6)
TRAINING_LOG = os.path.join(LOG_DIR, 'training.log')
SERVER_LOG = os.path.join(LOG_DIR, 'server.log')
COLLECTION_LOG = os.path.join(LOG_DIR, 'collection.log')

# Report outputs (Phase 4)
TRAINING_REPORT_JSON = os.path.join(REPORTS_DIR, 'training_report.json')
CLASSIFICATION_REPORT_TXT = os.path.join(REPORTS_DIR, 'classification_report.txt')
CONFUSION_MATRIX_PNG = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
NORMALIZED_CONFUSION_MATRIX_PNG = os.path.join(REPORTS_DIR, 'normalized_confusion_matrix.png')

# Default parameters
DEFAULT_QUALITY_FILTER = True
DEFAULT_FILTER_THRESHOLD = 0.85
DEFAULT_HAND_YAW = -0.55
DEFAULT_HAND_PITCH = 0.15

def get_file_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Creates a logger that outputs to both a file and standard output.
    Ensures safe handling of console encoding and unicode logs.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
