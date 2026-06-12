#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================
تدريب موديل لغة الإشارة العربية الموحدة
===========================================
This script trains the Ishara Arabic Sign Language recognition model.
Updated to support config.py, centralized logging, dual-format datasets,
automated metrics reporting, and timestamped model versioning.
"""

import csv
import numpy as np
import os
import shutil
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config

# Setup Logger
logger = config.get_file_logger('training', config.TRAINING_LOG)

logger.info("=" * 50)
logger.info("  تدريب موديل لغة الإشارة العربية - Training Started")
logger.info("=" * 50)

# 1. Loading Languages Config and Labels
import sys

LANGUAGES_CONFIG_PATH = 'arabic_data/languages.json'
lang_code = 'ar'
if len(sys.argv) > 1:
    lang_code = sys.argv[1].strip().lower()

logger.info("Training language: %s", lang_code.upper())

if not os.path.exists(LANGUAGES_CONFIG_PATH):
    logger.error("languages.json not found: %s", LANGUAGES_CONFIG_PATH)
    raise FileNotFoundError(f"languages.json not found: {LANGUAGES_CONFIG_PATH}")

with open(LANGUAGES_CONFIG_PATH, 'r', encoding='utf-8') as f:
    languages_config = json.load(f)

lang_info = languages_config['languages'].get(lang_code)
if not lang_info:
    logger.error("Language code '%s' not found in languages.json", lang_code)
    raise ValueError(f"Language code '{lang_code}' not found in languages.json")

labels_dict = {idx: label for idx, label in enumerate(lang_info['labels'])}
logger.info("Letters available: %d", len(labels_dict))

train_csv_path = lang_info['dataset_path']

# 2. Loading Dataset (Robust auto-detection for dual format)
X, y_raw = [], []
if not os.path.exists(train_csv_path):
    logger.error("Training CSV dataset not found: %s", train_csv_path)
    raise FileNotFoundError(f"Training dataset not found: {train_csv_path}")

logger.info("Loading training dataset from: %s", train_csv_path)
with open(train_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row_idx, row in enumerate(reader):
        if not row:
            continue
        try:
            # Auto-detect format based on row length
            if len(row) == 44:  # New format: user_id, label, landmarks (42 values)
                label = int(row[1])
                landmarks = [float(v) for v in row[2:]]
            elif len(row) == 43:  # Old format: label, landmarks (42 values)
                label = int(row[0])
                landmarks = [float(v) for v in row[1:]]
            else:
                logger.warning("Row %d: Skipping malformed row with length %d", row_idx + 1, len(row))
                continue
            
            if len(landmarks) != 42:
                logger.warning("Row %d: Skipping row with invalid landmarks length %d", row_idx + 1, len(landmarks))
                continue
                
            y_raw.append(label)
            X.append(landmarks)
        except (ValueError, IndexError) as exc:
            logger.warning("Row %d: Error parsing row - %s", row_idx + 1, exc)
            continue

X = np.array(X, dtype=np.float32)
y_raw = np.array(y_raw, dtype=np.int32)

logger.info("Successfully loaded %d training samples.", len(X))

if len(X) == 0:
    logger.error("Dataset is empty. Cannot start training.")
    raise ValueError("No training data found in CSV.")

# 3. Label Re-encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# Build label map
encoded_labels = {new_idx: labels_dict.get(orig_idx, str(orig_idx))
                  for new_idx, orig_idx in enumerate(le.classes_)}

logger.info("Sample distribution across classes:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    letter = encoded_labels.get(label, '?')
    bar = "█" * min(count // 10, 20)
    status = "[OK]" if count >= 200 else "[..]"
    logger.info("  %s %s: %d %s", status, letter, count, bar)

# 4. Splitting Data
logger.info("Splitting dataset into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info("Training data size: %d", len(X_train))
logger.info("Testing data size: %d", len(X_test))

# 5. Building the Model
import tensorflow as tf
from tensorflow import keras

logger.info("Building Sequential MLP model architecture...")
model = keras.Sequential([
    keras.layers.Input(shape=(42,)),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary(print_fn=lambda x: logger.info(x))

# Callbacks
best_model_path = os.path.join(config.MODEL_DIR, f'{lang_code}_best_model.h5')
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=10, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# 6. Model Training
logger.info("Starting model training...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
logger.info("Model training completed.")

# 7. Model Evaluation
logger.info("Evaluating model on test partition...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
logger.info("  Test Accuracy: %0.2f%%", test_acc * 100)
logger.info("  Test Loss: %0.4f", test_loss)

# 8. Report Generation & Matrix Outputs
y_pred = np.argmax(model.predict(X_test), axis=1)
letter_names = [encoded_labels.get(i, str(i)) for i in range(num_classes)]

# Compute metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

# Save Classification Report
cls_report_str = classification_report(y_test, y_pred, target_names=letter_names, zero_division=0)
lang_cls_report_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_classification_report.txt")
with open(lang_cls_report_path, 'w', encoding='utf-8') as f:
    f.write(cls_report_str)
logger.info("Saved text classification report to: %s", lang_cls_report_path)
if lang_code == 'ar':
    with open(config.CLASSIFICATION_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(cls_report_str)

# Save Confusion Matrices
cm = confusion_matrix(y_test, y_pred)

# Standard Confusion Matrix
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
plt.title(f'Confusion Matrix - {lang_code.upper()}', fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
lang_cm_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_confusion_matrix.png")
plt.savefig(lang_cm_path, dpi=150)
if lang_code == 'ar':
    plt.savefig(config.CONFUSION_MATRIX_PNG, dpi=150)
plt.close()

# Normalized Confusion Matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16, 14))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
plt.title(f'Normalized Confusion Matrix - {lang_code.upper()}', fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
lang_cm_norm_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_normalized_confusion_matrix.png")
plt.savefig(lang_cm_norm_path, dpi=150)
if lang_code == 'ar':
    plt.savefig(config.NORMALIZED_CONFUSION_MATRIX_PNG, dpi=150)
plt.close()

logger.info("Saved confusion matrix plots to reports/ directory.")

# Save JSON Report
cls_report_dict = classification_report(y_test, y_pred, target_names=letter_names, output_dict=True, zero_division=0)
training_report = {
    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'accuracy': float(test_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classification_report': cls_report_dict
}
lang_report_json_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_training_report.json")
with open(lang_report_json_path, 'w', encoding='utf-8') as f:
    json.dump(training_report, f, indent=2, ensure_ascii=False)
if lang_code == 'ar':
    with open(config.TRAINING_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False)
logger.info("Saved JSON metrics report to: %s", lang_report_json_path)

# 9. Model Saving and Versioning (Phase 5)
logger.info("Executing model saving and versioning...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Versioned names
timestamped_keras_name = f"{lang_code}_sign_model_{timestamp}.keras"
timestamped_keras_path = os.path.join(config.MODEL_DIR, timestamped_keras_name)
timestamped_tflite_name = f"{lang_code}_sign_model_{timestamp}.tflite"
timestamped_tflite_path = os.path.join(config.MODEL_DIR, timestamped_tflite_name)

# Legacy naming for backward compatibility
legacy_tflite_name = f"{lang_code}_sign_model_{datetime.now().strftime('%Y-%m-%d')}_{test_acc*100:.2f}.tflite"
legacy_tflite_path = os.path.join(config.MODEL_DIR, legacy_tflite_name)

# Load best weights from checkpoint if checkpoint exists
if os.path.exists(best_model_path):
    best_model = keras.models.load_model(best_model_path)
    # Save standard backup
    best_model.save(os.path.join(config.MODEL_DIR, f"{lang_code}_best_model.h5"))
else:
    best_model = model
    best_model.save(os.path.join(config.MODEL_DIR, f"{lang_code}_best_model.h5"))

# Save timestamped Keras model
best_model.save(timestamped_keras_path)
logger.info("Saved timestamped Keras model: %s", timestamped_keras_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Write TFLite files
with open(timestamped_tflite_path, 'wb') as f:
    f.write(tflite_model)
with open(legacy_tflite_path, 'wb') as f:
    f.write(tflite_model)

# For backward compatibility if lang_code is 'ar'
if lang_code == 'ar':
    with open(config.MODEL_PATH_TFLITE, 'wb') as f:
        f.write(tflite_model)

logger.info("Exported TFLite models: %s and %s", timestamped_tflite_path, legacy_tflite_path)

# Update languages.json with the new model path
try:
    with open(LANGUAGES_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    config_data['languages'][lang_code]['model_path'] = f"arabic_model/{legacy_tflite_name}"
    with open(LANGUAGES_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    logger.info("Successfully updated languages.json model_path to: arabic_model/%s", legacy_tflite_name)
except Exception as e:
    logger.error("Failed to update languages.json: %s", e)

# 10. Automatically Update server.py Model Reference (Non-interactive) for active Arabic lang
def update_server_model_path(new_model_name):
    server_path = 'server.py'
    if not os.path.exists(server_path):
        logger.warning("server.py not found in the root workspace; skipping reference update.")
        return False
    
    try:
        with open(server_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        pattern = r"MODEL_PATH\s*=\s*['\"]arabic_model/[^'\"]+\.tflite['\"]"
        replacement = f"MODEL_PATH = 'arabic_model/{new_model_name}'"
        new_content, count = re.subn(pattern, replacement, content)
        
        if count > 0:
            with open(server_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info("Successfully updated server.py MODEL_PATH to: %s", new_model_name)
            return True
        else:
            logger.warning("Could not find matching MODEL_PATH pattern in server.py.")
            return False
    except Exception as e:
        logger.error("Failed to update server.py: %s", e)
        return False

if lang_code == 'ar':
    update_server_model_path(legacy_tflite_name)

logger.info("Training pipeline complete! Final Test Accuracy: %0.2f%%", test_acc * 100)