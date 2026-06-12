#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script to assess model performance on external test datasets.
Generates metrics reports, text logs, and confusion matrix visualizations.
"""
import os
import csv
import json
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config

logger = config.get_file_logger('evaluation', config.SERVER_LOG)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TFLite model on an external test dataset.")
    parser.add_argument(
        "--model", 
        type=str, 
        default=config.PRODUCTION_MODEL_TFLITE,
        help="Path to the TFLite model file to evaluate."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=config.TEST_CSV,
        help="Path to the CSV dataset (expects: tester_id, label, landmarks...)."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=config.REPORTS_DIR,
        help="Directory where evaluation reports will be saved."
    )
    return parser.parse_args()

def load_labels(labels_path: str) -> dict:
    """Loads the label mapping from the specified CSV file."""
    labels = {}
    if not os.path.exists(labels_path):
        logger.error("Labels CSV file not found: %s", labels_path)
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if row:
                labels[int(row[0])] = row[1]
    return labels

def load_test_dataset(dataset_path: str):
    """
    Loads test dataset and auto-detects format.
    Supports:
    - Tester format: tester_id, label, landmarks... (len = 44)
    - User format: user_id, label, landmarks... (len = 44)
    - Standard format: label, landmarks... (len = 43)
    """
    if not os.path.exists(dataset_path):
        logger.error("Dataset CSV file not found: %s", dataset_path)
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    X, y = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) == 44:  # ID (tester/user), label, landmarks...
                try:
                    label = int(row[1])
                    landmarks = [float(v) for v in row[2:]]
                    X.append(landmarks)
                    y.append(label)
                except ValueError:
                    continue
            elif len(row) == 43:  # label, landmarks...
                try:
                    label = int(row[0])
                    landmarks = [float(v) for v in row[1:]]
                    X.append(landmarks)
                    y.append(label)
                except ValueError:
                    continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def evaluate_tflite_model(model_path: str, X: np.ndarray) -> np.ndarray:
    """Runs inference with a TFLite model on the given dataset."""
    if not os.path.exists(model_path):
        logger.error("TFLite model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    for sample in X:
        inp = np.array([sample], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        y_pred.append(np.argmax(output))
    
    return np.array(y_pred, dtype=np.int32)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting evaluation of model: %s", args.model)
    logger.info("Using dataset: %s", args.dataset)

    try:
        labels_dict = load_labels(config.LABELS_CSV)
        X, y_true = load_test_dataset(args.dataset)
        
        if len(X) == 0:
            logger.error("No valid samples found in dataset.")
            return

        y_pred = evaluate_tflite_model(args.model, X)

        # Compute Metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        # Print to console
        logger.info("Overall Accuracy: %0.2f%%", acc * 100)
        logger.info("Weighted Precision: %0.2f%%", precision * 100)
        logger.info("Weighted Recall: %0.2f%%", recall * 100)
        logger.info("Weighted F1-Score: %0.2f%%", f1 * 100)

        letter_names = [labels_dict.get(i, str(i)) for i in range(len(labels_dict))]
        
        # 1. Classification report txt
        cls_report_str = classification_report(y_true, y_pred, target_names=letter_names, zero_division=0)
        cls_report_path = os.path.join(args.output_dir, 'classification_report.txt')
        with open(cls_report_path, 'w', encoding='utf-8') as f:
            f.write(cls_report_str)
        logger.info("Saved classification report to: %s", cls_report_path)

        # 2. Confusion Matrices
        cm = confusion_matrix(y_true, y_pred)
        
        # Standard Confusion Matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
        plt.title('Confusion Matrix - External Test Set', fontsize=16)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # Normalized Confusion Matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
        plt.title('Normalized Confusion Matrix - External Test Set', fontsize=16)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_norm_path = os.path.join(args.output_dir, 'normalized_confusion_matrix.png')
        plt.savefig(cm_norm_path, dpi=150)
        plt.close()

        logger.info("Saved confusion matrix plots to output directory.")

        # 3. Save JSON Report
        cls_report_dict = classification_report(y_true, y_pred, target_names=letter_names, output_dict=True, zero_division=0)
        
        training_report = {
            'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_evaluated': args.model,
            'dataset_evaluated': args.dataset,
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': cls_report_dict
        }
        
        report_json_path = os.path.join(args.output_dir, 'training_report.json')
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(training_report, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON metrics report to: %s", report_json_path)

        logger.info("External evaluation complete.")

    except Exception as exc:
        logger.error("Error during evaluation script execution: %s", exc, exc_info=True)

if __name__ == "__main__":
    main()
