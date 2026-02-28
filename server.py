#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask Server - Arabic Sign Language Recognition
النسخة المحسّنة: MediaPipe يشتغل على الموبايل، السيرفر بس للموديل
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import csv
import os

app = Flask(__name__)

# =============================================
# تحميل الموديل والتسميات
# =============================================
MODEL_PATH = 'arabic_model/arabic_sign_model.tflite'
LABELS_PATH = 'arabic_data/arabic_labels.csv'

labels_dict = {}
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        labels_dict[int(row[0])] = row[1]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ الموديل جاهز!")
print(f"✅ الحروف: {list(labels_dict.values())}")

# =============================================
# دالة التنبؤ - تستقبل landmarks مباشرة
# =============================================
def predict_from_landmarks(landmarks):
    """
    landmarks: قائمة من 42 رقم (21 نقطة × x,y) بعد التطبيع
    """
    input_data = np.array([landmarks], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

def normalize_landmarks(raw_landmarks):
    """
    raw_landmarks: قائمة من 42 رقم خام [x0,y0,x1,y1,...]
    بترجع نفس القائمة بعد التطبيع
    """
    # تحويل لأزواج
    points = [(raw_landmarks[i], raw_landmarks[i+1]) for i in range(0, 42, 2)]

    # إحداثيات نسبية من المعصم
    base_x, base_y = points[0]
    rel = []
    for x, y in points:
        rel.extend([x - base_x, y - base_y])

    # تطبيع
    max_val = max(abs(v) for v in rel) or 1
    return [v / max_val for v in rel]

# =============================================
# Routes
# =============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    يستقبل: { landmarks: [x0,y0,x1,y1,...,x20,y20] }  — 42 رقم فقط
    يرجع:  { letter, confidence, status }
    """
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'No landmarks'}), 400

    try:
        raw = data['landmarks']
        if len(raw) != 42:
            return jsonify({'error': f'Expected 42 values, got {len(raw)}'}), 400

        # تطبيع
        normalized = normalize_landmarks(raw)

        # تنبؤ
        probs = predict_from_landmarks(normalized)
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        letter = labels_dict.get(predicted_idx, '?')

        return jsonify({
            'letter': letter,
            'confidence': round(confidence * 100, 1),
            'status': 'ok'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/collect', methods=['POST'])
def collect_sample():
    """
    يستقبل: { landmarks: [...42 رقم...], label: int }
    يحفظ في CSV مباشرة
    """
    data = request.get_json()
    if not data or 'landmarks' not in data or 'label' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    try:
        raw = data['landmarks']
        if len(raw) != 42:
            return jsonify({'success': False, 'error': 'Expected 42 landmarks'})

        label = int(data['label'])
        normalized = normalize_landmarks(raw)

        csv_path = 'arabic_data/arabic_keypoints.csv'
        os.makedirs('arabic_data', exist_ok=True)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label] + normalized)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sample_counts')
def sample_counts():
    counts = {}
    csv_path = 'arabic_data/arabic_keypoints.csv'
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    label = int(row[0])
                    counts[label] = counts.get(label, 0) + 1
    return jsonify({'counts': counts})

@app.route('/collect-data')
def collect_page():
    return render_template('collect.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'letters': len(labels_dict)})

# =============================================
# تشغيل
# =============================================
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  🌐 السيرفر شغال!")
    print(f"  💻 على الكمبيوتر: http://localhost:5000")
    print(f"  📱 على الموبايل:  http://{local_ip}:5000")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
