#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask Server - Arabic Sign Language Recognition
شغّل: python server.py
افتح المتصفح على: http://localhost:5000
أو من الموبايل: http://[IP الجهاز]:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import csv
import os
import base64
import cv2
from mediapipe.python.solutions import hands as mp_hands_solutions

app = Flask(__name__)

# =============================================
# تحميل الموديل والتسميات
# =============================================
MODEL_PATH = 'arabic_model/arabic_sign_model.tflite'
LABELS_PATH = 'arabic_data/arabic_labels.csv'

# تحميل التسميات
labels_dict = {}
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        labels_dict[int(row[0])] = row[1]

# تحميل الموديل
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# إعداد MediaPipe
mp_hands = mp_hands_solutions.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
)

print("✅ الموديل جاهز!")
print(f"✅ الحروف المتاحة: {list(labels_dict.values())}")

# =============================================
# دوال المعالجة
# =============================================
def extract_landmarks(hand_landmarks, image_shape):
    h, w = image_shape[:2]
    landmark_list = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        landmark_list.append([x, y])

    base_x, base_y = landmark_list[0]
    rel = []
    for x, y in landmark_list:
        rel.extend([x - base_x, y - base_y])

    max_val = max(abs(v) for v in rel) or 1
    return [v / max_val for v in rel]

def predict(landmarks):
    input_data = np.array([landmarks], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

def process_frame(image_data_url):
    """معالجة صورة base64 وإرجاع الحرف المتعرف عليه"""
    try:
        # تحويل base64 لصورة
        header, data = image_data_url.split(',', 1)
        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return None, 0, "Invalid image"

        # تحويل لـ RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # معالجة MediaPipe
        results = mp_hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, 0, "No hand detected"

        # استخراج النقاط
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = extract_landmarks(hand_landmarks, img.shape)

        # تنبؤ الموديل
        probs = predict(landmarks)
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        letter = labels_dict.get(predicted_idx, '?')

        return letter, confidence, "ok"

    except Exception as e:
        return None, 0, str(e)

# =============================================
# Routes
# =============================================
@app.route('/')
def index():
    return render_template('collect.html', letters=list(labels_dict.values()))

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image'}), 400

    letter, confidence, status = process_frame(data['image'])

    return jsonify({
        'letter': letter,
        'confidence': round(confidence * 100, 1),
        'status': status
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'letters': len(labels_dict)})

# =============================================
# صفحة جمع البيانات
# =============================================
@app.route('/collect-data')
def collect_page():
    return render_template('collect.html')

@app.route('/collect', methods=['POST'])
def collect_sample():
    data = request.get_json()
    if not data or 'image' not in data or 'label' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    image_data = data['image']
    label = int(data['label'])

    try:
        header, img_data = image_data.split(',', 1)
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image'})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({'success': False, 'error': 'لم يتم اكتشاف يد'})

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = extract_landmarks(hand_landmarks, img.shape)

        csv_path = 'arabic_data/arabic_keypoints.csv'
        os.makedirs('arabic_data', exist_ok=True)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label] + landmarks)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/check_hand', methods=['POST'])
def check_hand():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'detected': False})
    try:
        header, img_data = data['image'].split(',', 1)
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(img_rgb)
        return jsonify({'detected': results.multi_hand_landmarks is not None})
    except:
        return jsonify({'detected': False})

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

# =============================================
# تشغيل السيرفر
# =============================================
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  🌐 السيرفر شغال!")
    print(f"  💻 على الكمبيوتر: http://localhost:5000")
    print(f"  📱 على الموبايل:  http://{local_ip}:5000")
    print(f"  (تأكد إن الموبايل والكمبيوتر على نفس الـ WiFi)")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
