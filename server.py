#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, session, redirect
import numpy as np
import tensorflow as tf
import csv
import os
import json
from auth import (register_user, login_user, load_user, record_sample,
                  record_rejected, get_all_users_stats, delete_user,
                  verify_admin)

app = Flask(__name__)
app.secret_key = 'arabic_sign_secret_2026'

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

def normalize_landmarks(raw):
    points = [(raw[i], raw[i+1]) for i in range(0, 42, 2)]
    base_x, base_y = points[0]
    rel = []
    for x, y in points:
        rel.extend([x - base_x, y - base_y])
    max_val = max(abs(v) for v in rel) or 1
    return [v / max_val for v in rel]

def predict_landmarks(landmarks):
    input_data = np.array([landmarks], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# =============================================
# Main App
# =============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'No landmarks'}), 400
    try:
        raw = data['landmarks']
        if len(raw) != 42:
            return jsonify({'error': 'Expected 42 values'}), 400
        normalized = normalize_landmarks(raw)
        probs = predict_landmarks(normalized)
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        letter = labels_dict.get(predicted_idx, '?')
        return jsonify({'letter': letter, 'confidence': round(confidence * 100, 1), 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================
# Auth Routes
# =============================================
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json()
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    ua = request.headers.get('User-Agent', '')
    user_id, error = login_user(data.get('name',''), data.get('password',''), ip=ip, user_agent=ua)
    if error:
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    user = load_user(user_id)
    return jsonify({'success': True, 'name': user['name']})

@app.route('/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json()
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    ua = request.headers.get('User-Agent', '')
    user_id, error = register_user(data.get('name',''), data.get('password',''), ip=ip, user_agent=ua)
    if error:
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    return jsonify({'success': True})

@app.route('/auth/logout')
def auth_logout():
    session.clear()
    return redirect('/login')

@app.route('/auth/me')
def auth_me():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'logged_in': False})
    user = load_user(user_id)
    if not user:
        return jsonify({'logged_in': False})
    return jsonify({'logged_in': True, 'name': user['name'],
                    'samples': user.get('samples', {}),
                    'total': user.get('total_accepted', 0),
                    'rejected': user.get('rejected', 0),
                    'created': user.get('created', ''),
                    'is_admin': session.get('is_admin', False)})

# =============================================
# Collect Data
# =============================================
@app.route('/profile')
def profile_page():
    if not session.get('user_id'):
        return redirect('/login')
    return render_template('profile.html')

@app.route('/collect-data')
def collect_page():
    if not session.get('user_id'):
        return redirect('/login')
    return render_template('collect.html')

@app.route('/collect', methods=['POST'])
def collect_sample():
    data = request.get_json()
    if not data or 'landmarks' not in data or 'label' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    user_id = session.get('user_id')
    raw = data['landmarks']
    label = int(data['label'])

    if len(raw) != 42:
        return jsonify({'success': False, 'error': 'Expected 42 landmarks'})

    try:
        normalized = normalize_landmarks(raw)

        # فلتر الجودة: الموديل يتحقق من العينة
        probs = predict_landmarks(normalized)
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        # لو الموديل متأكد إن الحرف غلط (ثقة > 70% في حرف تاني)
        if predicted_idx != label and confidence > 0.70:
            if user_id:
                record_rejected(user_id)
            wrong_letter = labels_dict.get(predicted_idx, '?')
            return jsonify({
                'success': False,
                'rejected': True,
                'error': f'الإيماءة تشبه حرف {wrong_letter} — جرّب تاني'
            })

        # حفظ في CSV
        csv_path = 'arabic_data/arabic_keypoints.csv'
        os.makedirs('arabic_data', exist_ok=True)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label] + normalized)

        # تحديث سجل المستخدم
        if user_id:
            record_sample(user_id, label)

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

# =============================================
# Admin Routes
# =============================================
@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/admin/verify', methods=['POST'])
def admin_verify():
    data = request.get_json()
    ok = verify_admin(data.get('password', ''))
    if ok:
        session['is_admin'] = True
    return jsonify({'ok': ok})

@app.route('/admin/stats')
def admin_stats():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    users = get_all_users_stats()
    return jsonify({'users': users})

@app.route('/admin/delete_user', methods=['POST'])
def admin_delete_user():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    delete_user(data.get('user_id'))
    return jsonify({'ok': True})

@app.route('/admin/export')
def admin_export():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    users = get_all_users_stats()
    return jsonify({'users': users, 'total_users': len(users),
                    'total_samples': sum(u['total'] for u in users)})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# =============================================
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  🌐 السيرفر شغال!")
    print(f"  💻 على الكمبيوتر: http://localhost:5000")
    print(f"  📱 على الموبايل:  http://{local_ip}:5000")
    print(f"  🔐 الأدمن: http://localhost:5000/admin  (كلمة المرور: admin123)")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
