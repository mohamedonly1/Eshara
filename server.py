#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, session, redirect
from functools import wraps
import numpy as np
import tensorflow as tf
import csv
import os
import json
import logging

from dotenv import load_dotenv

from auth import (
    register_user, login_user, load_user,
    record_sample, record_rejected,
    get_all_users_stats, delete_user,
    verify_admin
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_fallback_secret_change_me')

# =============================================
# Login Required Decorator
# =============================================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            if request.path.startswith(('/predict', '/collect', '/history', '/sample_counts')):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return wrapper

# =============================================
# Global Settings (Admin)
# =============================================
SETTINGS_PATH = os.path.join('arabic_data', 'settings.json')

def _load_global_settings():
    default = {
        'quality_filter': True,
        'filter_threshold': 0.95
    }
    if not os.path.exists(SETTINGS_PATH):
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    try:
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'quality_filter': bool(data.get('quality_filter', True)),
            'filter_threshold': float(data.get('filter_threshold', 0.95)),
        }
    except Exception as exc:
        logger.error("Failed to load global settings: %s", exc)
        return default


def _save_global_settings():
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(GLOBAL_SETTINGS, f, ensure_ascii=False, indent=2)


GLOBAL_SETTINGS = _load_global_settings()

# =============================================
# Model Setup
# =============================================
MODEL_PATH = 'arabic_model/arabic_sign_model.tflite'
LABELS_PATH = 'arabic_data/arabic_labels.csv'
POSES_PATH = os.path.join('static', 'poses.js')

labels_dict = {}
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    for row in csv.reader(f):
        labels_dict[int(row[0])] = row[1]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("الموديل جاهز")

# =============================================
# Helpers
# =============================================
def normalize_landmarks(raw):
    points = [(raw[i], raw[i + 1]) for i in range(0, 42, 2)]
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
    return interpreter.get_tensor(output_details[0]['index'])[0]

# =============================================
# Pages
# =============================================
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    if session.get('user_id'):
        return redirect('/')
    return render_template('login.html')

@app.route('/profile')
@login_required
def profile_page():
    return render_template('profile.html')

@app.route('/collect-data')
@login_required
def collect_page():
    return render_template('collect.html')

@app.route('/translate')
@login_required
def translate_page():
    return render_template('translate.html')

@app.route('/pose-editor')
@login_required
def pose_editor_page():
    return render_template('pose_editor.html')

# =============================================
# Auth APIs
# =============================================
@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json(silent=True) or {}
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    ua = request.headers.get('User-Agent', '')
    user_id, error = login_user(
        data.get('name', ''),
        data.get('password', ''),
        ip=ip, user_agent=ua
    )
    if error:
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    return jsonify({'success': True, 'name': load_user(user_id)['name']})

@app.route('/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json(silent=True) or {}
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    ua = request.headers.get('User-Agent', '')
    user_id, error = register_user(
        data.get('name', ''),
        data.get('password', ''),
        ip=ip, user_agent=ua
    )
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
    return jsonify({
        'logged_in': True,
        'name': user['name'],
        'samples': user.get('samples', {}),
        'total': user.get('total_accepted', 0),
        'rejected': user.get('rejected', 0),
        'created': user.get('created', ''),
        'is_admin': session.get('is_admin', False)
    })

# =============================================
# Prediction & Collection APIs
# =============================================
@app.route('/predict', methods=['POST'])
@login_required
def predict_route():
    data = request.get_json(silent=True) or {}
    raw = data.get('landmarks', [])
    if len(raw) != 42:
        return jsonify({'error': 'Invalid landmarks'}), 400
    normalized = normalize_landmarks(raw)
    probs = predict_landmarks(normalized)
    idx = int(np.argmax(probs))
    return jsonify({
        'letter': labels_dict.get(idx, '?'),
        'confidence': round(float(probs[idx]) * 100, 1),
        'status': 'ok'
    })

@app.route('/collect', methods=['POST'])
@login_required
def collect_sample():
    data = request.get_json(silent=True) or {}
    if not data or 'landmarks' not in data or 'label' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    user_id = session.get('user_id')
    raw = data.get('landmarks', [])
    label = int(data.get('label', -1))

    if len(raw) != 42:
        return jsonify({'success': False, 'error': 'Invalid landmarks'})

    try:
        normalized = normalize_landmarks(raw)

        if GLOBAL_SETTINGS['quality_filter']:
            probs = predict_landmarks(normalized)
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            if idx != label and confidence > GLOBAL_SETTINGS['filter_threshold']:
                if user_id:
                    record_rejected(user_id)
                return jsonify({
                    'success': False,
                    'rejected': True,
                    'error': f'الإيماءة تشبه حرف {labels_dict.get(idx, "?")} — جرّب تاني'
                })

        os.makedirs('arabic_data', exist_ok=True)
        with open('arabic_data/arabic_keypoints.csv', 'a', newline='') as f:
            csv.writer(f).writerow([label] + normalized)

        if user_id:
            record_sample(user_id, label)

        return jsonify({'success': True})

    except Exception as exc:
        logger.error("Error in /collect: %s", exc)
        return jsonify({'success': False, 'error': str(exc)})

@app.route('/sample_counts')
@login_required
def sample_counts():
    counts = {}
    path = 'arabic_data/arabic_keypoints.csv'
    if os.path.exists(path):
        with open(path, 'r') as f:
            for row in csv.reader(f):
                if row:
                    label = int(row[0])
                    counts[label] = counts.get(label, 0) + 1
    return jsonify({'counts': counts})

# =============================================
# History
# =============================================
@app.route('/history/save', methods=['POST'])
@login_required
def history_save():
    from history import save_entry
    data = request.get_json(silent=True) or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'success': False, 'error': 'نص فاضي'})
    save_entry(session['user_id'], text)
    return jsonify({'success': True})

@app.route('/history/list')
@login_required
def history_list():
    from history import get_entries
    return jsonify({'entries': get_entries(session['user_id'])})

@app.route('/history/delete', methods=['POST'])
@login_required
def history_delete():
    from history import delete_entry
    data = request.get_json(silent=True) or {}
    delete_entry(session['user_id'], data.get('id'))
    return jsonify({'success': True})

# =============================================
# Admin
# =============================================
@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/admin/verify', methods=['POST'])
def admin_verify():
    data = request.get_json(silent=True) or {}
    ok = verify_admin(data.get('password', ''))
    if ok:
        session['is_admin'] = True
    return jsonify({'ok': ok})

@app.route('/admin/stats')
def admin_stats():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'users': get_all_users_stats()})

@app.route('/admin/delete_user', methods=['POST'])
def admin_delete_user():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    delete_user(data.get('user_id'))
    return jsonify({'ok': True})

@app.route('/admin/settings', methods=['GET'])
def admin_get_settings():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(GLOBAL_SETTINGS)

@app.route('/admin/settings', methods=['POST'])
def admin_update_settings():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    if 'quality_filter' in data:
        GLOBAL_SETTINGS['quality_filter'] = bool(data['quality_filter'])
    if 'filter_threshold' in data:
        GLOBAL_SETTINGS['filter_threshold'] = max(0.5, min(1.0, float(data['filter_threshold'])))
    _save_global_settings()
    return jsonify({'ok': True, 'settings': GLOBAL_SETTINGS})

@app.route('/admin/export')
def admin_export():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    users = get_all_users_stats()
    return jsonify({'users': users, 'total_users': len(users),
                    'total_samples': sum(u['total'] for u in users)})

@app.route('/pose-editor/save', methods=['POST'])
@login_required
def pose_editor_save():
    """Update /static/poses.js with a new/updated pose for a letter."""
    data = request.get_json(silent=True) or {}
    letter = data.get('letter')
    pose = data.get('pose')
    if not letter or not isinstance(pose, dict):
        return jsonify({'success': False, 'error': 'بيانات غير صالحة'}), 400

    # تحميل poses الحالية إن وُجد الملف
    poses = {}
    if os.path.exists(POSES_PATH):
        try:
            with open(POSES_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                poses = json.loads(json_str)
        except Exception as exc:
            logger.error("Failed to read existing poses.js: %s", exc)

    poses[letter] = pose

    try:
        os.makedirs(os.path.dirname(POSES_PATH), exist_ok=True)
        with open(POSES_PATH, 'w', encoding='utf-8') as f:
            f.write('const POSES = ')
            json.dump(poses, f, ensure_ascii=False, indent=2)
            f.write(';\n')
    except Exception as exc:
        logger.error("Failed to write poses.js: %s", exc)
        return jsonify({'success': False, 'error': 'تعذّر حفظ الملف'}), 500

    return jsonify({'success': True})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# =============================================
# PWA Files
# =============================================
@app.route('/static/pwa/sw.js')
def service_worker():
    from flask import send_from_directory, make_response
    res = make_response(send_from_directory('static/pwa', 'sw.js'))
    res.headers['Content-Type'] = 'application/javascript'
    res.headers['Service-Worker-Allowed'] = '/'
    res.headers['Cache-Control'] = 'no-cache'
    return res

@app.route('/static/pwa/manifest.json')
@app.route('/manifest.json')
def manifest():
    from flask import send_from_directory, make_response
    res = make_response(send_from_directory('static/pwa', 'manifest.json'))
    res.headers['Content-Type'] = 'application/manifest+json'
    res.headers['Cache-Control'] = 'no-cache'
    return res

# =============================================
if __name__ == '__main__':
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    banner = (
        f"\n{'='*50}\n"
        f"  🌐 السيرفر شغال!\n"
        f"  💻 على الكمبيوتر: http://localhost:5000\n"
        f"  📱 على الموبايل:  http://{local_ip}:5000\n"
        f"  🔐 الأدمن: http://localhost:5000/admin\n"
        f"{'='*50}\n"
    )
    logger.info(banner)
    app.run(host='0.0.0.0', port=5000, debug=False)
