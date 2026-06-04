#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, session, redirect, send_from_directory
from functools import wraps
import numpy as np
import tensorflow as tf
import csv
import os
import json
import logging
import threading
import secrets

from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from auth import (
    register_user, login_user, load_user,
    record_sample, record_rejected,
    get_all_users_stats, delete_user, set_user_role, user_is_admin,
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
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

_session_secret = os.getenv('FLASK_SECRET_KEY')
if not _session_secret:
    _env_name = os.getenv('FLASK_ENV') or os.getenv('APP_ENV') or os.getenv('ENV') or ''
    if _env_name.lower() in {'prod', 'production'}:
        raise RuntimeError('FLASK_SECRET_KEY must be set in production')
    _session_secret = secrets.token_hex(32)
    logger.warning("FLASK_SECRET_KEY is not set; using a temporary development secret.")
app.secret_key = _session_secret

# =============================================
# Login Required Decorator
# =============================================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user_id = session.get('user_id')
        user = load_user(user_id) if user_id else None
        if not user or user.get('status') == 'disabled':
            if user_id:
                session.clear()
            wants_json = (
                request.is_json or
                request.headers.get('Accept', '').find('application/json') != -1 or
                request.path.startswith('/api/')
            )
            if wants_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return wrapper

def current_user_is_admin():
    if session.get('is_admin'):
        return True
    user_id = session.get('user_id')
    if not user_id:
        return False
    return user_is_admin(load_user(user_id))

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user_is_admin():
            wants_json = (
                request.is_json or
                request.headers.get('Accept', '').find('application/json') != -1 or
                request.path.startswith('/admin/') or
                request.path.startswith('/pose-editor/')
            )
            if wants_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/admin')
        return f(*args, **kwargs)
    return wrapper

# =============================================
# Global Settings (Admin)
# =============================================
SETTINGS_PATH = os.path.join('arabic_data', 'settings.json')

def _load_global_settings():
    default = {
        'quality_filter': True,
        'filter_threshold': 0.85,
        'hand_yaw': -0.55,
        'hand_pitch': 0.15
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
            'hand_yaw': float(data.get('hand_yaw', -0.55)),
            'hand_pitch': float(data.get('hand_pitch', 0.15)),
        }
    except Exception as exc:
        logger.error("Failed to load global settings: %s", exc)
        return default


def _save_global_settings():
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(GLOBAL_SETTINGS, f, ensure_ascii=False, indent=2)


GLOBAL_SETTINGS = _load_global_settings()
_settings_lock = threading.Lock()

# =============================================
# Model Setup
# =============================================
MODEL_PATH = 'arabic_model/arabic_sign_model_2026-05-22_95.96.tflite'

LABELS_PATH = 'arabic_data/arabic_labels.csv'
POSES_PATH = os.path.join('static', 'poses.js')

labels_dict = {}
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    for row in csv.reader(f):
        labels_dict[int(row[0])] = row[1]

_means_cache = None
_means_mtime = None

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_interpreter_lock = threading.Lock()
logger.info("Model loaded successfully")

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
    with _interpreter_lock:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0].copy()
# =============================================
# Pages
# =============================================
@app.route('/')
@login_required
def index():
    # جعلنا الصفحة الرئيسية هي صفحة التعرف (الكاميرا)
    return render_template('index.html', labels_map=labels_dict)

@app.route('/recognize')
@login_required
def recognize_page():
    # إعادة توجيه مسار التعرف إلى الصفحة الرئيسية
    return redirect('/')

@app.route('/login')
def login_page():
    user_id = session.get('user_id')
    if user_id:
        user = load_user(user_id)
        if user and user.get('status') != 'disabled':
            return redirect('/')
        else:
            session.clear()
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
    # المسار يعمل أيضاً في حال تم طلبه بشكل مباشر
    return render_template('translate.html', labels_map=labels_dict)

@app.route('/means')
@login_required
def means_route():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    global _means_cache, _means_mtime

    path = os.path.join('arabic_data', 'arabic_keypoints.csv')
    if not os.path.exists(path):
        _means_cache = {}
        _means_mtime = None
        return jsonify({})

    try:
        file_mtime = os.path.getmtime(path)
        if _means_cache is not None and _means_mtime == file_mtime:
            return jsonify(_means_cache)

        df = pd.read_csv(path, header=None)
        if df.empty or df.shape[1] < 43:
            _means_cache = {}
            _means_mtime = file_mtime
            return jsonify({})

        raw_labels = df.iloc[:, 0]
        features = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Keep LabelEncoder logic for parity with training-time class handling.
        encoder = LabelEncoder()
        encoder.fit([labels_dict[idx] for idx in sorted(labels_dict.keys())])

        letter_to_index = {letter: int(idx) for idx, letter in labels_dict.items()}
        aligned_labels = []
        for raw in raw_labels:
            idx = None

            # Dataset may store either numeric labels or Arabic letters.
            try:
                numeric = int(float(raw))
                if numeric in labels_dict:
                    idx = numeric
            except (TypeError, ValueError):
                pass

            if idx is None:
                label_str = str(raw).strip()
                if label_str in letter_to_index:
                    idx = letter_to_index[label_str]

            aligned_labels.append(idx)

        grouped = features.copy()
        grouped['label'] = aligned_labels
        grouped = grouped[grouped['label'].notna()].copy()
        grouped['label'] = grouped['label'].astype(int)
        if grouped.empty:
            _means_cache = {}
            _means_mtime = file_mtime
            return jsonify({})

        medians = grouped.groupby('label', sort=True).median(numeric_only=True)

        result = {
            str(int(idx)): [float(v) for v in row.tolist()]
            for idx, row in medians.iterrows()
        }
        _means_cache = result
        _means_mtime = file_mtime
        return jsonify(result)
    except Exception as exc:
        logger.error("Error in /means: %s", exc)
        return jsonify({'error': 'تعذّر تحميل المتوسطات'}), 500

@app.route('/pose-editor')
@login_required
@admin_required
def pose_editor_page():
    return render_template('pose_editor.html')


# =============================================
# Auth APIs
# =============================================
@app.route('/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def auth_login():
    data = request.get_json(silent=True) or {}
    ip = request.remote_addr
    ua = request.headers.get('User-Agent', '')
    user_id, error = login_user(
        data.get('name', ''),
        data.get('password', ''),
        ip=ip, user_agent=ua
    )
    if error:
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    if user_is_admin(load_user(user_id)):
        session['is_admin'] = True
    return jsonify({'success': True, 'name': load_user(user_id)['name']})

@app.route('/auth/register', methods=['POST'])
@limiter.limit("10 per minute")
def auth_register():
    data = request.get_json(silent=True) or {}
    ip = request.remote_addr
    ua = request.headers.get('User-Agent', '')
    user_id, error = register_user(
        data.get('name', ''),
        data.get('password', ''),
        ip=ip, user_agent=ua
    )
    if error:
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    session['is_admin'] = False
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
    if not user or user.get('status') == 'disabled':
        if user_id:
            session.clear()
        return jsonify({'logged_in': False})
    return jsonify({
        'logged_in': True,
        'name': user['name'],
        'samples': user.get('samples', {}),
        'total': user.get('total_accepted', 0),
        'rejected': user.get('rejected', 0),
        'created': user.get('created', ''),
        'is_admin': current_user_is_admin(),
        'role': 'admin' if user_is_admin(user) else 'user'
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

    if label not in range(len(labels_dict)):
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400

    if len(raw) != 42:
        return jsonify({'success': False, 'error': 'Invalid landmarks'})

    try:
        normalized = normalize_landmarks(raw)

        with _settings_lock:
            quality_filter = GLOBAL_SETTINGS['quality_filter']
            filter_threshold = GLOBAL_SETTINGS['filter_threshold']

        if quality_filter:
            probs = predict_landmarks(normalized)
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            if idx != label and confidence > filter_threshold:
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
        return jsonify({'success': False, 'error': 'حدث خطأ، يرجى المحاولة مجدداً'})

@app.route('/sample_counts')
@login_required
def sample_counts():
    user_id = session.get('user_id')
    user = load_user(user_id)
    if not user:
        return jsonify({'counts': {}})
    counts = {int(k): v for k, v in user.get('samples', {}).items()}
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
    if len(text) > 5000:
        return jsonify({'success': False, 'error': 'النص طويل جداً (الحد 5000 حرف)'}), 400
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
    if current_user_is_admin():
        session['is_admin'] = True
    return render_template('admin.html')

@app.route('/admin/verify', methods=['POST'])
@limiter.limit("5 per minute")
def admin_verify():
    data = request.get_json(silent=True) or {}
    ok = verify_admin(data.get('password', ''))
    if ok:
        session['is_admin'] = True
    else:
        logger.warning("Failed admin login attempt from %s", request.remote_addr)
    return jsonify({'ok': ok})

@app.route('/admin/stats')
def admin_stats():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'users': get_all_users_stats()})

@app.route('/admin/delete_user', methods=['POST'])
def admin_delete_user():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    delete_user(data.get('user_id'))
    return jsonify({'ok': True})

@app.route('/admin/disable_user', methods=['POST'])
def admin_disable_user():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    from auth import disable_user
    disable_user(data.get('user_id'))
    return jsonify({'ok': True})

@app.route('/admin/enable_user', methods=['POST'])
def admin_enable_user():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    from auth import enable_user
    enable_user(data.get('user_id'))
    return jsonify({'ok': True})

@app.route('/admin/set_user_role', methods=['POST'])
def admin_set_user_role():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    role = data.get('role')
    if user_id == session.get('user_id') and role != 'admin':
        return jsonify({'ok': False, 'error': 'لا يمكن إزالة صلاحية الأدمن من حسابك الحالي'}), 400

    ok, error = set_user_role(user_id, role)
    if not ok:
        return jsonify({'ok': False, 'error': error}), 400

    return jsonify({'ok': True})

@app.route('/admin/settings', methods=['GET'])
def admin_get_settings():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(GLOBAL_SETTINGS)

@app.route('/admin/settings', methods=['POST'])
def admin_update_settings():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    with _settings_lock:
        if 'quality_filter' in data:
            GLOBAL_SETTINGS['quality_filter'] = bool(data['quality_filter'])
        if 'filter_threshold' in data:
            GLOBAL_SETTINGS['filter_threshold'] = max(0.5, min(1.0, float(data['filter_threshold'])))
        if 'hand_yaw' in data:
            GLOBAL_SETTINGS['hand_yaw'] = max(-3.14, min(3.14, float(data['hand_yaw'])))
        if 'hand_pitch' in data:
            GLOBAL_SETTINGS['hand_pitch'] = max(-1.57, min(1.57, float(data['hand_pitch'])))
        _save_global_settings()
    return jsonify({'ok': True, 'settings': GLOBAL_SETTINGS})

@app.route('/admin/export')
def admin_export():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    users = get_all_users_stats()
    return jsonify({'users': users, 'total_users': len(users),
                    'total_samples': sum(u['total'] for u in users)})

@app.route('/pose-editor/save', methods=['POST'])
@login_required
def pose_editor_save():
    """Update /static/poses.js — admin only."""
    if not current_user_is_admin():
        return jsonify({'success': False, 'error': 'غير مسموح'}), 403

    data = request.get_json(silent=True) or {}
    letter = data.get('letter')
    pose = data.get('pose')

    if not letter or not isinstance(pose, dict):
        return jsonify({'success': False, 'error': 'بيانات غير صالحة'}), 400

    # Whitelist: only allow valid Arabic letters (single char or لا)
    ALLOWED_LETTERS = set('أبتثجحخدذرزسشصضطظعغفقكلمنهوي') | {'لا'}
    if letter not in ALLOWED_LETTERS:
        return jsonify({'success': False, 'error': 'حرف غير صالح'}), 400

    poses = {}
    if os.path.exists(POSES_PATH):
        try:
            with open(POSES_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                poses = json.loads(text[start:end+1])
        except Exception as exc:
            logger.error("Failed to read existing poses.js: %s", exc)

    from datetime import datetime
    from auth import load_user
    
    user = load_user(session.get('user_id', ''))
    user_name = user.get('name', 'Admin') if user else 'Admin'
    
    pose['_meta'] = {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'saved_by': user_name
    }

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

    from history import save_entry
    save_entry(session['user_id'], f"قام بضبط وتعديل وضعية الأصابع للحرف: {letter}")

    return jsonify({'success': True})


# =============================================
# Test Data Collection (no auth required)
# =============================================
TEST_CSV = os.path.join('arabic_data', 'test_keypoints.csv')
_test_csv_lock = threading.Lock()

@app.route('/collect-test')
def collect_test_page():
    return render_template('collect_test.html')

@app.route('/collect-test', methods=['POST'])
@limiter.limit("30 per minute")
def collect_test_sample():
    data = request.get_json(silent=True) or {}
    raw = data.get('landmarks', [])
    label = data.get('label')
    tester_id = str(data.get('tester_id', '')).strip()

    if not tester_id or label is None:
        return jsonify({'success': False, 'error': 'بيانات ناقصة'}), 400
    label = int(label)
    if label not in range(len(labels_dict)):
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400
    if len(raw) != 42:
        return jsonify({'success': False, 'error': 'Invalid landmarks'}), 400

    try:
        normalized = normalize_landmarks(raw)
        os.makedirs('arabic_data', exist_ok=True)
        with _test_csv_lock:
            with open(TEST_CSV, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([tester_id, label] + normalized)
            # count for this tester+label
            total = 0
            if os.path.exists(TEST_CSV):
                with open(TEST_CSV, 'r', encoding='utf-8') as f:
                    for row in csv.reader(f):
                        if len(row) >= 2 and row[0] == tester_id and row[1] == str(label):
                            total += 1
        return jsonify({'success': True, 'total': total})
    except Exception as exc:
        logger.error("Error in /collect-test POST: %s", exc)
        return jsonify({'success': False, 'error': 'حدث خطأ'}), 500

@app.route('/collect-test/export')
def collect_test_export():
    if not os.path.exists(TEST_CSV):
        return jsonify({'error': 'لا توجد بيانات بعد'}), 404
    try:
        with open(TEST_CSV, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in csv.reader(f))
        if row_count < 50:
            return jsonify({'error': f'عدد العينات غير كافٍ ({row_count}/50)'}), 403
    except Exception:
        return jsonify({'error': 'خطأ في قراءة الملف'}), 500
    return send_from_directory('arabic_data', 'test_keypoints.csv',
                               as_attachment=True,
                               download_name='test_keypoints.csv')

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
        f"  Server is running\n"
        f"  Local:   http://localhost:5000\n"
        f"  Network: http://{local_ip}:5000\n"
        f"  Admin:   http://localhost:5000/admin\n"
        f"{'='*50}\n"
    )
    logger.info(banner)
    app.run(host='0.0.0.0', port=5000, debug=False)
