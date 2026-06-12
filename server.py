#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Flask web application server for Ishara.
Serves sign language recognition interfaces, data collection APIs,
history logging, and administration utilities.
"""
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

import config
from auth import (
    register_user, login_user, load_user,
    record_sample, record_rejected,
    get_all_users_stats, delete_user, set_user_role, user_is_admin,
    verify_admin
)

# Central Logger setup (Phase 6)
logger = config.get_file_logger('server', config.SERVER_LOG)
logger.info("Initializing Ishara Flask Server...")

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
                request.path.startswith('/api/') or
                request.path.startswith('/auth/')
            )
            if wants_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/login')

        # Enforce email verification (exclude verification and logout routes to prevent infinite loops)
        if 'email' in user and not user.get('email_verified', False):
            if request.path not in ['/verify-email', '/auth/verify-email', '/auth/resend-code', '/auth/logout']:
                wants_json = (
                    request.is_json or
                    request.headers.get('Accept', '').find('application/json') != -1 or
                    request.path.startswith('/api/') or
                    request.path.startswith('/auth/')
                )
                if wants_json:
                    return jsonify({'error': 'Email verification required', 'needs_verification': True}), 403
                return redirect('/verify-email')

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
SETTINGS_PATH = config.SETTINGS_JSON

def _load_global_settings():
    default = {
        'quality_filter': config.DEFAULT_QUALITY_FILTER,
        'filter_threshold': config.DEFAULT_FILTER_THRESHOLD,
        'hand_yaw': config.DEFAULT_HAND_YAW,
        'hand_pitch': config.DEFAULT_HAND_PITCH
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
            'filter_threshold': float(data.get('filter_threshold', 0.85)),
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
# This variable name and path is preserved exactly for train_model.py regex update compatibility
MODEL_PATH = 'arabic_model/arabic_sign_model_2026-05-22_95.96.tflite'

LABELS_PATH = config.LABELS_CSV
POSES_PATH = config.POSES_JS

labels_dict = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if row:
                labels_dict[int(row[0])] = row[1]

_means_cache = None
_means_mtime = None

# Thread-safe model load
interpreter = None
_interpreter_lock = threading.Lock()

try:
    logger.info("Loading TFLite model from path: %s", MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Failed to load TFLite model during startup: %s", e, exc_info=True)

# =============================================
# Helpers
# =============================================
def normalize_landmarks(raw):
    """
    Translates hand coordinates relative to the wrist (element 0)
    and normalizes them by dividing by the maximum absolute coordinate offset.
    """
    points = [(raw[i], raw[i + 1]) for i in range(0, 42, 2)]
    base_x, base_y = points[0]
    rel = []
    for x, y in points:
        rel.extend([x - base_x, y - base_y])
    max_val = max(abs(v) for v in rel) or 1
    return [v / max_val for v in rel]

def predict_landmarks(landmarks):
    """Runs thread-safe TFLite interpreter inference."""
    input_data = np.array([landmarks], dtype=np.float32)
    with _interpreter_lock:
        if interpreter is None:
            raise RuntimeError("TFLite interpreter is not initialized.")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0].copy()

# =============================================
# Pages
# =============================================
@app.route('/')
@login_required
def index():
    return render_template('index.html', labels_map=labels_dict)

@app.route('/recognize')
@login_required
def recognize_page():
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
    return render_template('translate.html', labels_map=labels_dict)

@app.route('/means')
@login_required
def means_route():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    global _means_cache, _means_mtime

    path = config.TRAIN_CSV
    if not os.path.exists(path):
        _means_cache = {}
        _means_mtime = None
        return jsonify({})

    try:
        file_mtime = os.path.getmtime(path)
        if _means_cache is not None and _means_mtime == file_mtime:
            return jsonify(_means_cache)

        # Robust parser for both 43-column and 44-column CSV rows
        rows_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if len(row) == 44:  # user_id, label, landmarks...
                    label = row[1]
                    landmarks = [float(v) for v in row[2:]]
                    rows_data.append([label] + landmarks)
                elif len(row) == 43:  # label, landmarks...
                    label = row[0]
                    landmarks = [float(v) for v in row[1:]]
                    rows_data.append([label] + landmarks)

        if not rows_data:
            _means_cache = {}
            _means_mtime = file_mtime
            return jsonify({})

        df = pd.DataFrame(rows_data)
        raw_labels = df.iloc[:, 0]
        features = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Keep LabelEncoder logic for parity with training-time class handling.
        encoder = LabelEncoder()
        encoder.fit([labels_dict[idx] for idx in sorted(labels_dict.keys())])

        letter_to_index = {letter: int(idx) for idx, letter in labels_dict.items()}
        aligned_labels = []
        for raw in raw_labels:
            idx = None
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
        logger.error("Error in /means: %s", exc, exc_info=True)
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
        logger.warning("Failed login attempt for name '%s' from IP %s", data.get('name', ''), ip)
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    if user_is_admin(load_user(user_id)):
        session['is_admin'] = True
    logger.info("User logged in successfully: %s", user_id)
    return jsonify({'success': True, 'name': load_user(user_id)['name']})

@app.route('/auth/register', methods=['POST'])
@limiter.limit("10 per minute")
def auth_register():
    data = request.get_json(silent=True) or {}
    ip = request.remote_addr
    ua = request.headers.get('User-Agent', '')
    user_id, error = register_user(
        data.get('name', ''),
        data.get('email', ''),
        data.get('password', ''),
        ip=ip, user_agent=ua
    )
    if error:
        logger.warning("Registration failed for name '%s' from IP %s: %s", data.get('name', ''), ip, error)
        return jsonify({'success': False, 'error': error})
    session['user_id'] = user_id
    session['is_admin'] = False
    logger.info("New user registered successfully: %s", user_id)
    return jsonify({'success': True})

@app.route('/auth/logout')
def auth_logout():
    user_id = session.get('user_id')
    session.clear()
    logger.info("User logged out: %s", user_id)
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
        'email': user.get('email', ''),
        'email_verified': user.get('email_verified', True),
        'profile_pic': user.get('profile_pic', ''),
        'samples': user.get('samples', {}),
        'total': user.get('total_accepted', 0),
        'rejected': user.get('rejected', 0),
        'created': user.get('created', ''),
        'is_admin': current_user_is_admin(),
        'role': 'admin' if user_is_admin(user) else 'user'
    })

# =============================================
# Email Verification & Profile Settings APIs
# =============================================
@app.route('/verify-email')
@login_required
def verify_email_page():
    return render_template('verify_email.html')

@app.route('/auth/verify-email', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def auth_verify_code():
    data = request.get_json(silent=True) or {}
    code = str(data.get('code', '')).strip()
    user_id = session['user_id']
    user = load_user(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'المستخدم غير موجود'}), 404

    if not code or user.get('verification_code') != code:
        return jsonify({'success': False, 'error': 'رمز التحقق غير صحيح'}), 400

    # Mark as verified
    user['email_verified'] = True
    user['verification_code'] = ''
    from auth import save_user, save_users, load_users
    save_user(user)

    # Update in users index as well
    users = load_users()
    if user_id in users:
        users[user_id]['email_verified'] = True
        save_users(users)

    logger.info("User %s email verified successfully", user_id)
    return jsonify({'success': True})

@app.route('/auth/resend-code', methods=['POST'])
@login_required
@limiter.limit("3 per minute")
def auth_resend_code():
    user_id = session['user_id']
    user = load_user(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'المستخدم غير موجود'}), 404

    if not user.get('email'):
        return jsonify({'success': False, 'error': 'لا يوجد بريد إلكتروني مسجل'}), 400

    import random
    new_code = f"{random.randint(100000, 999999)}"
    user['verification_code'] = new_code
    user['email_verified'] = False
    from auth import save_user, send_verification_email
    save_user(user)

    send_verification_email(user['email'], user['name'], new_code)
    logger.info("Resent verification code to user %s", user_id)
    return jsonify({'success': True})

@app.route('/auth/update_profile', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def auth_update_profile_api():
    data = request.get_json(silent=True) or {}
    user_id = session['user_id']
    
    from auth import update_profile
    new_user_id, error = update_profile(
        user_id,
        new_name=data.get('name', ''),
        current_password=data.get('current_password', ''),
        new_password=data.get('new_password'),
        new_email=data.get('email')
    )
    if error:
        return jsonify({'success': False, 'error': error}), 400

    # If username changed, update the session
    if new_user_id != user_id:
        session['user_id'] = new_user_id
        logger.info("Profile updated and user renamed from %s to %s", user_id, new_user_id)
    else:
        logger.info("Profile updated for user %s", user_id)
        
    return jsonify({'success': True})

@app.route('/profile/upload_avatar', methods=['POST'])
@login_required
def upload_avatar():
    if 'avatar' not in request.files:
        return jsonify({'success': False, 'error': 'لم يتم اختيار ملف'}), 400
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'لم يتم اختيار ملف'}), 400
    
    # Check file extension
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'صيغة الملف غير مدعومة (PNG, JPG, JPEG, GIF فقط)'}), 400

    # Ensure upload folder exists
    upload_dir = os.path.join('static', 'uploads', 'profile_pics')
    os.makedirs(upload_dir, exist_ok=True)

    user_id = session['user_id']
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{user_id}_{secrets.token_hex(4)}.{ext}" # use unique suffix to prevent browser cache issues
    filepath = os.path.join(upload_dir, filename)

    try:
        # Delete old profile pic if exists
        user = load_user(user_id)
        if user and user.get('profile_pic'):
            old_pic_path = user['profile_pic'].lstrip('/')
            # make sure it is relative and exists
            if old_pic_path.startswith('static/') and os.path.exists(old_pic_path):
                try:
                    os.remove(old_pic_path)
                except Exception as e:
                    logger.error("Failed to remove old profile pic: %s", e)

        file.save(filepath)
        
        # Update user profile pic path
        if user:
            user['profile_pic'] = f"/static/uploads/profile_pics/{filename}"
            from auth import save_user
            save_user(user)

        return jsonify({'success': True, 'profile_pic': user['profile_pic']})
    except Exception as e:
        logger.error("Failed to upload avatar: %s", e)
        return jsonify({'success': False, 'error': 'حدث خطأ أثناء حفظ الملف'}), 500

# =============================================
# Prediction & Collection APIs
# =============================================
@app.route('/predict', methods=['POST'])
@login_required
def predict_route():
    data = request.get_json(silent=True) or {}
    raw = data.get('landmarks', [])
    if len(raw) != 42:
        logger.warning("Predict request rejected: landmarks length is %d (expected 42)", len(raw))
        return jsonify({'error': 'Invalid landmarks'}), 400
    
    try:
        # Input Validation (Phase 2 & 9)
        float_raw = [float(v) for v in raw]
        if any(np.isnan(float_raw)) or any(np.isinf(float_raw)):
            logger.warning("Predict request rejected: landmarks contain NaN/Inf")
            return jsonify({'error': 'Invalid numeric values'}), 400
            
        normalized = normalize_landmarks(float_raw)
        probs = predict_landmarks(normalized)
        idx = int(np.argmax(probs))
        
        logger.debug("Successful prediction: class %d, confidence %f", idx, float(probs[idx]))
        return jsonify({
            'letter': labels_dict.get(idx, '?'),
            'confidence': round(float(probs[idx]) * 100, 1),
            'status': 'ok'
        })
    except Exception as exc:
        logger.error("Error in /predict: %s", exc, exc_info=True)
        return jsonify({'error': 'Internal prediction error'}), 500

@app.route('/collect', methods=['POST'])
@login_required
def collect_sample():
    data = request.get_json(silent=True) or {}
    if not data or 'landmarks' not in data or 'label' not in data:
        logger.warning("Collect request rejected: missing landmarks or label key")
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    user_id = session.get('user_id')
    raw = data.get('landmarks', [])
    
    try:
        label = int(data.get('label', -1))
    except (ValueError, TypeError):
        logger.warning("Collect request rejected: label '%s' is not integer", data.get('label'))
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400

    if label not in range(len(labels_dict)):
        logger.warning("Collect request rejected: label %d out of bounds", label)
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400

    if len(raw) != 42:
        logger.warning("Collect request rejected: landmarks length is %d (expected 42)", len(raw))
        return jsonify({'success': False, 'error': 'Invalid landmarks'})

    try:
        # Input float validation
        float_raw = [float(v) for v in raw]
        if any(np.isnan(float_raw)) or any(np.isinf(float_raw)):
            logger.warning("Collect request rejected: landmarks contain NaN/Inf")
            return jsonify({'success': False, 'error': 'Invalid landmark values'}), 400

        normalized = normalize_landmarks(float_raw)

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
                logger.info("Collected sample rejected by quality filter: label=%d, prediction=%d, confidence=%0.2f", label, idx, confidence)
                return jsonify({
                    'success': False,
                    'rejected': True,
                    'error': f'الإيماءة تشبه حرف {labels_dict.get(idx, "?")} — جرّب تاني'
                })

        # Save to main CSV with user tracking if logged in (Phase 3 metadata)
        os.makedirs(os.path.dirname(config.TRAIN_CSV), exist_ok=True)
        with open(config.TRAIN_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if user_id:
                writer.writerow([user_id, label] + normalized)
            else:
                writer.writerow([label] + normalized)

        if user_id:
            record_sample(user_id, label)

        logger.info("Collected sample recorded: user=%s, label=%d", user_id or "anonymous", label)
        return jsonify({'success': True})

    except Exception as exc:
        logger.error("Error in /collect route: %s", exc, exc_info=True)
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
        logger.info("Admin verification successful from IP %s", request.remote_addr)
    else:
        logger.warning("Failed admin login attempt from IP %s", request.remote_addr)
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
    user_to_del = data.get('user_id')
    delete_user(user_to_del)
    logger.info("Admin deleted user account: %s", user_to_del)
    return jsonify({'ok': True})

@app.route('/admin/disable_user', methods=['POST'])
def admin_disable_user():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    user_to_dis = data.get('user_id')
    from auth import disable_user
    disable_user(user_to_dis)
    logger.info("Admin disabled user account: %s", user_to_dis)
    return jsonify({'ok': True})

@app.route('/admin/enable_user', methods=['POST'])
def admin_enable_user():
    if not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    user_to_en = data.get('user_id')
    from auth import enable_user
    enable_user(user_to_en)
    logger.info("Admin enabled user account: %s", user_to_en)
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

    logger.info("Admin changed role of user %s to: %s", user_id, role)
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
    logger.info("Admin updated settings: %s", GLOBAL_SETTINGS)
    return jsonify({'ok': True, 'settings': GLOBAL_SETTINGS})

@app.route('/admin/export')
def admin_export():
    if not current_user_is_admin():
        logger.warning("Unauthorized admin export request from IP %s", request.remote_addr)
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        users = get_all_users_stats()
        total_samples = sum(u['total'] for u in users)
        logger.info("Admin export statistics retrieved successfully: %d users, %d samples", len(users), total_samples)
        return jsonify({'users': users, 'total_users': len(users),
                        'total_samples': total_samples})
    except Exception as exc:
        logger.error("Admin export failure: %s", exc)
        return jsonify({'error': 'Export failed'}), 500

@app.route('/pose-editor/save', methods=['POST'])
@login_required
def pose_editor_save():
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
    logger.info("Pose editor updated configuration for letter: %s by user: %s", letter, user_name)
    return jsonify({'success': True})

# =============================================
# Test Data Collection (no auth required)
# =============================================
TEST_CSV = config.TEST_CSV
_test_csv_lock = threading.Lock()

@app.route('/collect-test')
@app.route('/collect_test')
def collect_test_page():
    return render_template('collect_test.html')

@app.route('/collect-test', methods=['POST'])
@app.route('/collect_test', methods=['POST'])
@limiter.limit("30 per minute")
def collect_test_sample():
    data = request.get_json(silent=True) or {}
    raw = data.get('landmarks', [])
    label = data.get('label')
    tester_id = str(data.get('tester_id', '')).strip()

    if not tester_id or label is None:
        return jsonify({'success': False, 'error': 'بيانات ناقصة'}), 400
    try:
        label = int(label)
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400

    if label not in range(len(labels_dict)):
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400
    if len(raw) != 42:
        return jsonify({'success': False, 'error': 'Invalid landmarks'}), 400

    try:
        # Validate elements are floats
        float_raw = [float(v) for v in raw]
        if any(np.isnan(float_raw)) or any(np.isinf(float_raw)):
            return jsonify({'success': False, 'error': 'Invalid landmark coordinate values'}), 400

        normalized = normalize_landmarks(float_raw)
        os.makedirs(os.path.dirname(TEST_CSV), exist_ok=True)
        with _test_csv_lock:
            with open(TEST_CSV, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([tester_id, label] + normalized)
            
            # Count samples for this tester+label
            total = 0
            if os.path.exists(TEST_CSV):
                with open(TEST_CSV, 'r', encoding='utf-8') as f:
                    for row in csv.reader(f):
                        if len(row) >= 2 and row[0] == tester_id and row[1] == str(label):
                            total += 1
        
        logger.info("Test sample recorded successfully: tester=%s, label=%d", tester_id, label)
        return jsonify({'success': True, 'total': total})
    except Exception as exc:
        logger.error("Error in /collect-test POST: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'حدث خطأ'}), 500

@app.route('/collect-test/export')
@app.route('/collect_test/export')
def collect_test_export():
    if not os.path.exists(TEST_CSV):
        logger.warning("Export test CSV requested but file does not exist.")
        return jsonify({'error': 'لا توجد بيانات بعد'}), 404
    try:
        with open(TEST_CSV, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in csv.reader(f))
        if row_count < 50:
            logger.warning("Export test CSV blocked: insufficient samples count %d < 50", row_count)
            return jsonify({'error': f'عدد العينات غير كافٍ ({row_count}/50)'}), 403
    except Exception as exc:
        logger.error("Failed to read test CSV for export: %s", exc)
        return jsonify({'error': 'خطأ في قراءة الملف'}), 500
        
    logger.info("Exporting test CSV containing %d samples.", row_count)
    return send_from_directory(os.path.dirname(TEST_CSV), os.path.basename(TEST_CSV),
                               as_attachment=True,
                               download_name='test_keypoints.csv')

@app.route('/forgot-password')
@app.route('/forgot_password')
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route('/auth/forgot-password', methods=['POST'])
@limiter.limit("5 per minute")
def auth_forgot_password():
    data = request.get_json(silent=True) or {}
    identifier = data.get('identifier', '')
    if not identifier:
        return jsonify({'success': False, 'error': 'يرجى إدخال اسم المستخدم أو البريد الإلكتروني'}), 400

    from auth import request_password_reset
    ok, error = request_password_reset(identifier)
    if not ok:
        return jsonify({'success': False, 'error': error}), 400

    return jsonify({'success': True, 'message': 'تم إرسال رمز إعادة التعيين لبريدك الإلكتروني'})

@app.route('/auth/reset-password', methods=['POST'])
@limiter.limit("5 per minute")
def auth_reset_password():
    data = request.get_json(silent=True) or {}
    identifier = data.get('identifier', '')
    code = data.get('code', '')
    new_password = data.get('new_password', '')

    if not identifier or not code or not new_password:
        return jsonify({'success': False, 'error': 'يرجى ملء جميع الحقول المطلوبة'}), 400

    from auth import reset_password_with_code
    ok, error = reset_password_with_code(identifier, code, new_password)
    if not ok:
        return jsonify({'success': False, 'error': error}), 400

    return jsonify({'success': True, 'message': 'تم تغيير كلمة المرور بنجاح. يمكنك تسجيل الدخول الآن.'})

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
