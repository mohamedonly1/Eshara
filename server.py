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
from flask_wtf.csrf import CSRFProtect, CSRFError, generate_csrf
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

# =============================================
# CSRF Protection (SEC-01)
# =============================================
csrf = CSRFProtect(app)

@app.after_request
def set_csrf_cookie(response):
    """Expose the CSRF token as a cookie so JavaScript fetch calls can read and send it."""
    response.set_cookie('csrf_token', generate_csrf(), samesite='Lax', httponly=False)
    return response

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    """Return a clear JSON or HTML error when a CSRF token is missing or invalid."""
    wants_json = (
        request.is_json or
        request.headers.get('Accept', '').find('application/json') != -1 or
        request.path.startswith('/api/') or
        request.path.startswith('/auth/') or
        request.path.startswith('/admin/')
    )
    if wants_json:
        return jsonify({'error': 'CSRF token missing or invalid', 'csrf_error': True}), 400
    return redirect('/login')

_session_secret = os.getenv('FLASK_SECRET_KEY')
if not _session_secret:
    _env_name = os.getenv('FLASK_ENV') or os.getenv('APP_ENV') or os.getenv('ENV') or ''
    if _env_name.lower() in {'prod', 'production'}:
        raise RuntimeError('FLASK_SECRET_KEY must be set in production')
    _session_secret = secrets.token_hex(32)
    logger.warning("FLASK_SECRET_KEY is not set; using a temporary development secret.")
app.secret_key = _session_secret

# =============================================
# Translation & Internationalization System (Part 3)
# =============================================
TRANSLATIONS = {}

# Maps sign-language dialect codes → UI display language code.
# Dialects of Arabic all show Arabic UI; other languages map to themselves.
SIGN_LANG_TO_UI_LANG = {
    # Arabic dialects → Arabic UI
    'ar': 'ar', 'sa': 'ar', 'eg': 'ar', 'ma': 'ar', 'ly': 'ar',
    'tn': 'ar', 'dz': 'ar', 'jo': 'ar', 'iq': 'ar', 'sy': 'ar',
    'ye': 'ar', 'lb': 'ar', 'kw': 'ar', 'bh': 'ar', 'qa': 'ar',
    'ae': 'ar', 'om': 'ar', 'ps': 'ar', 'sd': 'ar',
    # Major languages → their own UI
    'en': 'en', 'fr': 'fr', 'es': 'es', 'de': 'de', 'tr': 'tr',
    'ur': 'ur', 'fa': 'fa', 'zh': 'zh', 'hi': 'hi', 'bn': 'bn',
    'ru': 'ru', 'pt': 'pt', 'it': 'it', 'nl': 'nl', 'pl': 'pl',
    'id': 'id', 'ms': 'ms', 'ko': 'ko', 'ja': 'ja', 'sw': 'sw',
}

# Tracks background translation-generation jobs: {lang_code: 'pending'|'done'|'failed'|'exists'}
_translation_jobs = {}
_translation_jobs_lock = threading.Lock()

def load_translations():
    """Dynamically load ALL translation JSON files found in the translations/ folder."""
    global TRANSLATIONS
    translations_dir = os.path.join(app.root_path, 'translations')
    if not os.path.exists(translations_dir):
        logger.warning("Translations directory not found: %s", translations_dir)
        return
    for filename in os.listdir(translations_dir):
        if not filename.endswith('.json'):
            continue
        lang = filename[:-5]
        path = os.path.join(translations_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                TRANSLATIONS[lang] = json.load(f)
            logger.info("Loaded translation: %s (%d keys)", lang, len(TRANSLATIONS[lang]))
        except Exception as e:
            logger.error("Failed to load translation file %s: %s", path, e)
            TRANSLATIONS[lang] = {}

load_translations()

def t(key, lang=None, **kwargs):
    if not lang:
        try:
            lang = session.get('active_lang')
            if not lang:
                lang = languages_config.get('active_language', 'ar')
        except Exception:
            lang = 'ar'
    if lang not in TRANSLATIONS:
        lang = 'ar'
    val = TRANSLATIONS.get(lang, {}).get(key)
    if val is None:
        # fallback to ar
        val = TRANSLATIONS.get('ar', {}).get(key, key)
    if kwargs and isinstance(val, str):
        try:
            val = val.format(**kwargs)
        except Exception:
            pass
    return val

@app.context_processor
def inject_t():
    try:
        lang = session.get('active_lang')
        if not lang:
            lang = languages_config.get('active_language', 'ar')
    except Exception:
        lang = 'ar'
    return dict(
        t=lambda key, **kwargs: t(key, lang=lang, **kwargs),
        active_lang=lang
    )

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
# Model Setup (Multi-language aware)
# =============================================
# This variable name and path is preserved exactly for train_model.py regex update compatibility
MODEL_PATH = 'arabic_model/ar_sign_model_2026-06-12_95.97.tflite'

LABELS_PATH = config.LABELS_CSV
POSES_PATH = config.POSES_JS

# Legacy fallback references
labels_dict = {}
interpreter = None
_interpreter_lock = threading.Lock()

LANGUAGES_CONFIG_PATH = 'languages_data/languages.json'
languages_config = {}
interpreters = {}
_interpreters_lock = threading.Lock()
# Protects all read-modify-write operations on languages.json
_languages_config_lock = threading.Lock()
# Tracks languages whose model failed validation (mismatch or missing)
_invalid_lang_reasons: dict = {}
language_health: dict = {}
# Prevents simultaneous training for the same language
_training_state_lock = threading.Lock()
_training_in_progress: dict = {}
# Prevents duplicate server restart on rapid button clicks
_restart_lock = threading.Lock()
_restart_in_progress = False

def cleanup_expired_deleted_languages_in_place(config_data):
    """Checks config_data['deleted_languages'] and removes/deletes expired entries."""
    import datetime
    deleted_langs = config_data.get('deleted_languages', {})
    if not deleted_langs:
        return False
        
    now = datetime.datetime.now()
    expired_codes = []
    for code, info in list(deleted_langs.items()):
        del_at_str = info.get('deleted_at')
        if del_at_str:
            try:
                del_at = datetime.datetime.strptime(del_at_str, "%Y-%m-%d %H:%M:%S")
                if (now - del_at).days >= 30:
                    expired_codes.append(code)
            except Exception as e:
                logger.error("Error parsing deleted_at for %s: %s", code, e)
                
    if not expired_codes:
        return False
        
    for code in expired_codes:
        info = deleted_langs.pop(code, None)
        if info:
            for path_key in ['model_path', 'labels_path', 'dataset_path']:
                path = info.get(path_key)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info("Permanently deleted expired file: %s", path)
                    except Exception as e:
                        logger.error("Failed to delete expired file %s: %s", path, e)
                        
    logger.info("Cleaned up expired deleted languages: %s", expired_codes)
    return True

def load_languages_config():
    global languages_config
    if os.path.exists(LANGUAGES_CONFIG_PATH):
        try:
            with open(LANGUAGES_CONFIG_PATH, 'r', encoding='utf-8') as f:
                languages_config = json.load(f)
        except Exception as e:
            logger.error("Failed to parse languages.json: %s", e)
    
    if not languages_config or 'languages' not in languages_config:
        languages_config = {
            "active_language": "ar",
            "languages": {
                "ar": {
                    "code": "ar",
                    "name": "العربية",
                    "labels": ["أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "لا"],
                    "model_path": MODEL_PATH,
                    "labels_path": "languages_data/ar/ar_labels.csv",
                    "dataset_path": "languages_data/ar/ar_keypoints.csv"
                }
            }
        }
        
    try:
        if cleanup_expired_deleted_languages_in_place(languages_config):
            tmp_path = LANGUAGES_CONFIG_PATH + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(languages_config, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, LANGUAGES_CONFIG_PATH)
    except Exception as e:
        logger.error("Failed to run cleanup of deleted languages: %s", e)
        
    return languages_config

def get_lang_labels(lang_info):
    if not lang_info:
        return []
    labels_path = lang_info.get('labels_path')
    if labels_path and os.path.exists(labels_path):
        labels = []
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        labels.append(row[1])
                    elif len(row) == 1:
                        labels.append(row[0])
            if labels:
                return labels
        except Exception as e:
            logger.error("Failed to read labels from %s: %s", labels_path, e)
    
    # Fallback to configured labels list, and if empty/missing check parent
    labels = lang_info.get('labels', [])
    if not labels and 'parent' in lang_info:
        parent_code = lang_info['parent']
        parent_info = languages_config['languages'].get(parent_code)
        if parent_info:
            return get_lang_labels(parent_info)
    return labels

def get_language_status(lang_code):
    """Calculates active/not_trained/invalid_model/missing_dataset status for a language."""
    config_data = load_languages_config()
    lang_info = config_data.get('languages', {}).get(lang_code)
    if not lang_info:
        return 'disabled'
    
    if lang_info.get('disabled', False) or lang_info.get('status') == 'disabled':
        return 'disabled'
        
    model_path = lang_info.get('model_path', '')
    labels_path = lang_info.get('labels_path', '')
    dataset_path = lang_info.get('dataset_path', '')
    
    model_exists = bool(model_path and os.path.exists(model_path))
    labels_exist = bool(labels_path and os.path.exists(labels_path))
    dataset_exists = bool(dataset_path and os.path.exists(dataset_path))
    
    # MISSING_DATASET: labels exist, dataset missing
    if labels_exist and not dataset_exists:
        return 'missing_dataset'
        
    # NOT_TRAINED: dataset exists, labels exist, model missing
    if dataset_exists and labels_exist and not model_exists:
        return 'not_trained'
        
    # INVALID_MODEL / ACTIVE
    if model_exists:
        with _interpreters_lock:
            loaded = lang_code in interpreters
        if loaded:
            return 'active'
        else:
            return 'invalid_model'
            
    return 'disabled'

def validate_labels_encoding():
    """Validates that all language label files are UTF-8 compliant and contain no mojibake."""
    config_data = load_languages_config()
    errors = {}
    for code, info in config_data.get('languages', {}).items():
        labels_path = info.get('labels_path')
        if not labels_path:
            continue
        if not os.path.exists(labels_path):
            errors[code] = "Label file missing"
            continue
        try:
            with open(labels_path, 'rb') as f:
                content = f.read()
            try:
                decoded = content.decode('utf-8')
            except UnicodeDecodeError:
                errors[code] = "Invalid UTF-8 encoding"
                continue
            
            # CP1252 / mojibake indicator check for Arabic or other text
            mojibake_chars = ['Ø', 'Ù', 'Â', 'Ã', 'æ', 'ç', 'è', 'é']
            is_mojibake = False
            for char in mojibake_chars:
                if code == 'fr' and char in ['ç', 'è', 'é']:
                    continue
                if char in decoded:
                    if decoded.count(char) > 2:
                        is_mojibake = True
                        break
            if is_mojibake:
                errors[code] = "Mojibake/corruption detected in label file encoding"
            else:
                errors[code] = ""
        except Exception as e:
            errors[code] = f"Encoding check failed: {e}"
    return errors

def init_interpreters():
    """Load TFLite interpreters for every configured language and validate their integrity.

    Safety rules (Phase 1 hardening):
    - NEVER auto-copy an Arabic placeholder model for another language.
    - NEVER load an interpreter whose output class count does not match
      the configured label count — this would produce silently wrong results.
    - Arabic model loading is unchanged; only the guard blocks are new.
    """
    global interpreters, interpreter, labels_dict, _invalid_lang_reasons, language_health
    config_data = load_languages_config()
    _invalid_lang_reasons = {}
    with _interpreters_lock:
        interpreters.clear()
        language_health.clear()
        for lang_code, lang_info in config_data['languages'].items():
            model_path = lang_info.get('model_path', '')
            labels = get_lang_labels(lang_info)
            label_count = len(labels)

            # Check parent fallback if model path is missing or doesn't exist
            if (not model_path or not os.path.exists(model_path)) and 'parent' in lang_info:
                parent_code = lang_info['parent']
                parent_info = config_data['languages'].get(parent_code)
                if parent_info:
                    parent_model_path = parent_info.get('model_path', '')
                    if parent_model_path and os.path.exists(parent_model_path):
                        model_path = parent_model_path
                        logger.info("Dialect '%s' falling back to parent '%s' model path: %s", lang_code, parent_code, model_path)

            if not model_path:
                reason = "Model path is empty"
                _invalid_lang_reasons[lang_code] = reason
                language_health[lang_code] = {
                    "model_classes": 0,
                    "label_count": label_count,
                    "valid": False,
                    "reason": reason
                }
                continue

            # --- Guard 1: model file must exist ---
            if not os.path.exists(model_path):
                reason = f"Model file not found: {model_path}"
                _invalid_lang_reasons[lang_code] = reason
                language_health[lang_code] = {
                    "model_classes": 0,
                    "label_count": label_count,
                    "valid": False,
                    "reason": reason
                }
                logger.warning("[%s] %s — language will show as Not Trained.", lang_code, reason)
                continue

            try:
                logger.info("Loading TFLite model for %s from path: %s", lang_code, model_path)
                interp = tf.lite.Interpreter(model_path=model_path)
                interp.allocate_tensors()

                # --- Guard 2: output class count must match label count ---
                out_classes = int(interp.get_output_details()[0]['shape'][-1])
                if label_count != out_classes:
                    reason = (f"Model outputs {out_classes} classes but "
                              f"label list has {label_count} entries")
                    _invalid_lang_reasons[lang_code] = reason
                    language_health[lang_code] = {
                        "model_classes": out_classes,
                        "label_count": label_count,
                        "valid": False,
                        "reason": reason
                    }
                    logger.warning(
                        "[%s] MISMATCH — %s. "
                        "Interpreter NOT loaded; language marked unavailable.",
                        lang_code, reason
                    )
                    continue  # do NOT add to interpreters

                interpreters[lang_code] = {
                    'interpreter': interp,
                    'input_details': interp.get_input_details(),
                    'output_details': interp.get_output_details(),
                    'lock': threading.Lock()
                }
                language_health[lang_code] = {
                    "model_classes": out_classes,
                    "label_count": label_count,
                    "valid": True,
                    "reason": ""
                }
                logger.info("Loaded interpreter for language: %s (%d classes)", lang_code, out_classes)
            except Exception as e:
                reason = f"TFLite load error: {e}"
                _invalid_lang_reasons[lang_code] = reason
                language_health[lang_code] = {
                    "model_classes": 0,
                    "label_count": label_count,
                    "valid": False,
                    "reason": reason
                }
                logger.error("Failed to load interpreter for %s: %s", lang_code, e)

        # Populate legacy fallback references (Arabic only — no cross-language fallback)
        if 'ar' in interpreters:
            interpreter = interpreters['ar']['interpreter']
            ar_info = config_data['languages']['ar']
            labels_dict = {idx: l for idx, l in enumerate(get_lang_labels(ar_info))}

init_interpreters()

def get_active_labels():
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    lang_info = languages_config['languages'].get(lang_code, languages_config['languages'].get('ar'))
    if lang_info:
        labels = get_lang_labels(lang_info)
        return {idx: label for idx, label in enumerate(labels)}
    return labels_dict

def get_active_labels_list():
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    lang_info = languages_config['languages'].get(lang_code, languages_config['languages'].get('ar'))
    if lang_info:
        return get_lang_labels(lang_info)
    return list(labels_dict.values())

def get_active_dataset_path():
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    lang_info = languages_config['languages'].get(lang_code, languages_config['languages'].get('ar'))
    if lang_info:
        dataset_path = lang_info.get('dataset_path')
        if not dataset_path and 'parent' in lang_info:
            parent_code = lang_info['parent']
            parent_info = languages_config['languages'].get(parent_code)
            if parent_info:
                dataset_path = parent_info.get('dataset_path')
        if dataset_path:
            return dataset_path
    return config.TRAIN_CSV

def get_active_test_dataset_path(lang_code=None):
    if not lang_code:
        lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    lang_info = languages_config['languages'].get(lang_code, languages_config['languages'].get('ar'))
    if lang_info:
        if 'test_dataset_path' in lang_info:
            return lang_info['test_dataset_path']
        if 'parent' in lang_info:
            parent_code = lang_info['parent']
            parent_info = languages_config['languages'].get(parent_code)
            if parent_info and 'test_dataset_path' in parent_info:
                return parent_info['test_dataset_path']
    if lang_code == 'ar':
        return config.TEST_CSV
    return f"languages_data/{lang_code}/{lang_code}_test_keypoints.csv"

_means_cache = {}
_means_mtime = {}

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

def predict_landmarks(landmarks, lang_code=None):
    """Runs thread-safe TFLite interpreter inference for the active language.

    Phase 1 hardening: NEVER falls back to Arabic silently. If the requested
    language's interpreter is not loaded (missing model or validation failure),
    a RuntimeError is raised so callers can return a clear API error.
    Arabic behaviour is completely unchanged.
    """
    if not lang_code:
        try:
            lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
        except Exception:
            lang_code = languages_config.get('active_language', 'ar')

    with _interpreters_lock:
        lang_item = interpreters.get(lang_code)

        if lang_item is None:
            reason = _invalid_lang_reasons.get(lang_code, 'model not loaded')
            raise RuntimeError(f"Model not available for language: {lang_code} ({reason})")

    # Serialize calls to this specific interpreter; allows concurrency across languages.
    lock = lang_item['lock']
    with lock:
        interp = lang_item['interpreter']
        in_details = lang_item['input_details']
        out_details = lang_item['output_details']

        # Dynamically determine feature mode from model input dimensions
        expected_size = in_details[0]['shape'][-1]
        if expected_size == 104:
            derived = config.extract_derived_features(landmarks)
            input_data = np.array([landmarks + derived], dtype=np.float32)
        else:
            input_data = np.array([landmarks], dtype=np.float32)

        interp.set_tensor(in_details[0]['index'], input_data)
        interp.invoke()
        return interp.get_tensor(out_details[0]['index'])[0].copy()

# =============================================
# Pages
# =============================================
@app.route('/')
@login_required
def index():
    active_lang = session.get('active_lang', languages_config.get('active_language', 'ar'))
    with _interpreters_lock:
        model_available = (active_lang in interpreters)
    return render_template(
        'index.html',
        labels_map=get_active_labels(),
        letters=get_active_labels_list(),
        active_lang=active_lang,
        model_available=model_available
    )

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
    active_lang = session.get('active_lang', languages_config.get('active_language', 'ar'))
    return render_template('profile.html', letters=get_active_labels_list(), active_lang=active_lang)

@app.route('/collect-data')
@login_required
def collect_page():
    active_lang = session.get('active_lang', languages_config.get('active_language', 'ar'))
    return render_template('collect.html', letters=get_active_labels_list(), active_lang=active_lang)

@app.route('/translate')
@login_required
def translate_page():
    active_lang = session.get('active_lang', languages_config.get('active_language', 'ar'))
    with _interpreters_lock:
        model_available = (active_lang in interpreters)
    return render_template(
        'translate.html',
        labels_map=get_active_labels(),
        letters=get_active_labels_list(),
        active_lang=active_lang,
        model_available=model_available
    )

@app.route('/means')
@login_required
def means_route():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    global _means_cache, _means_mtime

    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    path = get_active_dataset_path()
    if not os.path.exists(path):
        _means_cache[lang_code] = {}
        _means_mtime[lang_code] = None
        return jsonify({})

    try:
        file_mtime = os.path.getmtime(path)
        if _means_cache.get(lang_code) is not None and _means_mtime.get(lang_code) == file_mtime:
            return jsonify(_means_cache[lang_code])

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
            _means_cache[lang_code] = {}
            _means_mtime[lang_code] = file_mtime
            return jsonify({})

        df = pd.DataFrame(rows_data)
        raw_labels = df.iloc[:, 0]
        features = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Keep LabelEncoder logic for parity with training-time class handling.
        encoder = LabelEncoder()
        labels_map = get_active_labels()
        encoder.fit([labels_map[idx] for idx in sorted(labels_map.keys())])

        letter_to_index = {letter: int(idx) for idx, letter in labels_map.items()}
        aligned_labels = []
        for raw in raw_labels:
            idx = None
            try:
                numeric = int(float(raw))
                if numeric in labels_map:
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
            _means_cache[lang_code] = {}
            _means_mtime[lang_code] = file_mtime
            return jsonify({})

        medians = grouped.groupby('label', sort=True).median(numeric_only=True)

        result = {}
        for idx, row in medians.iterrows():
            result[str(int(idx))] = [float(v) for v in row.tolist()]

        _means_cache[lang_code] = result
        _means_mtime[lang_code] = file_mtime
        return jsonify(result)
    except Exception as exc:
        logger.error("Error in /means: %s", exc, exc_info=True)
        return jsonify({'error': 'تعذّر تحميل المتوسطات'}), 500

@app.route('/pose-editor')
@login_required
@admin_required
def pose_editor_page():
    active_lang = session.get('active_lang', languages_config.get('active_language', 'ar'))
    return render_template('pose_editor.html', letters=get_active_labels_list(), active_lang=active_lang)

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
        
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    all_samples = user.get('samples', {})
    
    # Migrate flat dict if needed
    has_flat = False
    for k in list(all_samples.keys()):
        if k.isdigit():
            has_flat = True
            break
    if has_flat:
        all_samples = {'ar': {k: v for k, v in all_samples.items() if k.isdigit()}}
        user['samples'] = all_samples
        from auth import save_user
        save_user(user)
        
    lang_samples = all_samples.get(lang_code, {})
    lang_accepted = sum(lang_samples.values())
    
    # Lazy migration for statistics loaded on the fly
    if 'rejected_by_lang' not in user:
        user['rejected_by_lang'] = {}
    if 'ar' not in user['rejected_by_lang'] and 'rejected' in user:
        user['rejected_by_lang']['ar'] = user['rejected']
        from auth import save_user
        save_user(user)
        
    lang_rejected = user['rejected_by_lang'].get(lang_code, 0)
    
    return jsonify({
        'logged_in': True,
        'name': user['name'],
        'email': user.get('email', ''),
        'email_verified': user.get('email_verified', True),
        'pending_email': user.get('pending_email', ''),
        'profile_pic': user.get('profile_pic', ''),
        'samples': lang_samples,
        'total': lang_accepted,
        'rejected': lang_rejected,
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

    # Validate code
    if not code or user.get('verification_code') != code:
        return jsonify({'success': False, 'error': 'رمز التحقق غير صحيح'}), 400

    # Check expiration
    from datetime import datetime
    expires_str = user.get('verification_code_expires', '')
    if expires_str:
        try:
            expires_at = datetime.strptime(expires_str, '%Y-%m-%d %H:%M')
            if datetime.now() > expires_at:
                return jsonify({'success': False, 'error': 'رمز التحقق منتهي الصلاحية. أعد إرسال رمز جديد.'}), 400
        except Exception:
            pass

    from auth import save_user, save_users, load_users

    # SEC-05: If there's a pending_email, this is an email change verification
    pending = user.get('pending_email', '')
    if pending:
        user['email'] = pending
        user['pending_email'] = ''
        user['email_verified'] = True
        user['verification_code'] = ''
        user['verification_code_expires'] = ''
        save_user(user)

        users = load_users()
        if user_id in users:
            users[user_id]['email'] = pending
            users[user_id]['email_verified'] = True
            save_users(users)

        logger.info("User %s email changed and verified to %s", user_id, pending)
        return jsonify({'success': True, 'email_changed': True})

    # Standard registration verification
    user['email_verified'] = True
    user['verification_code'] = ''
    user['verification_code_expires'] = ''
    save_user(user)

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

    from auth import generate_secure_otp, save_user, send_verification_email
    from datetime import datetime, timedelta
    new_code = generate_secure_otp()
    user['verification_code'] = new_code
    user['verification_code_expires'] = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')

    # Send to pending email if one exists, otherwise to primary email
    target_email = user.get('pending_email') or user.get('email')
    if not user.get('pending_email'):
        user['email_verified'] = False
    save_user(user)

    send_verification_email(target_email, user['name'], new_code)
    logger.info("Resent verification code to user %s (target: %s)", user_id, target_email)
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

    # Check if a pending email verification was initiated
    updated_user = load_user(new_user_id)
    pending = updated_user.get('pending_email', '') if updated_user else ''
    return jsonify({'success': True, 'pending_email_verification': bool(pending)})

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
        # Input Validation
        float_raw = [float(v) for v in raw]
        if any(np.isnan(float_raw)) or any(np.isinf(float_raw)):
            logger.warning("Predict request rejected: landmarks contain NaN/Inf")
            return jsonify({'error': 'Invalid numeric values'}), 400

        normalized = normalize_landmarks(float_raw)

        lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
        status = get_language_status(lang_code)
        if status != 'active':
            return jsonify({
                "error": "Language not ready",
                "status": status
            }), 400

        probs = predict_landmarks(normalized, lang_code=lang_code)
        idx = int(np.argmax(probs))

        labels_map = get_active_labels()

        logger.debug("Successful prediction: class %d, confidence %f", idx, float(probs[idx]))
        return jsonify({
            'letter': labels_map.get(idx, '?'),
            'confidence': round(float(probs[idx]) * 100, 1),
            'status': 'ok'
        })
    except RuntimeError as exc:
        # Model unavailable for this language — surface a clear 503 (not a generic 500)
        logger.warning("Predict blocked for unavailable model: %s", exc)
        return jsonify({'error': str(exc), 'status': 'unavailable'}), 503
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
    
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    labels_map = get_active_labels()
    
    try:
        label = int(data.get('label', -1))
    except (ValueError, TypeError):
        logger.warning("Collect request rejected: label '%s' is not integer", data.get('label'))
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400

    if label not in range(len(labels_map)):
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
            try:
                probs = predict_landmarks(normalized, lang_code=lang_code)
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                if idx != label and confidence > filter_threshold:
                    if user_id:
                        record_rejected(user_id, lang_code=lang_code)
                    logger.info("Collected sample rejected by quality filter: label=%d, prediction=%d, confidence=%0.2f", label, idx, confidence)
                    return jsonify({
                        'success': False,
                        'rejected': True,
                        'error': f'الإيماءة تشبه حرف {labels_map.get(idx, "?")} — جرّب تاني'
                    })
            except RuntimeError as qf_err:
                # No model for this language yet — skip quality filter, still save sample
                logger.info("Quality filter skipped for %s (no model): %s", lang_code, qf_err)

        # Save to main CSV with user tracking if logged in (Phase 3 metadata)
        dataset_path = get_active_dataset_path()
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if user_id:
                writer.writerow([user_id, label] + normalized)
            else:
                writer.writerow([label] + normalized)

        if user_id:
            record_sample(user_id, label, lang_code=lang_code)

        logger.info("Collected sample recorded: user=%s, label=%d, lang=%s", user_id or "anonymous", label, lang_code)
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
    
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    all_samples = user.get('samples', {})
    
    # Migrate flat dict if needed
    has_flat = False
    for k in list(all_samples.keys()):
        if k.isdigit():
            has_flat = True
            break
    if has_flat:
        all_samples = {'ar': {k: v for k, v in all_samples.items() if k.isdigit()}}
        user['samples'] = all_samples
        from auth import save_user
        save_user(user)
        
    lang_samples = all_samples.get(lang_code, {})
    counts = {int(k): v for k, v in lang_samples.items()}
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

# Maps UI lang codes (as typed by admin) → correct ISO 639-1/BCP-47 code for MyMemory API
_LANG_TO_API_CODE = {
    # Common alternate codes admins might type
    'jp': 'ja',   'jap': 'ja',
    'cn': 'zh-CN', 'zh': 'zh-CN', 'zht': 'zh-TW',
    'kr': 'ko',   'kor': 'ko',
    'gr': 'el',   'gre': 'el',
    'dk': 'da',   'dan': 'da',
    'cz': 'cs',   'cze': 'cs',
    'se': 'sv',   'swe': 'sv',
    'no': 'no',
    'fi': 'fi',
    'pl': 'pl',
    'ro': 'ro',
    'hu': 'hu',
    'sk': 'sk',
    'bg': 'bg',
    'hr': 'hr',
    'sr': 'sr',
    'uk': 'uk',
    'he': 'he',   'heb': 'he',
    'fa': 'fa',   'per': 'fa',
    'ur': 'ur',
    'hi': 'hi',   'hin': 'hi',
    'bn': 'bn',   'ben': 'bn',
    'ta': 'ta',   'tel': 'te',
    'ml': 'ml',
    'th': 'th',   'tha': 'th',
    'vi': 'vi',   'vie': 'vi',
    'id': 'id',   'ind': 'id',
    'ms': 'ms',   'may': 'ms',
    'sw': 'sw',   'swa': 'sw',
    'tr': 'tr',   'tur': 'tr',
    'ru': 'ru',   'rus': 'ru',
    'pt': 'pt',   'por': 'pt',
    'es': 'es',   'spa': 'es',
    'de': 'de',   'deu': 'de',
    'fr': 'fr',   'fre': 'fr',
    'it': 'it',   'ita': 'it',
    'nl': 'nl',   'dut': 'nl',
    'en': 'en',   'eng': 'en',
    'ar': 'ar',   'ara': 'ar',
}

def _do_generate_translation(ui_lang: str):
    """
    Background worker: translates en.json into `ui_lang` using Google Translate API
    as primary engine (unlimited, fast) with MyMemory API as fallback.
    Saves result to translations/{ui_lang}.json.
    """
    import urllib.request as _urlreq
    import urllib.parse as _urlparse
    import time as _time

    with _translation_jobs_lock:
        _translation_jobs[ui_lang] = 'pending'

    # Resolve source: prefer English, fall back to Arabic
    source = TRANSLATIONS.get('en') or TRANSLATIONS.get('ar', {})
    if not source:
        with _translation_jobs_lock:
            _translation_jobs[ui_lang] = 'failed'
        logger.error("Auto-translate: no source translations loaded, cannot generate '%s'", ui_lang)
        return

    source_lang = 'en' if 'en' in TRANSLATIONS else 'ar'

    # Resolve correct API language code
    api_code = _LANG_TO_API_CODE.get(ui_lang.lower(), ui_lang)
    logger.info("Auto-translate: starting %s → %s (using code: %s)", source_lang, ui_lang, api_code)

    from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

    # Keys to keep as-is (app name, technical placeholders, etc.)
    SKIP_TRANSLATE = {'app_name', 'translate_letter_placeholder'}

    translated = {}
    failed_keys = []

    def google_translate(text, target, source='en'):
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source}&tl={target}&dt=t&q={_urlparse.quote(text)}"
            req = _urlreq.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with _urlreq.urlopen(req, timeout=5) as response:
                res = json.loads(response.read().decode('utf-8'))
                return ''.join([part[0] for part in res[0]])
        except Exception as e:
            return None

    def mymemory_translate(text, target, source='en'):
        try:
            import requests as _req
            resp = _req.get(
                'https://api.mymemory.translated.net/get',
                params={'q': text, 'langpair': f'{source}|{target}'},
                timeout=4
            )
            data = resp.json()
            if data.get('responseStatus') == 200:
                return data['responseData']['translatedText']
        except Exception:
            pass
        return None

    def translate_single_key(item):
        key, value = item
        if not isinstance(value, str) or not value.strip() or key in SKIP_TRANSLATE:
            return key, value, True

        # Try Google Translate
        res = google_translate(value, api_code, source_lang)
        if not res:
            # Fallback to MyMemory
            res = mymemory_translate(value, api_code, source_lang)

        if res and res.strip() and res.lower() != value.lower():
            # Fix brace formatting (e.g. { letter } -> {letter})
            import re
            fixed_res = re.sub(r'\{\s*(\w+)\s*\}', r'{\1}', res)
            return key, fixed_res, True
        return key, value, False

    # Run translations in parallel (max 25 threads for blazing fast completion)
    with _ThreadPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(translate_single_key, source.items()))

    for key, val, success in results:
        translated[key] = val
        if not success:
            failed_keys.append(key)

    if failed_keys:
        logger.warning("Auto-translate '%s': %d keys fell back to source: %s",
                       ui_lang, len(failed_keys), failed_keys[:5])

    # Save file
    out_path = os.path.join(app.root_path, 'translations', f'{ui_lang}.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)
        TRANSLATIONS[ui_lang] = translated
        with _translation_jobs_lock:
            _translation_jobs[ui_lang] = 'done'
        logger.info("Auto-translation for '%s' completed (%d keys, %d fallbacks).",
                    ui_lang, len(translated), len(failed_keys))
    except Exception as exc:
        with _translation_jobs_lock:
            _translation_jobs[ui_lang] = 'failed'
        logger.error("Failed to save translation for '%s': %s", ui_lang, exc)



def trigger_auto_translation(sign_lang_code: str):
    """
    Given a sign-language code (e.g. 'fr', 'tr', 'sa'), resolve the UI language
    and kick off a background translation job if no translation file exists yet.
    """
    ui_lang = SIGN_LANG_TO_UI_LANG.get(sign_lang_code, sign_lang_code)
    out_path = os.path.join(app.root_path, 'translations', f'{ui_lang}.json')

    if os.path.exists(out_path):
        with _translation_jobs_lock:
            _translation_jobs[ui_lang] = 'exists'
        logger.info("Translation for '%s' already exists — skipping generation.", ui_lang)
        return

    with _translation_jobs_lock:
        if _translation_jobs.get(ui_lang) == 'pending':
            return  # already running

    logger.info("Triggering auto-translation for sign lang '%s' → UI lang '%s'", sign_lang_code, ui_lang)
    t = threading.Thread(target=_do_generate_translation, args=(ui_lang,), daemon=True)
    t.start()


@app.route('/admin/translation-status', methods=['GET'])
@admin_required
def admin_translation_status():
    """
    Returns translation status for every registered sign language.
    Frontend polls this to show progress indicators.
    """
    config_data = load_languages_config()
    result = {}
    for code in config_data.get('languages', {}):
        ui_lang = SIGN_LANG_TO_UI_LANG.get(code, code)
        path = os.path.join(app.root_path, 'translations', f'{ui_lang}.json')
        with _translation_jobs_lock:
            job = _translation_jobs.get(ui_lang)
        if os.path.exists(path):
            status = 'exists'
        elif job == 'pending':
            status = 'pending'
        elif job == 'failed':
            status = 'failed'
        else:
            status = 'missing'
        result[code] = {'ui_lang': ui_lang, 'status': status}
    return jsonify({'translations': result})


@app.route('/admin/generate-translation', methods=['POST'])
@admin_required
def admin_generate_translation():
    """Manually trigger auto-translation for a sign language code."""
    data = request.get_json(silent=True) or {}
    sign_lang_code = data.get('lang_code', '').strip().lower()
    force = bool(data.get('force', False))

    if not sign_lang_code:
        return jsonify({'success': False, 'error': 'رمز اللغة مطلوب'}), 400

    ui_lang = SIGN_LANG_TO_UI_LANG.get(sign_lang_code, sign_lang_code)
    out_path = os.path.join(app.root_path, 'translations', f'{ui_lang}.json')

    if os.path.exists(out_path) and not force:
        # Reload into memory and return
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                TRANSLATIONS[ui_lang] = json.load(f)
        except Exception:
            pass
        return jsonify({'success': True, 'status': 'exists',
                        'message': f'ملف الترجمة ({ui_lang}.json) موجود بالفعل'})

    if force and os.path.exists(out_path):
        os.remove(out_path)

    with _translation_jobs_lock:
        if _translation_jobs.get(ui_lang) == 'pending':
            return jsonify({'success': True, 'status': 'pending',
                            'message': 'الترجمة جارية بالفعل...'})

    t = threading.Thread(target=_do_generate_translation, args=(ui_lang,), daemon=True)
    t.start()
    logger.info("Admin triggered translation for sign_lang=%s → ui_lang=%s", sign_lang_code, ui_lang)
    return jsonify({'success': True, 'status': 'pending',
                    'message': f'بدأت الترجمة التلقائية إلى {ui_lang}...'})


# =============================================
# Admin
# =============================================
@app.route('/admin')
def admin_page():
    if current_user_is_admin():
        session['is_admin'] = True
    return render_template('admin.html', letters=get_active_labels_list(), active_lang=session.get('active_lang', languages_config.get('active_language', 'ar')))

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
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
    return jsonify({'users': get_all_users_stats(lang_code)})

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
        lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
        users = get_all_users_stats(lang_code)
        total_samples = sum(u['total'] for u in users)
        logger.info("Admin export statistics retrieved successfully: %d users, %d samples for lang %s", len(users), total_samples, lang_code)
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
    active_lang = session.get('active_lang', languages_config.get('active_language', 'ar'))
    return render_template('collect_test.html', letters=get_active_labels_list(), active_lang=active_lang)

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

    active_labels = get_active_labels()
    if label not in active_labels:
        return jsonify({'success': False, 'error': 'تسمية غير صالحة'}), 400
    if len(raw) != 42:
        return jsonify({'success': False, 'error': 'Invalid landmarks'}), 400

    try:
        # Validate elements are floats
        float_raw = [float(v) for v in raw]
        if any(np.isnan(float_raw)) or any(np.isinf(float_raw)):
            return jsonify({'success': False, 'error': 'Invalid landmark coordinate values'}), 400

        normalized = normalize_landmarks(float_raw)
        active_test_csv = get_active_test_dataset_path()
        os.makedirs(os.path.dirname(active_test_csv), exist_ok=True)
        with _test_csv_lock:
            with open(active_test_csv, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([tester_id, label] + normalized)
            
            # Count samples for this tester+label
            total = 0
            if os.path.exists(active_test_csv):
                with open(active_test_csv, 'r', encoding='utf-8') as f:
                    for row in csv.reader(f):
                        if len(row) >= 2 and row[0] == tester_id and row[1] == str(label):
                            total += 1
        
        logger.info("Test sample recorded successfully: tester=%s, label=%d, file=%s", tester_id, label, active_test_csv)
        return jsonify({'success': True, 'total': total})
    except Exception as exc:
        logger.error("Error in /collect-test POST: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'حدث خطأ'}), 500

@app.route('/collect-test/export')
@app.route('/collect_test/export')
def collect_test_export():
    active_test_csv = get_active_test_dataset_path()
    if not os.path.exists(active_test_csv):
        logger.warning("Export test CSV requested but file does not exist: %s", active_test_csv)
        return jsonify({'error': 'لا توجد بيانات بعد'}), 404
    try:
        with open(active_test_csv, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in csv.reader(f))
        if row_count < 50:
            logger.warning("Export test CSV blocked: insufficient samples count %d < 50 for %s", row_count, active_test_csv)
            return jsonify({'error': f'عدد العينات غير كافٍ ({row_count}/50)'}), 403
    except Exception as exc:
        logger.error("Failed to read test CSV for export: %s", exc)
        return jsonify({'error': 'خطأ في قراءة الملف'}), 500
        
    logger.info("Exporting test CSV %s containing %d samples.", active_test_csv, row_count)
    return send_from_directory(os.path.dirname(active_test_csv), os.path.basename(active_test_csv),
                               as_attachment=True,
                               download_name=os.path.basename(active_test_csv))

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

# =============================================
# Language Switcher & Management Routes
# =============================================
@app.route('/auth/languages', methods=['GET'])
def get_languages_list():
    config_data = load_languages_config()
    active = session.get('active_lang', config_data.get('active_language', 'ar'))
    langs = []
    for code, info in config_data['languages'].items():
        langs.append({
            'code': code,
            'name': info['name'],
            'active': (code == active)
        })
    return jsonify({'languages': langs, 'active': active})

@app.route('/auth/set-active-lang', methods=['POST'])
def set_active_language():
    data = request.get_json(silent=True) or {}
    lang_code = data.get('lang_code', '')
    config_data = load_languages_config()
    if lang_code not in config_data['languages']:
        return jsonify({'success': False, 'error': 'اللغة غير مدعومة'}), 400
    session['active_lang'] = lang_code
    logger.info("Active language updated to %s for session", lang_code)
    return jsonify({'success': True})

@app.route('/admin/languages-detail', methods=['GET'])
@admin_required
def admin_languages_detail():
    config_data = load_languages_config()
    active = session.get('active_lang', config_data.get('active_language', 'ar'))
    langs_detail = []
    
    def format_size(size_in_bytes):
        if not size_in_bytes:
            return '0 B'
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B"
        elif size_in_bytes < 1024 * 1024:
            return f"{size_in_bytes / 1024:.2f} KB"
        else:
            return f"{size_in_bytes / (1024 * 1024):.2f} MB"
            
    encoding_errors = validate_labels_encoding()
    
    for code, info in config_data['languages'].items():
        # Get labels and class count
        labels = get_lang_labels(info)
        class_count = len(labels)
        
        # Get sample count from dataset_path CSV file
        dataset_path = info.get('dataset_path')
        sample_count = 0
        dataset_exists = bool(dataset_path and os.path.exists(dataset_path))
        if dataset_exists:
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    sample_count = sum(1 for line in f if line.strip())
            except Exception as e:
                logger.error("Failed to count samples in %s: %s", dataset_path, e)
                
        # Get training report info
        last_training_date = '-'
        latest_accuracy = '-'
        report_path = os.path.join(config.REPORTS_DIR, f"{code}_training_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    rep_data = json.load(f)
                    last_training_date = rep_data.get('trained_at', '-')
                    acc = rep_data.get('accuracy')
                    if acc is not None:
                        latest_accuracy = f"{float(acc)*100:.2f}%"
            except Exception as e:
                logger.error("Failed to read report for %s: %s", code, e)
        
        # Determine status
        status = get_language_status(code)
        
        # Health details
        health_info = language_health.get(code, {
            "model_classes": 0,
            "label_count": class_count,
            "valid": False,
            "reason": "Not initialized"
        })
        
        failure_reason = health_info.get("reason", "")
        model_classes = int(health_info.get("model_classes", 0))
        
        # Paths
        labels_path = info.get('labels_path', '')
        labels_exist = bool(labels_path and os.path.exists(labels_path))
        
        model_path = info.get('model_path', '')
        model_exists = bool(model_path and os.path.exists(model_path))
        
        # Health categories
        if not labels_exist:
            health_status = "Missing Labels"
        elif not dataset_exists:
            health_status = "Missing Dataset"
        elif not model_path:
            health_status = "Not Trained"
        elif not model_exists:
            health_status = "Missing Model"
        else:
            if health_info.get("valid"):
                health_status = "Healthy"
            else:
                health_status = "Invalid Model"
        
        # Warnings checks
        warnings = []
        if labels_exist and not dataset_exists:
            warnings.append("Dataset Missing")
        elif dataset_exists and sample_count == 0:
            warnings.append("Empty Dataset")
            
        if status == 'invalid_model':
            warnings.append("Invalid Model")
            
        enc_err = encoding_errors.get(code, '')
        if enc_err:
            warnings.append("Encoding Error")

        model_size = os.path.getsize(model_path) if model_exists else 0
        dataset_size = os.path.getsize(dataset_path) if dataset_exists else 0
        
        langs_detail.append({
            'code': code,
            'name': info.get('name', ''),
            'model_path': model_path,
            'labels_path': labels_path,
            'dataset_path': dataset_path,
            'sample_count': sample_count,
            'class_count': class_count,
            'model_classes': model_classes,
            'health_status': health_status,
            'failure_reason': failure_reason,
            'last_training_date': last_training_date,
            'latest_accuracy': latest_accuracy,
            'status': status,
            'invalid_reason': failure_reason or _invalid_lang_reasons.get(code, ''),
            'model_exists': model_exists,
            'labels_exists': labels_exist,
            'dataset_exists': dataset_exists,
            'encoding_error': enc_err,
            'warnings': warnings,
            'is_current_active': (code == active),
            'model_size': model_size,
            'dataset_size': dataset_size,
            'model_size_str': format_size(model_size) if model_exists else '-',
            'dataset_size_str': format_size(dataset_size) if dataset_exists else '-'
        })
        
    deleted_langs_detail = []
    deleted_langs = config_data.get('deleted_languages', {})
    import datetime
    now = datetime.datetime.now()
    for dcode, dinfo in deleted_langs.items():
        deleted_at_str = dinfo.get('deleted_at', '')
        days_left = 30
        if deleted_at_str:
            try:
                del_at = datetime.datetime.strptime(deleted_at_str, "%Y-%m-%d %H:%M:%S")
                elapsed_days = (now - del_at).days
                days_left = max(0, 30 - elapsed_days)
            except Exception as e:
                logger.error("Error parsing deleted_at: %s", e)
        deleted_langs_detail.append({
            'code': dcode,
            'name': dinfo.get('name', ''),
            'deleted_at': deleted_at_str,
            'days_left': days_left,
            'class_count': len(get_lang_labels(dinfo))
        })
        
    return jsonify({
        'languages': langs_detail,
        'deleted_languages': deleted_langs_detail,
        'active': active
    })


@app.route('/admin/delete-language', methods=['POST'])
@admin_required
def delete_language():
    data = request.get_json(silent=True) or {}
    code = data.get('code', '').strip().lower()
    
    if not code:
        return jsonify({'success': False, 'error': 'رمز اللغة مطلوب'}), 400
        
    config_data = load_languages_config()
    active = config_data.get('active_language', 'ar')
    
    if code == active:
        return jsonify({'success': False, 'error': 'لا يمكن حذف اللغة النشطة حالياً. يرجى تغيير اللغة النشطة أولاً.'}), 400
        
    if code not in config_data.get('languages', {}):
        return jsonify({'success': False, 'error': 'اللغة غير موجودة'}), 404
        
    with _languages_config_lock:
        config_data = load_languages_config()
        if code not in config_data.get('languages', {}):
            return jsonify({'success': False, 'error': 'اللغة غير موجودة'}), 404
            
        lang_info = config_data['languages'].pop(code)
        
        import datetime
        lang_info['deleted_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if 'deleted_languages' not in config_data:
            config_data['deleted_languages'] = {}
        config_data['deleted_languages'][code] = lang_info
        
        tmp_path = LANGUAGES_CONFIG_PATH + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, LANGUAGES_CONFIG_PATH)
        
    init_interpreters()
    logger.info("Language %s moved to recycle bin", code)
    return jsonify({'success': True})


@app.route('/admin/restore-language', methods=['POST'])
@admin_required
def restore_language():
    data = request.get_json(silent=True) or {}
    code = data.get('code', '').strip().lower()
    
    if not code:
        return jsonify({'success': False, 'error': 'رمز اللغة مطلوب'}), 400
        
    config_data = load_languages_config()
    deleted_langs = config_data.get('deleted_languages', {})
    
    if code not in deleted_langs:
        return jsonify({'success': False, 'error': 'اللغة غير موجودة في سلة المحذوفات'}), 404
        
    if code in config_data.get('languages', {}):
        return jsonify({'success': False, 'error': 'توجد لغة نشطة بنفس الرمز بالفعل'}), 400
        
    with _languages_config_lock:
        config_data = load_languages_config()
        deleted_langs = config_data.get('deleted_languages', {})
        if code not in deleted_langs:
            return jsonify({'success': False, 'error': 'اللغة غير موجودة في سلة المحذوفات'}), 404
            
        lang_info = deleted_langs.pop(code)
        lang_info.pop('deleted_at', None)
        
        config_data['languages'][code] = lang_info
        
        tmp_path = LANGUAGES_CONFIG_PATH + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, LANGUAGES_CONFIG_PATH)
        
    init_interpreters()
    logger.info("Language %s restored from recycle bin", code)
    return jsonify({'success': True})


@app.route('/admin/delete-language-permanently', methods=['POST'])
@admin_required
def delete_language_permanently():
    data = request.get_json(silent=True) or {}
    code = data.get('code', '').strip().lower()
    
    if not code:
        return jsonify({'success': False, 'error': 'رمز اللغة مطلوب'}), 400
        
    config_data = load_languages_config()
    deleted_langs = config_data.get('deleted_languages', {})
    
    if code not in deleted_langs:
        return jsonify({'success': False, 'error': 'اللغة غير موجودة في سلة المحذوفات'}), 404
        
    with _languages_config_lock:
        config_data = load_languages_config()
        deleted_langs = config_data.get('deleted_languages', {})
        if code not in deleted_langs:
            return jsonify({'success': False, 'error': 'اللغة غير موجودة في سلة المحذوفات'}), 404
            
        lang_info = deleted_langs.pop(code)
        
        for path_key in ['model_path', 'labels_path', 'dataset_path']:
            path = lang_info.get(path_key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info("Permanently deleted file: %s", path)
                except Exception as e:
                    logger.error("Failed to delete file %s: %s", path, e)
                    
        tmp_path = LANGUAGES_CONFIG_PATH + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, LANGUAGES_CONFIG_PATH)
        
    logger.info("Language %s permanently deleted", code)
    return jsonify({'success': True})


@app.route('/admin/add-language', methods=['POST'])
@admin_required
def add_new_language():
    import re
    import shutil
    import csv
    data = request.get_json(silent=True) or {}
    code = data.get('code', '').strip().lower()
    name = data.get('name', '').strip()
    labels = data.get('labels', [])
    parent = data.get('parent', '').strip().lower()

    if not code or not name or not labels:
        return jsonify({'success': False, 'error': 'جميع الحقول مطلوبة'}), 400

    if not isinstance(labels, list) or len(labels) == 0:
        return jsonify({'success': False, 'error': 'قائمة التسميات يجب أن تحتوي على تسمية واحدة على الأعل'}), 400

    # Ensure all labels are valid non-empty strings
    labels = [str(lbl).strip() for lbl in labels if str(lbl).strip()]
    if not labels:
        return jsonify({'success': False, 'error': 'يجب توفير تسميات صالحة'}), 400

    if not re.match(r'^[a-z]{2,3}$', code):
        return jsonify({'success': False, 'error': 'رمز اللغة يجب أن يكون من حرفين أو ثلاثة أحرف إنجليزية'}), 400

    config_data = load_languages_config()
    if code in config_data['languages']:
        return jsonify({'success': False, 'error': 'اللغة مسجلة بالفعل'}), 400

    if parent and parent not in config_data['languages']:
        return jsonify({'success': False, 'error': 'رمز اللغة الأب غير مسجل في النظام'}), 400

    # Paths
    dataset_path = f"languages_data/{code}/{code}_keypoints.csv"
    model_path = f"arabic_model/{code}_sign_model.tflite"
    labels_path = f"languages_data/{code}/{code}_labels.csv"

    # Create empty dataset file
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, 'w', newline='', encoding='utf-8') as f:
            pass

    # Create labels file
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    try:
        with open(labels_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for idx, lbl in enumerate(labels):
                writer.writerow([idx, lbl])
    except Exception as e:
        logger.error("Failed to save labels file for new language %s: %s", code, e)
        return jsonify({'success': False, 'error': 'فشل حفظ ملف التسميات'}), 500

    # Language is created WITHOUT a model — it will show as Not Trained until
    # an admin trains it. Never copy the Arabic placeholder model.

    # Save to config (protected by languages config lock)
    with _languages_config_lock:
        config_data['languages'][code] = {
            'code': code,
            'name': name,
            'labels': labels,
            'model_path': model_path,
            'labels_path': labels_path,
            'dataset_path': dataset_path
        }
        if parent:
            config_data['languages'][code]['parent'] = parent
        tmp_path = LANGUAGES_CONFIG_PATH + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, LANGUAGES_CONFIG_PATH)

    # Reload interpreters (new language will be unavailable until trained)
    init_interpreters()

    # Auto-generate UI translation for the new language in the background
    trigger_auto_translation(code)

    logger.info("Successfully added new language: %s (%s) — status: Not Trained", name, code)
    return jsonify({'success': True})

@app.route('/admin/train-active-model', methods=['POST'])
@admin_required
def train_active_model_route():
    import subprocess
    import sys
    lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))

    # ── Training lock: prevent simultaneous training for the same language ──
    with _training_state_lock:
        if _training_in_progress.get(lang_code):
            logger.warning("Training already in progress for %s — duplicate request rejected", lang_code)
            return jsonify({
                'success': False,
                'error': f'التدريب جاري بالفعل للغة {lang_code} — انتظر حتى ينتهي'
            }), 409
        _training_in_progress[lang_code] = True

    logger.info("Starting model training process for language: %s", lang_code)
    try:
        res = subprocess.run([sys.executable, 'train_model.py', lang_code], capture_output=True, text=True, timeout=180)
        if res.returncode == 0:
            logger.info("Model training completed successfully for %s", lang_code)
            init_interpreters()
            return jsonify({'success': True})
        else:
            logger.error("Model training failed for %s. Error: %s", lang_code, res.stderr)
            return jsonify({'success': False, 'error': f'فشل التدريب: {res.stderr[:200]}'})
    except subprocess.TimeoutExpired:
        logger.error("Model training timed out for %s", lang_code)
        return jsonify({'success': False, 'error': 'انتهت مهلة التدريب (أكثر من 3 دقائق)'})
    except Exception as e:
        logger.error("Failed to trigger training for %s: %s", lang_code, e)
        return jsonify({'success': False, 'error': f'حدث خطأ غير متوقع: {str(e)}'})
    finally:
        # Always release the lock, even on timeout or error
        with _training_state_lock:
            _training_in_progress[lang_code] = False

@app.route('/admin/restart-server', methods=['POST'])
@admin_required
def restart_server_route():
    import sys
    import subprocess
    logger.info("Restarting server programmatically as requested by Admin...")

    # ── Restart guard: prevent duplicate process spawning on double-click ──
    with _restart_lock:
        if _restart_in_progress:
            logger.warning("Restart already in progress — duplicate request ignored")
            return jsonify({'success': False, 'error': 'إعادة التشغيل جارية بالفعل'}), 409
        # Mark in-progress inside the lock before releasing
        globals()['_restart_in_progress'] = True

    def restart():
        import time
        time.sleep(1)
        subprocess.Popen([sys.executable] + sys.argv)
        os._exit(0)

    threading.Thread(target=restart, daemon=True).start()
    return jsonify({'success': True, 'message': 'جاري إعادة تشغيل الخادم...'})

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
