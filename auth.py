#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hmac
import json
import os
import re
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

USERS_DIR = 'arabic_data/users'
USERS_INDEX = 'arabic_data/users/users.json'
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
VALID_ROLES = {'user', 'admin'}

os.makedirs(USERS_DIR, exist_ok=True)

def hash_password(password: str) -> str:
    return generate_password_hash(password)

def load_users():
    if not os.path.exists(USERS_INDEX):
        return {}
    with open(USERS_INDEX, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_INDEX, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def load_user(user_id):
    path = os.path.join(USERS_DIR, f'{user_id}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_user(user_data):
    path = os.path.join(USERS_DIR, f"{user_data['id']}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

def register_user(name, password, ip=None, user_agent=None):
    users = load_users()
    user_id = name.strip().lower()
    user_id = re.sub(r'\s+', '_', user_id)
    # Remove any character that is not Arabic letter, Latin letter, digit, or underscore
    user_id = re.sub(r'[^\u0600-\u06ff\w]', '', user_id)
    if not user_id:
        return None, 'اسم غير صالح'
    if user_id in users:
        return None, 'الاسم موجود بالفعل'

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    user = {
        'id': user_id,
        'name': name.strip(),
        'password': hash_password(password),
        'created': now,
        'role': 'user',
        'status': 'active',
        'samples': {},
        'rejected': 0,
        'total_accepted': 0,
        'last_active': now,
        # معلومات الجهاز
        'device_info': {
            'ip': ip or 'unknown',
            'user_agent': user_agent or 'unknown',
            'device_type': parse_device_type(user_agent),
            'os': parse_os(user_agent),
            'browser': parse_browser(user_agent),
        },
        # سجل تسجيلات الدخول
        'login_history': [{
            'time': now,
            'ip': ip or 'unknown',
            'device': parse_device_type(user_agent)
        }]
    }
    users[user_id] = {'name': name.strip(), 'created': now, 'role': 'user', 'status': 'active'}
    save_users(users)
    save_user(user)
    return user_id, None

def login_user(name, password, ip=None, user_agent=None):
    user_id = name.strip().lower()
    user_id = re.sub(r'\s+', '_', user_id)
    user_id = re.sub(r'[^\u0600-\u06ff\w]', '', user_id)
    if not user_id:
        return None, 'اسم غير صالح'
    user = load_user(user_id)
    if not user:
        return None, 'المستخدم غير موجود'
    if user.get('status') == 'disabled':
        return None, 'هذا الحساب معطل'
    if not check_password_hash(user.get('password', ''), password):
        return None, 'كلمة المرور غلط'

    # تحديث معلومات الجهاز وسجل الدخول
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    user['device_info'] = {
        'ip': ip or user.get('device_info', {}).get('ip', 'unknown'),
        'user_agent': user_agent or 'unknown',
        'device_type': parse_device_type(user_agent),
        'os': parse_os(user_agent),
        'browser': parse_browser(user_agent),
    }
    # سجل آخر 10 عمليات دخول
    history = user.get('login_history', [])
    history.append({'time': now, 'ip': ip or 'unknown', 'device': parse_device_type(user_agent)})
    user['login_history'] = history[-10:]
    user['last_active'] = now
    save_user(user)
    return user_id, None

# ===== تحليل User-Agent =====
def parse_device_type(ua):
    if not ua:
        return 'unknown'
    ua = ua.lower()
    if any(x in ua for x in ['iphone', 'android', 'mobile']):
        return 'Mobile'
    if any(x in ua for x in ['ipad', 'tablet']):
        return 'Tablet'
    return 'Desktop'

def parse_os(ua):
    if not ua:
        return 'unknown'
    ua = ua.lower()
    if 'android' in ua:
        # استخراج إصدار Android
        try:
            idx = ua.index('android')
            ver = ua[idx+8:idx+11].strip('; ')
            return f'Android {ver}'
        except:
            return 'Android'
    if 'iphone' in ua or 'ipad' in ua:
        try:
            idx = ua.index('os ')
            ver = ua[idx+3:idx+8].replace('_','.').strip()
            return f'iOS {ver}'
        except:
            return 'iOS'
    if 'windows' in ua:
        if 'windows nt 10' in ua: return 'Windows 10/11'
        if 'windows nt 6.3' in ua: return 'Windows 8.1'
        return 'Windows'
    if 'mac os' in ua: return 'macOS'
    if 'linux' in ua: return 'Linux'
    return 'unknown'

def parse_browser(ua):
    if not ua:
        return 'unknown'
    ua = ua.lower()
    if 'chrome' in ua and 'chromium' not in ua and 'edg' not in ua:
        return 'Chrome'
    if 'firefox' in ua:
        return 'Firefox'
    if 'safari' in ua and 'chrome' not in ua:
        return 'Safari'
    if 'edg' in ua:
        return 'Edge'
    if 'samsung' in ua:
        return 'Samsung Browser'
    return 'Other'

# ===== بقية الدوال =====
def record_sample(user_id, label):
    user = load_user(user_id)
    if not user:
        return False
    label_str = str(label)
    user['samples'][label_str] = user['samples'].get(label_str, 0) + 1
    user['total_accepted'] = user.get('total_accepted', 0) + 1
    user['last_active'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_user(user)
    return True

def record_rejected(user_id):
    user = load_user(user_id)
    if not user:
        return
    user['rejected'] = user.get('rejected', 0) + 1
    save_user(user)

def get_all_users_stats():
    users = load_users()
    stats = []
    for user_id in users:
        user = load_user(user_id)
        if user:
            total = user.get('total_accepted', 0)
            rejected = user.get('rejected', 0)
            quality = round(total / (total + rejected) * 100) if (total + rejected) > 0 else 0
            device = user.get('device_info', {})
            stats.append({
                'id': user_id,
                'name': user['name'],
                'role': get_user_role(user),
                'status': user.get('status', 'active'),
                'total': total,
                'rejected': rejected,
                'quality': quality,
                'letters_count': len(user.get('samples', {})),
                'last_active': user.get('last_active', '-'),
                'created': user.get('created', '-'),
                'samples': user.get('samples', {}),
                'device': {
                    'ip': device.get('ip', 'unknown'),
                    'device_type': device.get('device_type', 'unknown'),
                    'os': device.get('os', 'unknown'),
                    'browser': device.get('browser', 'unknown'),
                },
                'login_history': user.get('login_history', []),
                'edit_history': __import__('history').get_entries(user_id)
            })
    return sorted(stats, key=lambda x: x['total'], reverse=True)

def delete_user(user_id):
    users = load_users()
    if user_id in users:
        del users[user_id]
        save_users(users)
    path = os.path.join(USERS_DIR, f'{user_id}.json')
    if os.path.exists(path):
        os.remove(path)

def disable_user(user_id):
    users = load_users()
    if user_id in users:
        users[user_id]['status'] = 'disabled'
        save_users(users)
    user = load_user(user_id)
    if user:
        user['status'] = 'disabled'
        save_user(user)

def enable_user(user_id):
    users = load_users()
    if user_id in users:
        users[user_id]['status'] = 'active'
        save_users(users)
    user = load_user(user_id)
    if user:
        user['status'] = 'active'
        save_user(user)

def get_user_role(user):
    role = user.get('role', 'user')
    return role if role in VALID_ROLES else 'user'

def user_is_admin(user):
    return bool(user) and get_user_role(user) == 'admin'

def set_user_role(user_id, role):
    if role not in VALID_ROLES:
        return False, 'نوع الحساب غير صالح'

    user = load_user(user_id)
    if not user:
        return False, 'المستخدم غير موجود'

    user['role'] = role
    save_user(user)

    users = load_users()
    if user_id in users:
        users[user_id]['role'] = role
        save_users(users)

    return True, None

def verify_admin(password):
    if not ADMIN_PASSWORD:
        return False
    # Use constant-time comparison to prevent timing attacks
    # (compare_digest raises ValueError if lengths differ)
    try:
        return hmac.compare_digest(password, ADMIN_PASSWORD)
    except (TypeError, ValueError):
        return False
