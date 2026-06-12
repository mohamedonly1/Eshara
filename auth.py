#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Authentication and user management module for Ishara.
Handles registration, login, statistics tracking, role settings,
email verification, and profile updates.
"""
import hmac
import json
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.header import Header
from typing import Dict, Tuple, Optional, List, Any
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

import config

load_dotenv()

USERS_DIR = config.USERS_DIR
USERS_INDEX = os.path.join(config.USERS_DIR, 'users.json')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
VALID_ROLES = {'user', 'admin'}

os.makedirs(USERS_DIR, exist_ok=True)

# Central Logger setup for emails
email_logger = config.get_file_logger('email', config.SERVER_LOG)

def generate_secure_otp(length: int = 6) -> str:
    """Generates a cryptographically secure numeric OTP using the secrets module."""
    return ''.join(secrets.choice('0123456789') for _ in range(length))

def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2 with SHA-256."""
    return generate_password_hash(password)

def load_users() -> Dict[str, Any]:
    """Loads the main user index database."""
    if not os.path.exists(USERS_INDEX):
        return {}
    with open(USERS_INDEX, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users(users: Dict[str, Any]) -> None:
    """Saves the user index database."""
    with open(USERS_INDEX, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def load_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Loads a specific user's detailed profile JSON file."""
    path = os.path.join(USERS_DIR, f'{user_id}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_user(user_data: Dict[str, Any]) -> None:
    """Saves a user's detailed profile JSON file."""
    path = os.path.join(USERS_DIR, f"{user_data['id']}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

def print_mock_email(to_email: str, subject: str, body: str) -> None:
    """Prints a clear mock representation of the sent email in the server logs."""
    box = (
        f"\n{'='*60}\n"
        f"  [MOCK EMAIL SENT]\n"
        f"  To:      {to_email}\n"
        f"  Subject: {subject}\n"
        f"  Body:    {body.replace('<br>', ' ').replace('<p>', '').replace('</p>', '')}\n"
        f"{'='*60}\n"
    )
    try:
        print(box)
    except UnicodeEncodeError:
        try:
            print(box.encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass
    email_logger.info(box)

def send_verification_email(to_email: str, name: str, code: str) -> bool:
    """Sends a verification email using SMTP configurations or falls back to server console printing."""
    subject = "رمز التحقق لحسابك في تطبيق إشارة"
    body = f"""
    <div style="direction: rtl; font-family: 'Cairo', sans-serif; text-align: right; padding: 20px; border: 1px solid #e1e8e5; border-radius: 12px; max-width: 500px; margin: auto;">
        <h2 style="color: #1D9E75;">مرحباً {name}،</h2>
        <p style="font-size: 16px; color: #4a5568;">شكرًا لتسجيلك في تطبيق <b>إشارة</b> لترجمة وجمع بيانات لغة الإشارة العربية.</p>
        <p style="font-size: 16px; color: #4a5568;">رمز التحقق الخاص بحسابك هو:</p>
        <div style="background-color: #f1f5f3; padding: 15px; border-radius: 8px; text-align: center; font-size: 28px; font-weight: bold; letter-spacing: 4px; color: #1D9E75; margin: 20px 0;">
            {code}
        </div>
        <p style="font-size: 14px; color: #718096;">إذا لم تقم بطلب هذا الرمز، يرجى تجاهل هذا البريد.</p>
        <hr style="border: 0; border-top: 1px solid #e1e8e5; margin: 20px 0;">
        <p style="font-size: 12px; color: #a0aec0; text-align: center;">تطبيق إشارة — 2026</p>
    </div>
    """
    
    mail_server = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    try:
        mail_port = int(os.getenv('MAIL_PORT', '587'))
    except (ValueError, TypeError):
        mail_port = 587
    mail_username = os.getenv('MAIL_USERNAME')
    mail_password = os.getenv('MAIL_PASSWORD')
    mail_use_tls = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 'yes')
    sender = os.getenv('MAIL_DEFAULT_SENDER', 'Ishara <no-reply@ishara.com>')

    # Form message
    msg = MIMEText(body, 'html', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = Header(sender, 'utf-8')
    msg['To'] = to_email

    if not mail_username or not mail_password:
        email_logger.error("SMTP credentials (MAIL_USERNAME / MAIL_PASSWORD) not set in .env. Cannot send verification email.")
        return False

    try:
        if mail_port == 465:
            server = smtplib.SMTP_SSL(mail_server, mail_port, timeout=10)
        else:
            server = smtplib.SMTP(mail_server, mail_port, timeout=10)
            if mail_use_tls:
                server.starttls()
        
        server.login(mail_username, mail_password)
        server.sendmail(mail_username, [to_email], msg.as_string())
        server.quit()
        email_logger.info("Verification email successfully sent via SMTP to %s", to_email)
        return True
    except Exception as e:
        email_logger.error("Failed to send verification email to %s: %s", to_email, e)
        return False

def register_user(name: str, email: str, password: str, ip: str = None, user_agent: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Registers a new user into the system with an email and sends verification code.
    Returns:
        (user_id, error_message)
    """
    users = load_users()
    user_id = name.strip().lower()
    user_id = re.sub(r'\s+', '_', user_id)
    # Remove any character that is not Arabic letter, Latin letter, digit, or underscore
    user_id = re.sub(r'[^\u0600-\u06ff\w]', '', user_id)
    if not user_id:
        return None, 'اسم غير صالح'
    if user_id in users:
        return None, 'الاسم موجود بالفعل'

    email = email.strip().lower()
    if not email or not re.match(r'[^@]+@[^@]+\.[^@]+', email):
        return None, 'البريد الإلكتروني غير صالح'

    # Check if email is already in use
    for uid, uinfo in users.items():
        if uinfo.get('email', '').lower() == email:
            return None, 'البريد الإلكتروني مستخدم بالفعل'

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    verification_code = generate_secure_otp()
    verification_expires = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')

    user = {
        'id': user_id,
        'name': name.strip(),
        'email': email,
        'email_verified': False,
        'verification_code': verification_code,
        'verification_code_expires': verification_expires,
        'password': hash_password(password),
        'created': now,
        'role': 'user',
        'status': 'active',
        'samples': {},
        'rejected': 0,
        'total_accepted': 0,
        'last_active': now,
        'profile_pic': '',
        # Device information
        'device_info': {
            'ip': ip or 'unknown',
            'user_agent': user_agent or 'unknown',
            'device_type': parse_device_type(user_agent),
            'os': parse_os(user_agent),
            'browser': parse_browser(user_agent),
        },
        # Login history
        'login_history': [{
            'time': now,
            'ip': ip or 'unknown',
            'device': parse_device_type(user_agent)
        }]
    }

    users[user_id] = {
        'name': name.strip(),
        'email': email,
        'created': now,
        'role': 'user',
        'status': 'active'
    }

    save_users(users)
    save_user(user)

    # Send verification email asynchronously / synchronously
    send_verification_email(email, name.strip(), verification_code)

    return user_id, None

def login_user(name: str, password: str, ip: str = None, user_agent: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Authenticates a user logging into the system using either username or email.
    Returns:
        (user_id, error_message)
    """
    input_str = name.strip().lower()
    user_id = None
    
    # Check if login is email or username
    if '@' in input_str:
        users = load_users()
        for uid, uinfo in users.items():
            if uinfo.get('email', '').lower() == input_str:
                user_id = uid
                break
        if not user_id:
            return None, 'البريد الإلكتروني غير مسجل'
    else:
        user_id = re.sub(r'\s+', '_', input_str)
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

    # Update device and login statistics
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    user['device_info'] = {
        'ip': ip or user.get('device_info', {}).get('ip', 'unknown'),
        'user_agent': user_agent or 'unknown',
        'device_type': parse_device_type(user_agent),
        'os': parse_os(user_agent),
        'browser': parse_browser(user_agent),
    }
    # Keep last 10 logins only
    history = user.get('login_history', [])
    history.append({'time': now, 'ip': ip or 'unknown', 'device': parse_device_type(user_agent)})
    user['login_history'] = history[-10:]
    user['last_active'] = now
    save_user(user)
    return user_id, None

# ===== Profile Modification =====
def update_profile(
    user_id: str,
    new_name: str,
    current_password: str,
    new_password: Optional[str] = None,
    new_email: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Updates user profile display name, login username, password, and email.
    Requires current password verification for security.
    Returns:
        (new_user_id, error_message)
    """
    user = load_user(user_id)
    if not user:
        return None, 'المستخدم غير موجود'
    
    # Verify current password
    if not check_password_hash(user.get('password', ''), current_password):
        return None, 'كلمة المرور الحالية غير صحيحة'

    users = load_users()
    
    # Check email changes — use pending_email pattern (SEC-05)
    email_change_requested = False
    if new_email:
        new_email = new_email.strip().lower()
        if new_email != user.get('email', '').lower():
            # Validate email
            if not re.match(r'[^@]+@[^@]+\.[^@]+', new_email):
                return None, 'البريد الإلكتروني غير صالح'
            # Check if email is already in use
            for uid, uinfo in users.items():
                if uid != user_id and uinfo.get('email', '').lower() == new_email:
                    return None, 'البريد الإلكتروني مستخدم بالفعل'
            # Store as pending — do NOT change the active email yet
            user['pending_email'] = new_email
            user['verification_code'] = generate_secure_otp()
            user['verification_code_expires'] = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')
            email_change_requested = True

    # Check password changes
    if new_password:
        new_password = new_password.strip()
        if len(new_password) < 4:
            return None, 'كلمة المرور الجديدة قصيرة جداً (أقل من 4 أحرف)'
        user['password'] = hash_password(new_password)

    # Check name changes
    new_user_id = user_id
    new_name = new_name.strip()
    if new_name and new_name != user['name']:
        test_new_user_id = new_name.lower()
        test_new_user_id = re.sub(r'\s+', '_', test_new_user_id)
        test_new_user_id = re.sub(r'[^\u0600-\u06ff\w]', '', test_new_user_id)
        if not test_new_user_id:
            return None, 'اسم غير صالح'
        
        if test_new_user_id != user_id and test_new_user_id in users:
            return None, 'الاسم الجديد موجود بالفعل'

        # We will rename the file and update users.json
        new_user_id = test_new_user_id
        user['id'] = new_user_id
        user['name'] = new_name

        # Update users.json index
        users[new_user_id] = users.pop(user_id)
        users[new_user_id]['name'] = new_name
        save_users(users)
        
        # Save to new file path
        save_user(user)
        
        # Rename history file if it exists
        old_history_path = os.path.join(USERS_DIR, f'{user_id}_history.json')
        new_history_path = os.path.join(USERS_DIR, f'{new_user_id}_history.json')
        if os.path.exists(old_history_path):
            try:
                os.rename(old_history_path, new_history_path)
            except Exception as e:
                email_logger.error("Failed to rename history file from %s to %s: %s", old_history_path, new_history_path, e)
        
        # Delete old file
        old_path = os.path.join(USERS_DIR, f'{user_id}.json')
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception as e:
                email_logger.error("Failed to delete old user file %s: %s", old_path, e)
    else:
        save_users(users)
        save_user(user)

    # If email change was requested, send verification to the PENDING address
    if email_change_requested:
        send_verification_email(user['pending_email'], user['name'], user['verification_code'])

    return new_user_id, None

# ===== User-Agent Parsers =====
def parse_device_type(ua: Optional[str]) -> str:
    if not ua:
        return 'unknown'
    ua = ua.lower()
    if any(x in ua for x in ['iphone', 'android', 'mobile']):
        return 'Mobile'
    if any(x in ua for x in ['ipad', 'tablet']):
        return 'Tablet'
    return 'Desktop'

def parse_os(ua: Optional[str]) -> str:
    if not ua:
        return 'unknown'
    ua = ua.lower()
    if 'android' in ua:
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

def parse_browser(ua: Optional[str]) -> str:
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

# ===== Data Recording helpers =====
def record_sample(user_id: str, label: int, lang_code: str = 'ar') -> bool:
    """Records an accepted data sample in the user statistics file for a specific language."""
    user = load_user(user_id)
    if not user:
        return False
    label_str = str(label)
    
    if 'samples' not in user:
        user['samples'] = {}
        
    # Migrate flat dicts to lang-structured dicts
    has_flat = False
    for k in list(user['samples'].keys()):
        if k.isdigit():
            has_flat = True
            break
    if has_flat:
        old_samples = user['samples']
        user['samples'] = {'ar': {k: v for k, v in old_samples.items() if k.isdigit()}}
        
    if lang_code not in user['samples']:
        user['samples'][lang_code] = {}
        
    user['samples'][lang_code][label_str] = user['samples'][lang_code].get(label_str, 0) + 1
    user['total_accepted'] = user.get('total_accepted', 0) + 1
    user['last_active'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_user(user)
    return True

def record_rejected(user_id: str) -> None:
    """Records a rejected gesture in the user statistics file."""
    user = load_user(user_id)
    if not user:
        return
    user['rejected'] = user.get('rejected', 0) + 1
    save_user(user)

def get_all_users_stats(lang_code: str = 'ar') -> List[Dict[str, Any]]:
    """Compiles statistics for all registered users, filtered by active language."""
    users = load_users()
    stats = []
    for user_id in users:
        user = load_user(user_id)
        if user:
            total = user.get('total_accepted', 0)
            rejected = user.get('rejected', 0)
            quality = round(total / (total + rejected) * 100) if (total + rejected) > 0 else 0
            device = user.get('device_info', {})
            
            # Extract language specific samples
            all_samples = user.get('samples', {})
            has_flat = False
            for k in list(all_samples.keys()):
                if k.isdigit():
                    has_flat = True
                    break
            if has_flat:
                all_samples = {'ar': {k: v for k, v in all_samples.items() if k.isdigit()}}
                user['samples'] = all_samples
                save_user(user)
                
            lang_samples = all_samples.get(lang_code, {})
            
            stats.append({
                'id': user_id,
                'name': user['name'],
                'email': user.get('email', ''),
                'email_verified': user.get('email_verified', True),
                'profile_pic': user.get('profile_pic', ''),
                'role': get_user_role(user),
                'status': user.get('status', 'active'),
                'total': sum(lang_samples.values()),
                'rejected': rejected,
                'quality': quality,
                'letters_count': len(lang_samples),
                'last_active': user.get('last_active', '-'),
                'created': user.get('created', '-'),
                'samples': lang_samples,
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

def delete_user(user_id: str) -> None:
    """Deletes a user completely from index and disk, including profile picture if any."""
    user = load_user(user_id)
    if user and user.get('profile_pic'):
        old_pic_path = user['profile_pic'].lstrip('/')
        if old_pic_path.startswith('static/') and os.path.exists(old_pic_path):
            try:
                os.remove(old_pic_path)
            except Exception as e:
                email_logger.error("Failed to delete profile picture of deleted user: %s", e)

    users = load_users()
    if user_id in users:
        del users[user_id]
        save_users(users)
    path = os.path.join(USERS_DIR, f'{user_id}.json')
    if os.path.exists(path):
        os.remove(path)

def disable_user(user_id: str) -> None:
    """Sets a user's status to disabled."""
    users = load_users()
    if user_id in users:
        users[user_id]['status'] = 'disabled'
        save_users(users)
    user = load_user(user_id)
    if user:
        user['status'] = 'disabled'
        save_user(user)

def enable_user(user_id: str) -> None:
    """Sets a user's status to active."""
    users = load_users()
    if user_id in users:
        users[user_id]['status'] = 'active'
        save_users(users)
    user = load_user(user_id)
    if user:
        user['status'] = 'active'
        save_user(user)

def get_user_role(user: Dict[str, Any]) -> str:
    role = user.get('role', 'user')
    return role if role in VALID_ROLES else 'user'

def user_is_admin(user: Optional[Dict[str, Any]]) -> bool:
    return bool(user) and get_user_role(user) == 'admin'

def set_user_role(user_id: str, role: str) -> Tuple[bool, Optional[str]]:
    """Sets a user's role (admin or user)."""
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

def verify_admin(password: str) -> bool:
    """Verifies the global admin password securely."""
    if not ADMIN_PASSWORD:
        return False
    try:
        return hmac.compare_digest(password, ADMIN_PASSWORD)
    except (TypeError, ValueError):
        return False

def request_password_reset(identifier: str) -> Tuple[bool, Optional[str]]:
    """
    Generates a 6-digit password reset code, sets expiration (15 mins),
    saves it to the user profile, and sends a password reset email.
    """
    identifier = identifier.strip().lower()
    users = load_users()
    user_id = None
    
    # Check if identifier is email or username
    if '@' in identifier:
        for uid, uinfo in users.items():
            if uinfo.get('email', '').lower() == identifier:
                user_id = uid
                break
    else:
        user_id = re.sub(r'\s+', '_', identifier)
        user_id = re.sub(r'[^\u0600-\u06ff\w]', '', user_id)

    if not user_id:
        return False, 'اسم المستخدم أو البريد الإلكتروني غير مسجل'

    user = load_user(user_id)
    if not user:
        return False, 'المستخدم غير موجود'

    email = user.get('email')
    if not email:
        return False, 'الحساب لا يحتوي على بريد إلكتروني مسجل. يرجى التواصل مع الإدارة لإعادة تعيين كلمة المرور.'

    # Generate cryptographically secure 6-digit reset code
    reset_code = generate_secure_otp()
    expires_at = (datetime.now() + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M')
    
    user['reset_code'] = reset_code
    user['reset_expires'] = expires_at
    save_user(user)

    # Send reset email
    subject = "إعادة تعيين كلمة المرور لحسابك في إشارة"
    body = f"""
    <div style="direction: rtl; font-family: 'Cairo', sans-serif; text-align: right; padding: 20px; border: 1px solid #e1e8e5; border-radius: 12px; max-width: 500px; margin: auto;">
        <h2 style="color: #1D9E75;">مرحباً {user.get('name', user_id)}،</h2>
        <p style="font-size: 16px; color: #4a5568;">لقد تلقينا طلباً لإعادة تعيين كلمة المرور لحسابك في تطبيق <b>إشارة</b>.</p>
        <p style="font-size: 16px; color: #4a5568;">رمز إعادة تعيين كلمة المرور الخاص بك هو:</p>
        <div style="background-color: #f1f5f3; padding: 15px; border-radius: 8px; text-align: center; font-size: 28px; font-weight: bold; letter-spacing: 4px; color: #1D9E75; margin: 20px 0;">
            {reset_code}
        </div>
        <p style="font-size: 14px; color: #e53e3e;">ملاحظة: هذا الرمز صالح لمدة 15 دقيقة فقط.</p>
        <p style="font-size: 14px; color: #718096;">إذا لم تقم بطلب هذا الإجراء، يرجى تجاهل هذا البريد والحرص على أمان حسابك.</p>
        <hr style="border: 0; border-top: 1px solid #e1e8e5; margin: 20px 0;">
        <p style="font-size: 12px; color: #a0aec0; text-align: center;">تطبيق إشارة — 2026</p>
    </div>
    """
    
    # Send email
    mail_server = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    try:
        mail_port = int(os.getenv('MAIL_PORT', '587'))
    except (ValueError, TypeError):
        mail_port = 587
    mail_username = os.getenv('MAIL_USERNAME')
    mail_password = os.getenv('MAIL_PASSWORD')
    mail_use_tls = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 'yes')
    sender = os.getenv('MAIL_DEFAULT_SENDER', 'Ishara <no-reply@ishara.com>')

    msg = MIMEText(body, 'html', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = Header(sender, 'utf-8')
    msg['To'] = email

    if not mail_username or not mail_password:
        email_logger.error("SMTP credentials not set in .env. Falling back to mock email.")
        print_mock_email(email, subject, body)
        return True, None

    try:
        if mail_port == 465:
            server = smtplib.SMTP_SSL(mail_server, mail_port, timeout=10)
        else:
            server = smtplib.SMTP(mail_server, mail_port, timeout=10)
            if mail_use_tls:
                server.starttls()
        
        server.login(mail_username, mail_password)
        server.sendmail(mail_username, [email], msg.as_string())
        server.quit()
        email_logger.info("Password reset email sent successfully to %s", email)
        return True, None
    except Exception as e:
        email_logger.error("Failed to send password reset email to %s: %s", email, e)
        # fallback to printing mock in dev environments even if SMTP failed
        print_mock_email(email, subject, body)
        return True, None

def reset_password_with_code(identifier: str, code: str, new_password: str) -> Tuple[bool, Optional[str]]:
    """
    Verifies the 6-digit reset code and expiration, then updates the user's password.
    """
    identifier = identifier.strip().lower()
    code = code.strip()
    new_password = new_password.strip()
    
    if len(new_password) < 4:
        return False, 'كلمة المرور الجديدة قصيرة جداً (أقل من 4 أحرف)'

    users = load_users()
    user_id = None
    
    # Check if identifier is email or username
    if '@' in identifier:
        for uid, uinfo in users.items():
            if uinfo.get('email', '').lower() == identifier:
                user_id = uid
                break
    else:
        user_id = re.sub(r'\s+', '_', identifier)
        user_id = re.sub(r'[^\u0600-\u06ff\w]', '', user_id)

    if not user_id:
        return False, 'اسم المستخدم أو البريد الإلكتروني غير صحيح'

    user = load_user(user_id)
    if not user:
        return False, 'المستخدم غير موجود'

    stored_code = user.get('reset_code')
    expires_str = user.get('reset_expires')

    if not stored_code or stored_code != code:
        return False, 'رمز إعادة التعيين غير صحيح'

    if expires_str:
        try:
            expires_at = datetime.strptime(expires_str, '%Y-%m-%d %H:%M')
            if datetime.now() > expires_at:
                return False, 'رمز إعادة التعيين منتهي الصلاحية'
        except Exception:
            return False, 'صيغة تاريخ انتهاء الرمز غير صالحة'

    # Update password
    user['password'] = hash_password(new_password)
    # Clear reset token fields
    user['reset_code'] = ''
    user['reset_expires'] = ''
    save_user(user)

    email_logger.info("Password reset successful for user: %s", user_id)
    return True, None
