# Audit & Verification Report: Post-Changes

This document presents a comprehensive audit of the security implementations, multi-language registry, dynamic model loading architecture, administrative controls, Git settings, and overall architecture of the **Ishara** Sign Language translation system.

---

## Executive Summary

A complete code-path verification was performed on the modifications reported in the system. The project has successfully migrated from a single-language Arabic sign language recognizer to a dynamic multi-language architecture utilizing a centralized `languages.json` registry. However, the audit revealed critical security vulnerabilities, concurrency bottlenecks, and reliability concerns that must be addressed before the project is ready for a production release.

* **Security & Auth**: Hashed password checks are properly enforced. However, the password reset and email verification mechanisms rely on predictable, non-cryptographically secure random OTPs generated via Python's standard `random` module. Furthermore, email verification codes never expire, and email changes are written to the database before verification. CSRF protection is entirely absent.
* **Multi-Language & Model Loading**: Dynamic model loading and user sessions work correctly. However, a global mutex lock during inference sequentially executes predictions, introducing a severe performance bottleneck. Memory usage is unconstrained, meaning loading numerous custom languages will eventually exhaust system RAM.
* **Admin Controls & Safety**: Endpoints are properly guarded with an `@admin_required` decorator that validates session flags and user records. Frontend confirmation guards are present for server reboots and user account actions but missing for CPU-heavy model training.
* **Git Security**: `.gitignore` is comprehensive, and active sensitive files like `.env` and `arabic_data/users/` are not tracked. However, legacy binary artifacts (`ngrok.exe`, `.ngrok.exe.old`, and legacy `.tflite` model files) remain committed to the Git history.

---

## Verified Changes

The following claims were verified by inspecting actual code paths, middleware, and database operations.

### 1. Profile Password Verification
* **Claim**: Fixed profile password verification when changing username/email.
* **File Path**: `auth.py`
* **Function**: `update_profile`
* **Status**: **PASS**
* **Evidence**:
  ```python
  # Verify current password
  if not check_password_hash(user.get('password', ''), current_password):
      return None, 'كلمة المرور الحالية غير صحيحة'
  ```
* **Notes**: Properly uses Werkzeug's `check_password_hash` rather than plain string comparison.

### 2. Multi-Language Sign Language Support & Registry
* **Claim**: Added multi-language sign language support and created `arabic_data/languages.json` registry.
* **File Path**: `arabic_data/languages.json` / `server.py`
* **Function**: `load_languages_config` / `init_interpreters`
* **Status**: **PASS**
* **Evidence**:
  ```json
  {
    "active_language": "ar",
    "languages": {
      "ar": { ... },
      "en": { ... },
      "fr": { ... }
    }
  }
  ```
* **Notes**: The structure successfully registers codes, names, labels, model paths, and dataset CSV paths.

### 3. Dynamic TFLite Model Loading
* **Claim**: Dynamic TFLite model loading based on selected language.
* **File Path**: `server.py`
* **Function**: `predict_landmarks`
* **Status**: **PASS**
* **Evidence**:
  ```python
  lang_code = session.get('active_lang', languages_config.get('active_language', 'ar'))
  with _interpreters_lock:
      lang_item = interpreters.get(lang_code, interpreters.get('ar'))
      ...
      interp = lang_item['interpreter']
      interp.set_tensor(in_details[0]['index'], input_data)
      interp.invoke()
  ```

### 4. Language Registration Form
* **Claim**: Added language registration form.
* **File Path**: `templates/admin.html` (frontend UI) & `server.py` (backend API)
* **Function**: `add_new_language` (backend) / `addNewLanguage` (frontend)
* **Status**: **PASS**
* **Evidence**:
  ```python
  @app.route('/admin/add-language', methods=['POST'])
  @admin_required
  def add_new_language():
      # Validates input keys and writes new language info to languages.json,
      # copying default model as a placeholder and calling init_interpreters().
  ```

### 5. Admin Controls for Training & Server Reboot
* **Claim**: Added admin controls for model training and server reboot.
* **File Path**: `server.py`
* **Function**: `train_active_model_route` / `restart_server_route`
* **Status**: **PASS**
* **Evidence**:
  ```python
  @app.route('/admin/train-active-model', methods=['POST'])
  @admin_required
  def train_active_model_route():
      # subprocess execution of train_model.py
      
  @app.route('/admin/restart-server', methods=['POST'])
  @admin_required
  def restart_server_route():
      # programmatically re-spawns sys.executable in a thread
  ```

### 6. Gitignore Setup
* **Claim**: Updated `.gitignore` to exclude secrets, datasets, user files, and logs.
* **File Path**: `.gitignore`
* **Status**: **PASS**
* **Evidence**:
  ```text
  .env
  arabic_data/users/
  users_keys.txt
  arabic_data/*.csv
  logs/
  *.tflite
  ```

---

## Failed Verifications

The following verification attempts failed or uncovered critical security weaknesses.

### 1. Verification Codes Expiration
* **Claim**: Verification codes expire correctly.
* **File Path**: `auth.py` / `server.py`
* **Function**: `register_user` / `auth_verify_code`
* **Status**: **FAIL**
* **Evidence**:
  The code generates a 6-digit OTP code `random.randint(100000, 999999)` and saves it to the user JSON. When verifying (`auth_verify_code`), it only checks `user.get('verification_code') != code` with **no expiration timestamp comparison**. The code remains valid indefinitely until used.

### 2. Email Change Verification Order
* **Claim**: Email change cannot occur without successful verification code validation.
* **File Path**: `auth.py`
* **Function**: `update_profile`
* **Status**: **FAIL**
* **Evidence**:
  When a user requests an email update, the backend writes the new email directly to `users.json` and `{user_id}.json` *immediately*:
  ```python
  user['email'] = new_email
  user['email_verified'] = False
  user['verification_code'] = f"{random.randint(100000, 999999)}"
  # ...
  save_user(user)
  ```
  The user is subsequently locked out of the app until they verify it (due to `login_required` middleware). However, if they entered a typo in the new email, the incorrect address is saved immediately, leaving them locked out with no way to receive the verification OTP. The email should only be modified after successful verification of the new address.

### 3. Password Reset Tokens Cryptographic Strength
* **Claim**: Password reset uses secure random tokens.
* **File Path**: `auth.py`
* **Function**: `request_password_reset`
* **Status**: **FAIL**
* **Evidence**:
  The reset token is a simple 6-digit integer generated using Python's standard `random` module:
  ```python
  reset_code = f"{random.randint(100000, 999999)}"
  ```
  The standard `random` module uses the Mersenne Twister algorithm, which is predictable if its internal state is observed. Additionally, a 6-digit code has only 1,000,000 possibilities, rendering it vulnerable to brute force if API rate limits are bypassed.

### 4. Admin Confirmation Safeguards for CPU-Heavy Training
* **Claim**: Confirmation safeguards are applied to all critical admin actions.
* **File Path**: `templates/admin.html`
* **Function**: `trainActiveModel`
* **Status**: **FAIL**
* **Evidence**:
  While server reboot, user suspensions, and role updates prompt the admin for confirmation (via `confirm()`), clicking the **"تدريب الموديل الحالي"** (Train Current Model) button immediately triggers the training pipeline process on the server without any warning or confirmation modal.

---

## Security Findings

| ID | Finding | Classification | Status | File / Route |
| :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | **No CSRF Protection** | **Critical** | **FAIL** | All State-Changing APIs |
| **SEC-02** | **Predictable PRNG (Mersenne Twister)** | **High** | **FAIL** | `auth.py` (OTP generation) |
| **SEC-03** | **No Expiration on Email Verification Codes** | **Medium** | **FAIL** | `auth.py` / `server.py` |
| **SEC-04** | **Unprotected Admin Panel Page Load** | **Medium** | **FAIL** | `server.py` (`/admin` route) |
| **SEC-05** | **Pre-Verification Database Writes on Email Change** | **Medium** | **FAIL** | `auth.py` (`update_profile`) |

### Detailed Vulnerability Analysis:
* **SEC-01 (No CSRF)**: The Flask backend lacks any CSRF middleware (like `Flask-WTF`'s `CSRFProtect`). Since admin actions (`/admin/delete_user`, `/admin/restart-server`) and profile actions rely entirely on cookie-based authentication, an attacker could host a malicious webpage that executes silent POST requests to these endpoints on behalf of an authenticated admin.
* **SEC-02 (Predictable PRNG)**: The use of `random.randint` is a violation of cryptographic standards. Password reset and verification codes must be generated using `secrets.SystemRandom` or `secrets.token_hex`.
* **SEC-04 (Unprotected Admin Page)**: The `/admin` page route lacks the `@admin_required` decorator. Although the administrative backend APIs are protected, the frontend HTML structure, script variables, and navbar contents are accessible to any user who navigates to `/admin`.

---

## Performance Findings

| ID | Finding | Classification | Status | File / Route |
| :--- | :--- | :--- | :--- | :--- |
| **PER-01** | **Global Inference Lock Bottleneck** | **High** | **FAIL** | `server.py` (`_interpreters_lock`) |
| **PER-02** | **Synchronous Model Training (Blocking Request Thread)**| **High** | **FAIL** | `server.py` (`/admin/train-active-model`)|
| **PER-03** | **Unbounded Memory Model Cache** | **Medium** | **FAIL** | `server.py` (`interpreters` dict) |
| **PER-04** | **Inefficient CSV File Parsing on Request** | **Medium** | **FAIL** | `server.py` (`/means` route) |

### Detailed Performance Analysis:
* **PER-01 (Global Lock)**: In `predict_landmarks`, the interpreter operations are placed inside a global lock `with _interpreters_lock`. This means that if 10 users send sign frames concurrently, they are processed sequentially. TFLite inferences will block each other, leading to latency spikes.
* **PER-02 (Synchronous Training)**: Model training is executed using `subprocess.run` inside the request thread:
  ```python
  res = subprocess.run([sys.executable, 'train_model.py', lang_code], capture_output=True, text=True, timeout=180)
  ```
  This blocks the WSGI worker handling the request for up to 3 minutes. In production environments, this will trigger gateway timeouts (504 Gateway Timeout from Nginx/Gunicorn) and exhaust Flask's request threads, causing temporary denial of service.
* **PER-04 (CSV Parsing in Request)**: The `/means` route parses the entire `keypoints.csv` dataset (~12.5MB for Arabic) on demand using `csv.reader` and builds a Pandas DataFrame to compute medians. Under load, this CPU-intensive operation will completely freeze the server.

---

## Architecture Findings

| ID | Finding | Classification | Status | File / Route |
| :--- | :--- | :--- | :--- | :--- |
| **ARC-01** | **File-Based JSON Database Database** | **High** | **FAIL** | `arabic_data/users/` |
| **ARC-02** | **Self-Modifying Code Anti-Pattern** | **High** | **FAIL** | `train_model.py` modifying `server.py` |
| **ARC-03** | **No ACID Guarantees (Concurrent File Writes)** | **Medium** | **FAIL** | `auth.py` / `server.py` |

### Detailed Architectural Analysis:
* **ARC-01 (File-based DB)**: Users are represented as individual `{user_id}.json` files, indexed in a centralized `users.json` file. As the user base grows, search queries, password lookups, and profile writes will become slow and lock-prone.
* **ARC-02 (Self-Modifying Code)**: At the end of training for the Arabic language, `train_model.py` dynamically modifies `server.py` using regular expressions to update the hardcoded `MODEL_PATH` variable:
  ```python
  pattern = r"MODEL_PATH\s*=\s*['\"]arabic_model/[^'\"]+\.tflite['\"]"
  replacement = f"MODEL_PATH = 'arabic_model/{new_model_name}'"
  ```
  This is a fragile anti-pattern that fails on read-only container filesystems (e.g. Docker, AWS Lambda) and risks corrupting `server.py` if the write operation is interrupted. The active model path should be read dynamically from `languages.json`.
* **ARC-03 (No ACID)**: Concurrent registrations or profile updates will trigger race conditions where two threads try to write to `users.json` at the same time, leading to silent data loss or corrupted JSON files.

---

## Git Security Audit

### 1. Git Ignore Checks
A check of ignored files shows that the following patterns are correctly present in `.gitignore`:
* `.env` (**Ignored**)
* `users_keys.txt` (**Ignored**)
* `arabic_data/users/` (**Ignored**)
* `*.csv` (**Ignored**)
* `logs/` (**Ignored**)

We verified using `git ls-files` that none of these active files are tracked by Git.

### 2. Tracked Bloat & Legacy Binaries
However, the repository contains legacy binaries and models that were committed in previous revisions. These are still tracked by Git:
1. `ngrok.exe` (~32.6 MB)
2. `.ngrok.exe.old` (~32.5 MB)
3. `arabic_model/arabic_sign_model_2026-05-22_95.96.tflite` (~74.1 MB)
4. `arabic_model/arabic_sign_model.tflite` (~74.1 MB)

* **Finding**: The presence of these files bloats the repository size (~213 MB in binaries alone) and violates best practices. They should be purged from the repository's git history using `git filter-repo` or `BFG Repo-Cleaner`.

---

## Recommended Fixes

### 1. Implement CSRF Protection (SEC-01)
Install `Flask-WTF` and enable global CSRF protection:
```python
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)
```
Ensure all AJAX requests pass the `X-CSRFToken` header in their request headers.

### 2. Secure OTP and Code Generation (SEC-02, SEC-03)
Rewrite code generation inside `auth.py` using `secrets` and enforce expiration checks:
```python
# auth.py
import secrets
from datetime import datetime, timedelta

def generate_secure_otp() -> str:
    # Generates a cryptographically secure 6-digit code
    return "".join(secrets.choice("0123456789") for _ in range(6))

# Save an email verification code expiration alongside the code
user['verification_code'] = generate_secure_otp()
user['verification_expires'] = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')
```

### 3. Queue Email Changes (SEC-05)
Modify the email update flow to store the proposed email address in a temporary field (e.g. `pending_email`) instead of writing it directly to `email`. Update the main `email` field only when the code sent to the proposed address is validated successfully.

### 4. Remove Self-Modifying Code (ARC-02)
Remove the regex rewrite function from `train_model.py`. Configure `server.py` to read the active model path dynamically from `languages.json` (or fall back to `config.py`) during `init_interpreters()`.

### 5. Move Model Training to a Background Thread (PER-02)
Do not use `subprocess.run` inside the Flask request handler. Instead, start the training script inside a background thread or a task queue (like Celery/RQ) and return a status token to the frontend immediately:
```python
import threading

@app.route('/admin/train-active-model', methods=['POST'])
@admin_required
def train_active_model_route():
    def run_training(lang):
        subprocess.run([sys.executable, 'train_model.py', lang])
        init_interpreters()
        
    threading.Thread(target=run_training, args=(lang_code,)).start()
    return jsonify({'success': True, 'message': 'Training started in background.'})
```

---

## Release Readiness Assessment

Based on the verified findings, the project readiness details are as follows:

* **Authentication & Security**: 55/100 (Missing CSRF, predictable OTPs, immediate database writes on email updates, open admin template).
* **Multi-Language & Model Loading**: 80/100 (Dynamic sessions work, but sequential inference lock impacts concurrent throughput).
* **Administration Panel**: 85/100 (Proper decorator protection on APIs, but missing confirmation on model training).
* **Git Security**: 75/100 (Config files ignored, but legacy large binaries remain in history).
* **Architecture & Database**: 40/100 (File-based DB with no ACID guarantees, self-modifying code script).

### Final Score: 62 / 100

### Project Status: **Development Ready**

> [!WARNING]
> The project is **not ready** for Beta or Production. Running a file-based JSON user database without transaction isolation, coupled with predictable security codes and missing CSRF protection, exposes the application to data corruption and security breaches. It is highly recommended to address `SEC-01`, `SEC-02`, and `ARC-02` prior to initiating any public beta testing.
