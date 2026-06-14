# Project Cleanup Report

This report catalogs and classifies files in the Ishara project to optimize disk usage, reduce technical debt, and ensure maintainability.

> [!IMPORTANT]
> In accordance with project safety rules, **NO FILES HAVE BEEN DELETED AUTOMATICALLY**. This is a read-only audit report for administrative review.

## File Classifications

### 1. SAFE_TO_DELETE
These files are redundant, duplicates, or temporary artifacts that have no runtime or build dependencies.

| File Path | Size | Category | Reason for Deletion | Dependency Impact |
| :--- | :--- | :--- | :--- | :--- |
| `app-release.apk` | 74.19 MB | APK Duplicate | Duplicate of `Ishara-v0.3.0.apk`. | None. Built binaries are distributed externally. |
| `.ngrok.exe.old` | 32.55 MB | Redundant Binary | Leftover previous ngrok version. | None. `ngrok.exe` is the active executable. |
| `arabic_model/arabic_sign_model_2026-05-17_93.59.tflite` | 87.35 KB | Stale Model | Legacy training run artifact. | None. Active Arabic model is `ar_sign_model_2026-06-12_95.97.tflite`. |
| `arabic_model/arabic_sign_model_2026-05-22_95.96.tflite` | 87.35 KB | Stale Model | Legacy training run artifact. | None. |
| `arabic_model/ar_sign_model_2026-06-12_96.03.tflite` | 87.35 KB | Stale Model | Legacy training run artifact. | None. |
| `arabic_model/ar_sign_model_20260612_203917.tflite` | 87.35 KB | Stale Model | Legacy training run artifact. | None. |
| `arabic_model/ar_sign_model_20260612_204125.tflite` | 87.35 KB | Stale Model | Legacy training run artifact. | None. |
| `arabic_model/arabic_sign_model.tflite` | 87.35 KB | Stale Model | Old backup/copy of model. | None. |
| `arabic_model/english_sign_model.tflite` | 87.35 KB | Placeholder Model | Redundant Arabic placeholder. English now uses empty path to signify `not_trained`. | None. Prediction routes gate non-active languages. |
| `arabic_model/fr_sign_model.tflite` | 87.35 KB | Placeholder Model | Redundant Arabic placeholder. French now uses empty path to signify `not_trained`. | None. |
| `arabic_model/ar_sign_model_20260612_203917.keras` | 1.07 MB | Stale Checkpoint | Keras training checkpoint file. | None. TFLite model is used for inference. |
| `arabic_model/ar_sign_model_20260612_204125.keras` | 1.07 MB | Stale Checkpoint | Keras training checkpoint file. | None. |
| `arabic_model/ar_best_model.h5` | 1.07 MB | Stale Checkpoint | Legacy H5 model checkpoint. | None. |
| `arabic_model/arabic_sign_model.h5` | 1.07 MB | Stale Checkpoint | Legacy H5 model checkpoint. | None. |
| `arabic_model/best_model.h5` | 1.07 MB | Stale Checkpoint | Legacy H5 model checkpoint. | None. |
| `arabic_model/app_arabic.py` | 12.35 KB | Unused Script | Old developer utility. | None. All server routes are in `server.py`. |
| `scratch/` (entire directory) | ~52.12 KB | Developer Scratch | Temporary research scripts (`fail_analysis.py`, `patch_admin.py`, etc.) and results. | None. Strictly local developer environment tools. |
| `AUDIT_REPORT_POST_CHANGES.md` | 17.07 KB | Old Report | Temporary report file. | None. |

---

### 2. REVIEW
These files should be kept for historical or documentation reasons, but are not active dependencies of the runtime application.

| File Path | Size | Category | Reason for Review | Dependency Impact |
| :--- | :--- | :--- | :--- | :--- |
| `Ishara-v0.3.0.apk` | 74.19 MB | Mobile Release | The compiled Android client application package. | Essential for distribution, but should be hosted on a release server (e.g., GitHub Releases) rather than in the root workspace. |
| `Eshara_Project_Book.docx` | 45.27 KB | Project Documentation| MS Word document generated from project logs. | None. Non-code asset. |
| `generate_book.py` | 30.51 KB | Doc Script | Script used to generate the project docx book. | Required if document regenerations are needed. |
| `walkthrough.md` | 3.04 KB | Documentation | Previous session documentation. | None. |
| `validate_translations.py` | 2.65 KB | Translation Utility | Script to check template keys against locale dictionaries. | Useful for local CI checks, but not part of production. |
| `test_keypoints.py` | 1.31 KB | Evaluation Utility | Evaluates model against `test_keypoints.csv` using classification report. | Useful for local model validation. |
| `evaluate_external.py` | 7.73 KB | Evaluation Utility | Evaluates model predictions on external sets. | Useful for testing. |
| `extract_landmarks_3d.py` | 3.99 KB | Preprocessing Utility| MediaPipe 3D coordinate converter helper. | Useful utility. |
| `record_poses.py` | 19.30 KB | Collection Utility | Standalone camera recorder script. | Standalone tool. |

---

### 3. KEEP
These are critical production/active development assets that cannot be deleted or relocated.

| File Path | Size | Reason | Dependency Impact |
| :--- | :--- | :--- | :--- |
| `server.py` | 72.62 KB | Core Flask web server and prediction endpoints. | Critical runtime dependency. |
| `auth.py` | 30.43 KB | User management, locks, and secure registration/login. | Critical runtime dependency. |
| `config.py` | 2.95 KB | Global environment and server configurations. | Critical runtime dependency. |
| `arabic_model/ar_sign_model_2026-06-12_95.97.tflite` | 87.35 KB | The active Arabic classification TFLite model. | Critical runtime dependency. |
| `arabic_data/languages.json` | 1.36 KB | Registered language configuration and readiness states. | Critical runtime dependency. |
| `arabic_data/arabic_labels.csv` | 195 B | Arabic character class names. | Critical runtime dependency. |
| `arabic_data/arabic_keypoints.csv` | 12.55 MB | Active Arabic training dataset. | Critical dependency for training. |
| `arabic_data/test_keypoints.csv` | 264.84 KB | Active Arabic testing keypoints dataset. | Critical dependency for validation. |
| `arabic_data/french_labels.csv` | 120 B | French sign character class names (LSF alphabet). | Critical multilingual dependency. |
| `arabic_data/english_labels.csv` | 120 B | English sign character class names. | Critical multilingual dependency. |
| `arabic_data/fr_keypoints.csv` | 1 B | French sign training keypoints dataset file. | Critical multilingual dependency. |
| `arabic_data/fr_test_keypoints.csv` | 1 B | French sign testing keypoints dataset file. | Critical multilingual dependency. |
| `arabic_data/english_keypoints.csv` | 1 B | English sign training keypoints dataset file. | Critical multilingual dependency. |
| `arabic_data/english_test_keypoints.csv` | 1 B | English sign testing keypoints dataset file. | Critical multilingual dependency. |
| `translations/ar.json`, `en.json`, `fr.json` | Various | Locale dictionaries for interface localization. | Critical UI dependency. |
| `templates/`, `static/` | Various | Jinja2 templates and CSS assets. | Critical UI dependency. |
| `Cairo-Regular.ttf`, `fonts/` | Various | Font files for RTL Arabic rendering. | UI styling dependency. |
| `ngrok.exe` | 32.60 MB | Tunneling executable for local testing exposure. | Optional execution dependency. |
