#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
أهلاً بك يا صديقي الذكي في ورشة تدريب العقول الإلكترونية! 🧠🚀

هذا الملف (train_model.py) هو "خط أنابيب التدريب" (Training Pipeline) الخاص بنا.
هنا نقوم بتعليم الكمبيوتر كيف يتعرف على لغة الإشارة العربية.
الخطوات التي يمر بها هذا الملف هي:
1. قراءة البيانات (النقاط التي قمنا بجمعها لليد).
2. تقسيم البيانات إلى قسم للتدريب وقسم للاختبار (للتأكد من ذكاء الموديل).
3. بناء شبكة عصبية اصطناعية (Neural Network) باستخدام مكتبة Keras/Tensorflow.
4. تدريب الشبكة العصبية وتكرار المحاولات وتصحيح أخطائها تلقائياً (Training & Backpropagation).
5. تقييم دقة الشبكة العصبية بعد التدريب.
6. حفظ النموذج الناتج وتحويله إلى صيغة TFLite المصغرة والسريعة لاستخدامه مباشرة في الهواتف وموقع الويب.

دعنا نتابع هذا الكود المثير خطوة بخطوة وبشكل بسيط جداً:
"""

import csv          # لقراءة ملفات جداول البيانات (CSV)
import numpy as np  # للعمليات الحسابية والتعامل مع المصفوفات الكبيرة
import os           # للتعامل مع المجلدات والملفات
import shutil       # لنقل ونسخ الملفات إن احتجنا
import json         # لقراءة وحفظ الإعدادات بصيغة JSON
from datetime import datetime  # لإعطاء النماذج طابعاً زمنياً (تاريخ وساعة التدريب)
import matplotlib.pyplot as plt # لرسم الأشكال البيانية
import seaborn as sns          # لرسم مصفوفة الالتباس بشكل ملون ومحترف
from sklearn.preprocessing import LabelEncoder      # لتحويل الكلمات أو أرقام الحروف إلى ترميز رقمي متسلسل تبدأ من 0
from sklearn.model_selection import train_test_split # لتقسيم البيانات لتدريب واختبار بنسب مئوية
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config  # ملف إعدادات المسارات الموحد للمشروع

# إعداد مسجل التقارير الخاص بالتدريب وحفظ المخرجات في ملف training.log
logger = config.get_file_logger('training', config.TRAINING_LOG)

logger.info("=" * 50)
logger.info("  بدء تدريب موديل لغة الإشارة العربية - Training Started")
logger.info("=" * 50)

# =========================================================================
# 1. تحميل إعدادات اللغة وأسماء الحروف (Languages Config & Labels)
# =========================================================================
import sys

# مسار ملف اللغات المدعومة في المشروع
LANGUAGES_CONFIG_PATH = 'languages_data/languages.json'

# تحديد لغة التدريب الافتراضية وهي العربية 'ar'
# وإذا قام المطور بتشغيل الملف هكذا: python train_model.py en، فسيقوم بتدريب النموذج الخاص بالإنجليزية
lang_code = 'ar'
if len(sys.argv) > 1:
    lang_code = sys.argv[1].strip().lower()

logger.info("لغة التدريب المحددة حالياً: %s", lang_code.upper())

# نتحقق من وجود ملف اللغات
if not os.path.exists(LANGUAGES_CONFIG_PATH):
    logger.error("لم يتم العثور على ملف languages.json في المسار: %s", LANGUAGES_CONFIG_PATH)
    raise FileNotFoundError(f"languages.json not found: {LANGUAGES_CONFIG_PATH}")

# قراءة محتويات ملف اللغات
with open(LANGUAGES_CONFIG_PATH, 'r', encoding='utf-8') as f:
    languages_config = json.load(f)

# جلب الإعدادات الخاصة باللغة المحددة
lang_info = languages_config['languages'].get(lang_code)
if not lang_info:
    logger.error("رمز اللغة '%s' غير موجود في ملف languages.json", lang_code)
    raise ValueError(f"Language code '{lang_code}' not found in languages.json")

# تحميل أسماء الحروف من ملف الـ CSV المحدد لهذه اللغة
labels_dict = {}
labels_path = lang_info.get('labels_path')
if labels_path and os.path.exists(labels_path):
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    labels_dict[int(row[0])] = row[1]
                elif len(row) == 1:
                    labels_dict[len(labels_dict)] = row[0]
        logger.info("تم تحميل الحروف بنجاح من ملف: %s", labels_path)
    except Exception as e:
        logger.error("فشل تحميل الحروف من %s: %s", labels_path, e)

# إذا فشل التحميل من الملف، نعتمد على الحروف المكتوبة بشكل افتراضي داخل ملف الإعدادات
if not labels_dict:
    labels_dict = {idx: label for idx, label in enumerate(lang_info.get('labels', []))}

logger.info("عدد الحروف المتاحة للتدريب: %d", len(labels_dict))

# تحديد مسار ملف البيانات المطلوب قراءته للتدريب
train_csv_path = lang_info['dataset_path']

# =========================================================================
# 2. تحميل وقراءة مجموعة البيانات (Dataset Loading & Feature Extraction)
# =========================================================================
X, y_raw = [], []  # X للميزات (features) و y_raw للأرقام التعريفية الأصلية للحروف
user_ids = []      # لتسجيل من هو الشخص الذي قام بتسجيل كل عينة (مفيد للتدقيق والتحليل)

if not os.path.exists(train_csv_path):
    logger.error("ملف بيانات التدريب (CSV) غير موجود: %s", train_csv_path)
    raise FileNotFoundError(f"Training dataset not found: {train_csv_path}")

logger.info("نمط استخراج الميزات الحالي (FEATURE_MODE) هو: %s", config.FEATURE_MODE)
logger.info("جاري قراءة البيانات من: %s", train_csv_path)

with open(train_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row_idx, row in enumerate(reader):
        if not row:
            continue
        try:
            # كشف تلقائي لتنسيق الملف بناءً على عدد الأعمدة في السطر:
            # 1. التنسيق الجديد (44 عموداً): يحتوي على معرف المستخدم/المسجل في البداية
            if len(row) == 44:  
                signer = row[0].strip()
                label = int(row[1])
                landmarks = [float(v) for v in row[2:]]
            # 2. التنسيق القديم (43 عموداً): يبدأ بالرقم التعريفي للحرف مباشرة دون اسم المستخدم
            elif len(row) == 43:  
                signer = "default_signer"
                label = int(row[0])
                landmarks = [float(v) for v in row[1:]]
            else:
                logger.warning("السطر %d: تم تخطيه لأنه غير صالح (طوله %d)", row_idx + 1, len(row))
                continue
            
            if len(landmarks) != 42:
                logger.warning("السطر %d: تم تخطيه لأن عدد النقاط لا يساوي 42 (عدده %d)", row_idx + 1, len(landmarks))
                continue
                
            # حساب واستخراج الميزات الرياضية الإضافية بناءً على نمط معالجة البيانات المحدد
            if config.FEATURE_MODE == "enhanced":
                derived = config.extract_derived_features(landmarks)  # استخراج الـ 62 ميزة الإضافية
                features = landmarks + derived  # دمج الميزات ليصبح المجموع 104 قيمة (42 + 62)
            else:
                features = landmarks  # نكتفي بنقاط اليد الأساسية الـ 42 فقط
                
            y_raw.append(label)
            X.append(features)
            user_ids.append(signer)
        except (ValueError, IndexError) as exc:
            logger.warning("السطر %d: خطأ أثناء تحليل السطر - %s", row_idx + 1, exc)
            continue

# تحويل القوائم إلى مصفوفات Numpy ذات كفاءة رياضية عالية
X = np.array(X, dtype=np.float32)
y_raw = np.array(y_raw, dtype=np.int32)
user_ids = np.array(user_ids)

logger.info("تم تحميل %d عينة تدريب بنجاح.", len(X))

if len(X) == 0:
    logger.error("ملف البيانات فارغ! لا يمكن البدء في التدريب.")
    raise ValueError("No training data found in CSV.")

# =========================================================================
# 3. إعادة ترميز التصنيفات وتجهيزها (Label Re-encoding)
# =========================================================================
# نستخدم LabelEncoder للتأكد من أن أرقام الفئات متسلسلة بدون فجوات وتبدأ من الصفر (0, 1, 2...)
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)  # عدد الفئات (الحروف) التي سنقوم بتدريب الموديل عليها

# خريطة لربط الفئة الرقمية المرمزة الجديدة بالاسم العربي الفعلي للحرف
encoded_labels = {new_idx: labels_dict.get(orig_idx, str(orig_idx))
                  for new_idx, orig_idx in enumerate(le.classes_)}

# طباعة تقرير توزيع العينات لكل حرف للتأكد من توازن البيانات
logger.info("توزيع عينات التدريب لكل حرف:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    letter = encoded_labels.get(label, '?')
    bar = "█" * min(count // 10, 20)
    status = "[كافي]" if count >= 200 else "[قليل]"
    logger.info("  %s %s: %d عينة %s", status, letter, count, bar)

# =========================================================================
# 4. تقسيم البيانات إلى مجموعتي تدريب واختبار (Train/Test Split)
# =========================================================================
logger.info("جاري تقسيم البيانات إلى مجموعة للتدريب ومجموعة للاختبار والتحقق...")
# نقوم بحجز 20% من البيانات للاختبار (test_size=0.2)، والـ 80% الباقية للتدريب.
# نستخدم stratify=y للتأكد من توزيع الحروف بنسب متساوية في المجموعتين لمنع التحيز.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info("حجم بيانات التدريب: %d عينة", len(X_train))
logger.info("حجم بيانات الاختبار: %d عينة", len(X_test))

# =========================================================================
# 5. بناء الشبكة العصبية الاصطناعية (Model Architecture)
# =========================================================================
import tensorflow as tf
from tensorflow import keras

logger.info("جاري بناء وتصميم هيكلية الشبكة العصبية (Sequential MLP)...")

# نقوم بتصميم موديل متعدد الطبقات (MLP) متسلسل:
model = keras.Sequential([
    # طبقة المدخلات: حجمها يساوي عدد الميزات في X (سواء 42 أو 104)
    keras.layers.Input(shape=(X.shape[1],)),

    # الطبقة الأولى: 128 نيورون مفعلة بدالة relu
    keras.layers.Dense(128, activation='relu'),
    # طبقة تطبيع الدفعة (Batch Normalization) لتسريع التدريب واستقرار الأوزان
    keras.layers.BatchNormalization(),
    # طبقة إسقاط (Dropout) بنسبة 30% لمنع الإفراط في التخصيص (Overfitting) عن طريق تعطيل بعض الخلايا عشوائياً
    keras.layers.Dropout(0.3),

    # الطبقة الثانية: 256 نيورون
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    # الطبقة الثالثة: 128 نيورون
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    # الطبقة الرابعة: 64 نيورون
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    # طبقة المخرجات الكلية: حجمها يساوي عدد الحروف، ومفعلة بدالة softmax 
    # لتعطينا نسب احتمالية لكل حرف بحيث يكون مجموع الاحتمالات 1.0 (100%)
    keras.layers.Dense(num_classes, activation='softmax')
])

# تجميع الموديل وتحديد المحسن (Adam optimizer) ودالة الخسارة وطريقة قياس الأداء (Accuracy)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # مناسب للتصنيف متعدد الفئات بأرقام صحيحة
    metrics=['accuracy']
)

# طباعة ملخص كامل لشكل وهيكل الطبقات وعدد البارامترات القابلة للتدريب
model.summary(print_fn=lambda x: logger.info(x))

# =========================================================================
# 6. المراقبة التلقائية أثناء التدريب (Callbacks Setup)
# =========================================================================
best_model_path = os.path.join(config.MODEL_DIR, f'{lang_code}_best_model.h5')

callbacks = [
    # 1. إيقاف مبكر (Early Stopping): لو لاحظ البرنامج أن دقة التحقق لا تتطور لمدة 20 خطوة متتالية، 
    # يتوقف التدريب تلقائياً لتوفير الوقت ومنع الـ Overfitting، ويستعيد أفضل أوزان مسجلة.
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20,
        restore_best_weights=True, verbose=1
    ),
    # 2. تقليل سرعة التعلم (Reduce LR on Plateau): لو ثبتت الخسارة ولم تنخفض لمدة 10 خطوات،
    # يتم تقليل معامل التعلم (Learning Rate) للنصف لتسهيل استقرار الأوزان في الحل الأمثل.
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=10, min_lr=1e-6, verbose=1
    ),
    # 3. حفظ أفضل نسخة (Model Checkpoint): يقوم بحفظ الموديل الحاصل على أعلى دقة تحقق باستمرار
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# =========================================================================
# 7. بدء التدريب الفعلي للنموذج (Model Training)
# =========================================================================
logger.info("بدء تدريب النموذج الآن...")
# نقوم بتدريب الموديل لـ 200 دورة تدريبية (epochs) بحد أقصى، وبأحجام دفعات 32 عينة (batch_size).
# ونقتطع 20% من بيانات التدريب للتحقق الداخلي المستمر أثناء التعلم.
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
logger.info("اكتمل تدريب الموديل بنجاح.")

# =========================================================================
# 8. تقييم دقة النموذج على مجموعة الاختبار المستقلة (Model Evaluation)
# =========================================================================
logger.info("جاري تقييم دقة الموديل على بيانات الاختبار...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
logger.info("  نسبة دقة الموديل في الاختبار: %0.2f%%", test_acc * 100)
logger.info("  قيمة الخسارة في الاختبار: %0.4f", test_loss)

# =========================================================================
# 9. توليد وحفظ التقارير والرسومات البيانية (Reports & Visualization)
# =========================================================================
y_pred = np.argmax(model.predict(X_test), axis=1)
letter_names = [encoded_labels.get(i, str(i)) for i in range(num_classes)]

# حساب المقاييس
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

# توليد وحفظ التقرير النصي المفصل
cls_report_str = classification_report(y_test, y_pred, target_names=letter_names, zero_division=0)
lang_cls_report_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_classification_report.txt")
with open(lang_cls_report_path, 'w', encoding='utf-8') as f:
    f.write(cls_report_str)
logger.info("تم حفظ تقرير التصنيف النصي في: %s", lang_cls_report_path)

if lang_code == 'ar':
    with open(config.CLASSIFICATION_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(cls_report_str)

# حساب مصفوفة الالتباس ورسمها بالرسم البياني
cm = confusion_matrix(y_test, y_pred)

# 1. رسم مصفوفة الالتباس بالأرقام الصحيحة
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
plt.title(f'Confusion Matrix - {lang_code.upper()}', fontsize=16)
plt.ylabel('Actual (الحقيقي)')
plt.xlabel('Predicted (المتوقع)')
plt.tight_layout()
lang_cm_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_confusion_matrix.png")
plt.savefig(lang_cm_path, dpi=150)
if lang_code == 'ar':
    plt.savefig(config.CONFUSION_MATRIX_PNG, dpi=150)
plt.close()

# 2. رسم مصفوفة الالتباس بالنسبة المئوية المطبعة
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16, 14))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
plt.title(f'Normalized Confusion Matrix - {lang_code.upper()}', fontsize=16)
plt.ylabel('Actual (الحقيقي)')
plt.xlabel('Predicted (المتوقع)')
plt.tight_layout()
lang_cm_norm_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_normalized_confusion_matrix.png")
plt.savefig(lang_cm_norm_path, dpi=150)
if lang_code == 'ar':
    plt.savefig(config.NORMALIZED_CONFUSION_MATRIX_PNG, dpi=150)
plt.close()

logger.info("تم حفظ الرسوم البيانية لمصفوفة الالتباس في مجلد reports/.")

# توليد وحفظ تقرير المقاييس بصيغة JSON
cls_report_dict = classification_report(y_test, y_pred, target_names=letter_names, output_dict=True, zero_division=0)
training_report = {
    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'accuracy': float(test_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'classification_report': cls_report_dict
}
lang_report_json_path = os.path.join(config.REPORTS_DIR, f"{lang_code}_training_report.json")
with open(lang_report_json_path, 'w', encoding='utf-8') as f:
    json.dump(training_report, f, indent=2, ensure_ascii=False)
if lang_code == 'ar':
    with open(config.TRAINING_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False)
logger.info("تم حفظ تقرير المقاييس الرقمية JSON في: %s", lang_report_json_path)

# =========================================================================
# 10. حفظ النموذج المكتمل وتحويله لنسخة خفيفة (Model Archiving & TFLite Conversion)
# =========================================================================
logger.info("جاري أرشفة وحفظ الموديل وتحويله لصيغة TFLite...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# أسماء ملفات الموديل المحفوظة بطابع زمني
timestamped_keras_name = f"{lang_code}_sign_model_{timestamp}.keras"
timestamped_keras_path = os.path.join(config.MODEL_DIR, timestamped_keras_name)
timestamped_tflite_name = f"{lang_code}_sign_model_{timestamp}.tflite"
timestamped_tflite_path = os.path.join(config.MODEL_DIR, timestamped_tflite_name)

# اسم كلاسيكي متوافق مع الأنظمة القديمة
legacy_tflite_name = f"{lang_code}_sign_model_{datetime.now().strftime('%Y-%m-%d')}_{test_acc*100:.2f}.tflite"
legacy_tflite_path = os.path.join(config.MODEL_DIR, legacy_tflite_name)

# تحميل أفضل أوزان مسجلة أثناء التدريب
if os.path.exists(best_model_path):
    best_model = keras.models.load_model(best_model_path)
    best_model.save(os.path.join(config.MODEL_DIR, f"{lang_code}_best_model.h5"))
else:
    best_model = model
    best_model.save(os.path.join(config.MODEL_DIR, f"{lang_code}_best_model.h5"))

# حفظ الموديل بصيغة Keras المدعومة رسمياً
best_model.save(timestamped_keras_path)
logger.info("تم حفظ موديل Keras بنجاح في: %s", timestamped_keras_path)

# 11. تحويل الموديل إلى TFLite
# مفسر TFLite يحول الشبكة العصبية الضخمة إلى نسخة مصغرة ومحسنة للعمل بسرعة خيالية على متصفح الويب والهاتف المحمول
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # تفعيل تحسينات الحجم والأداء الافتراضية
tflite_model = converter.convert()

# كتابة ملفات الـ TFLite على الهارد ديسك
with open(timestamped_tflite_path, 'wb') as f:
    f.write(tflite_model)
with open(legacy_tflite_path, 'wb') as f:
    f.write(tflite_model)

# للمحافظة على التوافق الخلفي للمشاريع
if lang_code == 'ar':
    with open(config.MODEL_PATH_TFLITE, 'wb') as f:
        f.write(tflite_model)

logger.info("تم تصدير نسخ الموديل المخفف بنجاح في: %s و %s", timestamped_tflite_path, legacy_tflite_path)

# 12. تحديث المسارات تلقائياً في ملف إعداد اللغات languages.json
try:
    with open(LANGUAGES_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    # نقوم بتحديث قيمة مسار الموديل (model_path) لتشير تلقائياً للنموذج الأفضل الجديد الذي تم تدريبه للتو!
    config_data['languages'][lang_code]['model_path'] = f"arabic_model/{legacy_tflite_name}"
    with open(LANGUAGES_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    logger.info("نجح تحديث ملف الإعدادات المركزي بالمسار الجديد للموديل: arabic_model/%s", legacy_tflite_name)
except Exception as e:
    logger.error("فشل تحديث ملف languages.json: %s", e)

logger.info("اكتملت عملية التدريب بأكملها بنجاح! نسبة دقة الاختبار النهائية للموديل: %0.2f%%", test_acc * 100)