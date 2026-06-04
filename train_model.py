#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================
تدريب موديل لغة الإشارة العربية الموحدة
===========================================
شغّل الملف ده بعد ما تجمع البيانات بـ collect_data.py
"""

import csv
import numpy as np
import os
import shutil
from datetime import datetime

# =============================================
# 1. تحميل البيانات
# =============================================
print("=" * 50)
print("  تدريب موديل لغة الإشارة العربية")
print("=" * 50)

DATA_PATH = 'arabic_data/arabic_keypoints.csv'
LABELS_PATH = 'arabic_data/arabic_labels.csv'
MODEL_SAVE_PATH = 'arabic_model'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# تحميل التسميات
labels_dict = {}
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        labels_dict[int(row[0])] = row[1]

print(f"\nLetters available: {len(labels_dict)}")

# تحميل البيانات
X, y_raw = [], []
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            y_raw.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

X = np.array(X)
y_raw = np.array(y_raw)

print(f"Loaded {len(X)} samples")

# =============================================
# FIX: re-encode labels to 0..N-1
# =============================================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y_raw)          # e.g. [0,1,3,9,28] → [0,1,2,3,4]
num_classes = len(le.classes_)

# بناء قاموس الحروف المعاد ترميزها
encoded_labels = {new_idx: labels_dict.get(orig_idx, str(orig_idx))
                  for new_idx, orig_idx in enumerate(le.classes_)}

print(f"Sample distribution:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    letter = encoded_labels.get(label, '?')
    bar = "█" * (count // 10)
    status = "[OK]" if count >= 200 else "[..]"
    print(f"  {status} {letter}: {count} {bar}")

# =============================================
# 2. تقسيم البيانات
# =============================================
from sklearn.model_selection import train_test_split

print(f"\nNumber of classes: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data: {len(X_train)}")
print(f"Testing data: {len(X_test)}")

# =============================================
# 3. بناء الموديل
# =============================================
import tensorflow as tf
from tensorflow import keras

print("\nBuilding model...")

model = keras.Sequential([
    keras.layers.Input(shape=(42,)),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================================
# 4. Callbacks
# =============================================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=10, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# =============================================
# 5. التدريب
# =============================================
print("\nStarting training...")

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# =============================================
# 6. التقييم
# =============================================
print("\nEvaluating model on test data:")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Accuracy: {test_acc * 100:.2f}%")
print(f"  Loss: {test_loss:.4f}")

# =============================================
# 7. Confusion Matrix
# =============================================
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)

letter_names = [encoded_labels.get(i, str(i)) for i in range(num_classes)]
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=letter_names))

plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=letter_names, yticklabels=letter_names)
plt.title('Confusion Matrix - لغة الإشارة العربية', fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'), dpi=150)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True)
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_curves.png'), dpi=150)
plt.show()

# =============================================
# 8. حفظ الموديل + حفظ label encoder
# =============================================
print("\nSaving model...")

best_h5_path = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
final_h5_path = os.path.join(MODEL_SAVE_PATH, 'arabic_sign_model.h5')

if os.path.exists(best_h5_path):
    best_model = keras.models.load_model(best_h5_path)
    shutil.copy2(best_h5_path, final_h5_path)
else:
    best_model = model
    best_model.save(final_h5_path)

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_SAVE_PATH, 'arabic_sign_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

date_str = datetime.now().strftime('%Y-%m-%d')
acc_pct = test_acc * 100.0
versioned_name = f"arabic_sign_model_{date_str}_{acc_pct:.2f}.tflite"
with open(os.path.join(MODEL_SAVE_PATH, versioned_name), 'wb') as f:
    f.write(tflite_model)

# =============================================
# FIX: حفظ labels الصحيحة المعاد ترميزها
# =============================================
fixed_labels_path = 'arabic_data/arabic_labels.csv'
with open(fixed_labels_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for new_idx, letter in encoded_labels.items():
        writer.writerow([new_idx, letter])

print(f"Labels file updated: arabic_labels.csv ({num_classes} classes)")
print(f"Saved:")
print(f"   - {MODEL_SAVE_PATH}/arabic_sign_model.h5")
print(f"   - {MODEL_SAVE_PATH}/arabic_sign_model.tflite")
print(f"   - {MODEL_SAVE_PATH}/{versioned_name}")
print(f"\nTraining complete! Final accuracy: {test_acc * 100:.2f}%")

# =============================================
# تحديث موديل السيرفر تلقائياً أو تفاعلياً
# =============================================
def update_server_model_path(new_versioned_name):
    import re
    server_path = 'server.py'
    if not os.path.exists(server_path):
        print(f"⚠️ لم يتم العثور على ملف {server_path} لتحديث مسار الموديل.")
        return False
    
    with open(server_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # نمط للبحث عن مسار الموديل في السيرفر واستبداله بالجديد
    pattern = r"MODEL_PATH\s*=\s*['\"]arabic_model/[^'\"]+\.tflite['\"]"
    replacement = f"MODEL_PATH = 'arabic_model/{new_versioned_name}'"
    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        with open(server_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ تم تحديث مسار الموديل بنجاح في {server_path} إلى:")
        print(f"   MODEL_PATH = 'arabic_model/{new_versioned_name}'")
        return True
    else:
        print(f"⚠️ لم يتم العثور على تعريف MODEL_PATH في {server_path} لتعديله تلقائياً.")
        return False

print("\n" + "="*50)
print(f"هل تريد تفعيل الموديل الجديد في السيرفر (server.py)؟")
print(f"الموديل النشط حالياً سيتم استبداله بـ: arabic_model/{versioned_name}")

import sys
if not sys.stdin.isatty():
    print("🤖 بيئة التشغيل غير تفاعلية (Non-interactive)، سيتم تحديث الموديل تلقائياً في السيرفر...")
    update_server_model_path(versioned_name)
else:
    try:
        choice = input("اكتب 'y' أو اضغط Enter للتحديث، أو 'n' للإلغاء: ").strip().lower()
        if choice in {'', 'y', 'yes', 'نعم'}:
            update_server_model_path(versioned_name)
        else:
            print("❌ لم يتم تحديث مسار الموديل في السيرفر. يمكنك تحديثه يدوياً في server.py.")
    except (EOFError, KeyboardInterrupt):
        print("\n🤖 تم استشعار إدخال غير صالح أو مقطوع، سيتم تحديث الموديل تلقائياً...")
        update_server_model_path(versioned_name)