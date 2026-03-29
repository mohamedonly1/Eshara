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

print(f"\n📂 الحروف المتاحة: {len(labels_dict)}")

# تحميل البيانات
X, y = [], []
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])

X = np.array(X)
y = np.array(y)

print(f"✅ تم تحميل {len(X)} عينة")
print(f"📊 توزيع العينات:")

unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    letter = labels_dict.get(label, '?')
    bar = "█" * (count // 10)
    status = "✅" if count >= 200 else "⚠️ "
    print(f"  {status} {letter}: {count} {bar}")

# =============================================
# 2. تقسيم البيانات
# =============================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

num_classes = len(np.unique(y))
print(f"\n🔢 عدد الفئات: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ بيانات التدريب: {len(X_train)}")
print(f"✅ بيانات الاختبار: {len(X_test)}")

# =============================================
# 3. بناء الموديل
# =============================================
import tensorflow as tf
from tensorflow import keras

print("\n🏗️  بناء الموديل...")

model = keras.Sequential([
    keras.layers.Input(shape=(42,)),                          # 21 نقطة × 2 (x,y)

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
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =============================================
# 5. التدريب
# =============================================
print("\n🚀 بدء التدريب...")

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
print("\n📊 تقييم الموديل على بيانات الاختبار:")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  ✅ Accuracy: {test_acc * 100:.2f}%")
print(f"  📉 Loss: {test_loss:.4f}")

# =============================================
# 7. Confusion Matrix
# =============================================
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n📋 تقرير التصنيف:")
letter_names = [labels_dict.get(i, str(i)) for i in range(num_classes)]
print(classification_report(y_test, y_pred, target_names=letter_names))

# رسم Confusion Matrix
plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=letter_names,
            yticklabels=letter_names)
plt.title('Confusion Matrix - لغة الإشارة العربية', fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'), dpi=150)
plt.show()
print(f"✅ تم حفظ Confusion Matrix")

# رسم منحنيات التدريب
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_curves.png'), dpi=150)
plt.show()
print(f"✅ تم حفظ منحنيات التدريب")

# =============================================
# 8. حفظ الموديل بصيغة TFLite
# =============================================
print("\n💾 حفظ الموديل...")

# تحديد مسارات ملفات H5
best_h5_path = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
final_h5_path = os.path.join(MODEL_SAVE_PATH, 'arabic_sign_model.h5')

# تحميل أفضل موديل تم حفظه أثناء التدريب، أو استخدام الموديل الحالي كاحتياطي
if os.path.exists(best_h5_path):
    print("📌 تحميل أفضل موديل من best_model.h5 للتحويل إلى TFLite")
    best_model = keras.models.load_model(best_h5_path)
    # نسخ أفضل موديل إلى الاسم الثابت ليستخدمه السيرفر
    shutil.copy2(best_h5_path, final_h5_path)
else:
    print("⚠️ best_model.h5 غير موجود، سيتم استخدام الموديل الحالي")
    best_model = model
    best_model.save(final_h5_path)

# تحويل أفضل موديل لـ TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# اسم ملف ثابت لتحميله من السيرفر
tflite_path = os.path.join(MODEL_SAVE_PATH, 'arabic_sign_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# اسم ملف مـرقّم يحتوي التاريخ والدقة
date_str = datetime.now().strftime('%Y-%m-%d')
acc_pct = test_acc * 100.0
versioned_name = f"arabic_sign_model_{date_str}_{acc_pct:.2f}.tflite"
versioned_tflite_path = os.path.join(MODEL_SAVE_PATH, versioned_name)
with open(versioned_tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ تم الحفظ:")
print(f"   - {MODEL_SAVE_PATH}/arabic_sign_model.h5")
print(f"   - {MODEL_SAVE_PATH}/arabic_sign_model.tflite")
print(f"   - {MODEL_SAVE_PATH}/{versioned_name}")
print(f"   - {MODEL_SAVE_PATH}/confusion_matrix.png")
print(f"   - {MODEL_SAVE_PATH}/training_curves.png")
print(f"📦 نسخة TFLite المرقمة: {versioned_name}")
print(f"\n🎉 انتهى التدريب! الدقة النهائية: {test_acc * 100:.2f}%")
