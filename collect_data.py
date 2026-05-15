#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================
أداة جمع بيانات لغة الإشارة العربية الموحدة
===========================================
التعليمات:
- اضغط حرف عربي (من لوحة المفاتيح بالأرقام) لاختيار الحرف
- اضغط SPACE لتسجيل الإيماءة الحالية
- اضغط ESC للخروج
- الهدف: 200 عينة لكل حرف على الأقل

مخطط لوحة المفاتيح:
  1=أ  2=ب  3=ت  4=ث  5=ج  6=ح  7=خ  8=د  9=ذ  0=ر
  q=ز  w=س  e=ش  r=ص  t=ض  y=ط  u=ظ  i=ع  o=غ  p=ف
  a=ق  s=ك  d=ل  f=م  g=ن  h=ه  j=و  k=ي  l=لا
"""

import csv
import copy
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# =============================================
# إعداد MediaPipe
# =============================================
from mediapipe.python.solutions import hands as mp_hands_solutions
from mediapipe.python.solutions import drawing_utils as mp_drawing

# =============================================
# الحروف العربية الموحدة (28 حرف + لا)
# =============================================
ARABIC_LETTERS = {
    '1': 'أ', '2': 'ب', '3': 'ت', '4': 'ث', '5': 'ج',
    '6': 'ح', '7': 'خ', '8': 'د', '9': 'ذ', '0': 'ر',
    'q': 'ز', 'w': 'س', 'e': 'ش', 'r': 'ص', 't': 'ض',
    'y': 'ط', 'u': 'ظ', 'i': 'ع', 'o': 'غ', 'p': 'ف',
    'a': 'ق', 's': 'ك', 'd': 'ل', 'f': 'م', 'g': 'ن',
    'h': 'ه', 'j': 'و', 'k': 'ي', 'l': 'لا'
}

KEY_TO_INDEX = {k: i for i, k in enumerate(ARABIC_LETTERS.keys())}
INDEX_TO_LETTER = {i: v for i, v in enumerate(ARABIC_LETTERS.values())}

# =============================================
# مسارات الملفات
# =============================================
DATA_DIR = 'arabic_data'
CSV_PATH = os.path.join(DATA_DIR, 'arabic_keypoints.csv')
LABELS_PATH = os.path.join(DATA_DIR, 'arabic_labels.csv')

os.makedirs(DATA_DIR, exist_ok=True)

# إنشاء ملف labels لو مش موجود
if not os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i, letter in INDEX_TO_LETTER.items():
            writer.writerow([i, letter])
    print(f"Labels file created: {LABELS_PATH}")

# =============================================
# دالة استخراج نقاط اليد
# =============================================
def extract_landmarks(hand_landmarks, image_shape):
    h, w = image_shape[:2]
    landmark_list = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        landmark_list.append([x, y])

    # تحويل لإحداثيات نسبية
    base_x, base_y = landmark_list[0]
    rel_landmarks = []
    for x, y in landmark_list:
        rel_landmarks.extend([x - base_x, y - base_y])

    # تطبيع
    max_val = max(abs(v) for v in rel_landmarks) or 1
    normalized = [v / max_val for v in rel_landmarks]
    return normalized

# =============================================
# حساب عدد العينات الحالية
# =============================================
def count_samples():
    counts = {i: 0 for i in range(len(ARABIC_LETTERS))}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    label = int(row[0])
                    counts[label] = counts.get(label, 0) + 1
    return counts

# =============================================
# رسم واجهة المستخدم
# =============================================
def draw_ui(image, current_key, current_letter, sample_counts, recorded_this_session, landmark_detected):
    h, w = image.shape[:2]

    # خلفية شفافة للمعلومات
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # الحرف الحالي
    if current_letter:
        total = sample_counts.get(KEY_TO_INDEX.get(current_key, -1), 0)
        status_color = (0, 255, 0) if total >= 200 else (0, 165, 255)
        cv.putText(image, f"Current: {current_letter} | Samples: {total}/200",
                   (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    else:
        cv.putText(image, "Press a key to select letter",
                   (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # حالة الكاميرا
    hand_status = "Hand Detected" if landmark_detected else "No Hand Detected"
    hand_color = (0, 255, 0) if landmark_detected else (0, 0, 255)
    cv.putText(image, hand_status, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

    # عدد العينات المسجلة في الجلسة
    cv.putText(image, f"Recorded this session: {recorded_this_session}",
               (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # تعليمات
    cv.putText(image, "SPACE: Record | ESC: Exit | Keys 1-9,0,q-l: Select Letter",
               (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # مستوى التقدم الكلي
    completed = sum(1 for c in sample_counts.values() if c >= 200)
    progress_text = f"Progress: {completed}/29 letters completed"
    cv.putText(image, progress_text, (w - 380, 35),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    return image

# =============================================
# البرنامج الرئيسي
# =============================================
def main():
    print("=" * 50)
    print("  أداة جمع بيانات لغة الإشارة العربية")
    print("=" * 50)
    print("\nالحروف المتاحة:")
    for key, letter in ARABIC_LETTERS.items():
        idx = KEY_TO_INDEX[key]
        print(f"  [{key}] = {letter}", end="  ")
        if (idx + 1) % 5 == 0:
            print()
    print("\n")

    # إعداد الكاميرا
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    # إعداد MediaPipe
    hands = mp_hands_solutions.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    current_key = None
    current_letter = None
    recorded_this_session = 0
    sample_counts = count_samples()
    landmark_detected = False
    current_landmarks = None

    print("Ready! Press a key to select a letter, then SPACE to record")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # معالجة MediaPipe
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = hands.process(rgb_image)
        rgb_image.flags.writeable = True

        landmark_detected = False
        current_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_detected = True
                current_landmarks = extract_landmarks(hand_landmarks, image.shape)

                # رسم نقاط اليد
                mp_drawing.draw_landmarks(debug_image, hand_landmarks,
                                          mp_hands_solutions.HAND_CONNECTIONS)

        # رسم الواجهة
        debug_image = draw_ui(debug_image, current_key, current_letter,
                              sample_counts, recorded_this_session, landmark_detected)

        cv.imshow('Arabic Sign Language - Data Collection', debug_image)

        # معالجة الأزرار
        key = cv.waitKey(10) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE - تسجيل
            if current_letter and landmark_detected and current_landmarks:
                label_idx = KEY_TO_INDEX[current_key]
                with open(CSV_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label_idx] + current_landmarks)

                sample_counts[label_idx] = sample_counts.get(label_idx, 0) + 1
                recorded_this_session += 1

                total = sample_counts[label_idx]
                print(f"[{current_letter}] recorded! Total: {total}/200", end="\r")

                # وميض أخضر للتأكيد
                flash = debug_image.copy()
                cv.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (0, 255, 0), 20)
                cv.addWeighted(flash, 0.3, debug_image, 0.7, 0, debug_image)
                cv.imshow('Arabic Sign Language - Data Collection', debug_image)
                cv.waitKey(100)

            elif not current_letter:
                print("\nSelect a letter first!")
            elif not landmark_detected:
                print("\nHand not detected! Check lighting")
        else:
            # اختيار حرف
            char = chr(key).lower() if key < 128 else None
            if char and char in ARABIC_LETTERS:
                current_key = char
                current_letter = ARABIC_LETTERS[char]
                idx = KEY_TO_INDEX[char]
                count = sample_counts.get(idx, 0)
                print(f"\nSelected: {current_letter} | Current samples: {count}/200")

    cap.release()
    cv.destroyAllWindows()
    hands.close()

    # ملخص نهائي
    print("\n" + "=" * 50)
    print("  ملخص الجلسة")
    print("=" * 50)
    sample_counts = count_samples()
    total_samples = sum(sample_counts.values())
    completed = sum(1 for c in sample_counts.values() if c >= 200)
    print(f"إجمالي العينات: {total_samples}")
    print(f"الحروف المكتملة (200+ عينة): {completed}/29")
    print(f"\nتفاصيل كل حرف:")
    for i, letter in INDEX_TO_LETTER.items():
        count = sample_counts.get(i, 0)
        bar = "█" * min(count // 10, 20)
        status = "[OK]" if count >= 200 else "[..]"
        print(f"  {status} {letter}: {count:3d}/200 {bar}")

if __name__ == '__main__':
    main()
