#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================
تطبيق التعرف على لغة الإشارة العربية
===========================================
"""

import csv
import copy
import cv2 as cv
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display
import os

from mediapipe.python.solutions import hands as mp_hands_solutions
from mediapipe.python.solutions import drawing_utils as mp_drawing

# =============================================
# إعدادات
# =============================================
MODEL_PATH = 'arabic_model/arabic_sign_model.tflite'
LABELS_PATH = 'arabic_data/arabic_labels.csv'
FONT_PATH = 'Cairo-Regular.ttf'

CONFIDENCE_THRESHOLD = 0.75
HISTORY_LENGTH = 35       # زيادة للتأكيد أكتر
AGREEMENT_RATIO = 0.85    # 85% اتفاق قبل تسجيل الحرف

# =============================================
# تحميل التسميات
# =============================================
labels_dict = {}
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        labels_dict[int(row[0])] = row[1]

# =============================================
# تحميل الموديل TFLite
# =============================================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(landmarks):
    input_data = np.array([landmarks], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# =============================================
# استخراج نقاط اليد
# =============================================
def extract_landmarks(hand_landmarks, image_shape):
    h, w = image_shape[:2]
    landmark_list = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        landmark_list.append([x, y])

    base_x, base_y = landmark_list[0]
    rel = []
    for x, y in landmark_list:
        rel.extend([x - base_x, y - base_y])

    max_val = max(abs(v) for v in rel) or 1
    return [v / max_val for v in rel]

# =============================================
# كتابة نص عربي على الصورة (PIL)
# =============================================
def put_arabic_text(img, text, position, font_size=40, color=(0, 255, 0)):
    if not os.path.exists(FONT_PATH):
        cv.putText(img, text, position, cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return img
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(FONT_PATH, font_size)
        # تحويل اللون من BGR لـ RGB
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, bidi_text, font=font, fill=rgb_color)
        return cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    except Exception as e:
        cv.putText(img, text, position, cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return img

# =============================================
# البرنامج الرئيسي
# =============================================
def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    hands = mp_hands_solutions.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    prediction_history = deque(maxlen=HISTORY_LENGTH)
    current_word = []
    last_letter = None
    no_hand_count = 0
    # cooldown بعد تسجيل حرف عشان متتكررش
    cooldown = 0
    COOLDOWN_FRAMES = 30

    print("✅ التطبيق جاهز!")
    print("  SPACE: مسافة | BACKSPACE: حذف | C: مسح | ESC: خروج")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = hands.process(rgb_image)
        rgb_image.flags.writeable = True

        detected_letter = None
        confidence = 0.0

        if cooldown > 0:
            cooldown -= 1

        if results.multi_hand_landmarks:
            no_hand_count = 0
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(debug_image, hand_landmarks,
                                          mp_hands_solutions.HAND_CONNECTIONS)

                landmarks = extract_landmarks(hand_landmarks, image.shape)
                probs = predict(landmarks)
                predicted_idx = np.argmax(probs)
                confidence = probs[predicted_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_letter = labels_dict.get(predicted_idx, '?')
                    prediction_history.append(predicted_idx)

                # bounding box
                h, w = image.shape[:2]
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1, y1 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
                x2, y2 = min(max(xs) + 20, w), min(max(ys) + 20, h)

                # لون الـ box حسب الـ cooldown
                box_color = (0, 165, 255) if cooldown > 0 else (0, 255, 0)
                cv.rectangle(debug_image, (x1, y1), (x2, y2), box_color, 2)
        else:
            no_hand_count += 1
            if no_hand_count > 10:
                prediction_history.clear()
                last_letter = None

        # =============================================
        # تأكيد الحرف وإضافته للكلمة
        # =============================================
        confirmed_letter = None
        fill_ratio = len(prediction_history) / HISTORY_LENGTH

        if len(prediction_history) == HISTORY_LENGTH and cooldown == 0:
            most_common = Counter(prediction_history).most_common(1)[0]
            if most_common[1] >= HISTORY_LENGTH * AGREEMENT_RATIO:
                confirmed_letter = labels_dict.get(most_common[0], '?')

        if confirmed_letter and confirmed_letter != last_letter:
            current_word.append(confirmed_letter)
            last_letter = confirmed_letter
            prediction_history.clear()
            cooldown = COOLDOWN_FRAMES  # انتظر شوية قبل الحرف الجاي

        elif not confirmed_letter and no_hand_count > 15:
            last_letter = None

        # =============================================
        # رسم الواجهة
        # =============================================
        overlay = debug_image.copy()
        cv.rectangle(overlay, (0, 0), (960, 170), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.65, debug_image, 0.35, 0, debug_image)

        # الحرف الحالي - بالعربي
        if detected_letter and confidence >= CONFIDENCE_THRESHOLD:
            conf_color = (0, 255, 0) if confidence > 0.9 else (0, 165, 255)
            letter_text = f"{detected_letter}  ({confidence*100:.0f}%)"
            debug_image = put_arabic_text(debug_image, letter_text,
                              (500, 8), font_size=40, color=conf_color)
        elif results.multi_hand_landmarks:
            cv.putText(debug_image, f"Low confidence: {confidence*100:.0f}%",
                       (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv.putText(debug_image, "No hand detected",
                       (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # شريط تقدم التأكيد
        bar_width = int(fill_ratio * 200)
        bar_color = (0, 255, 0) if fill_ratio >= 1.0 else (0, 165, 255)
        cv.rectangle(debug_image, (10, 58), (210, 72), (50, 50, 50), -1)
        cv.rectangle(debug_image, (10, 58), (10 + bar_width, 72), bar_color, -1)
        cv.putText(debug_image, "Confirm", (215, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # الكلمة المبنية
        word_text = ''.join(current_word) if current_word else '...'
        debug_image = put_arabic_text(debug_image, word_text,
                                      (10, 85), font_size=48,
                                      color=(255, 255, 0))

        # التعليمات
        cv.putText(debug_image, "SPACE: Space | BACKSPACE: Delete | C: Clear | ESC: Exit",
                   (10, 158), cv.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

        cv.imshow('Arabic Sign Language Recognition', debug_image)

        key = cv.waitKey(10) & 0xFF
        if key == 27:       # ESC
            break
        elif key == 32:     # SPACE
            current_word.append(' ')
        elif key == 8:      # BACKSPACE
            if current_word:
                current_word.pop()
        elif key == ord('c') or key == ord('C'):
            current_word = []
            last_letter = None
            prediction_history.clear()

    cap.release()
    cv.destroyAllWindows()
    hands.close()

    if current_word:
        final_word = ''.join(current_word).strip()
        print(f"\n📝 آخر كلمة: {final_word}")

if __name__ == '__main__':
    main()