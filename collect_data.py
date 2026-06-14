#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
أهلاً بك يا بطل البرمجة! 

هذا الملف (collect_data.py) هو "أداة جمع البيانات" المخصصة للغة الإشارة العربية.
لكي يتعلم نموذج الذكاء الاصطناعي الخاص بنا الحروف العربية، يجب أن نغذيه بأمثلة كثيرة (صور أو نقاط لليد).
هذا البرنامج يقوم بفتح الكاميرا الخاصة بك، وباستخدام مكتبة MediaPipe يتعرف على يدك ويرسم عليها نقاطاً، 
وعندما تضغط على زر المسافة (SPACE)، يقوم بحفظ نقاط يدك في ملف جداول (CSV) لكي نستخدمها لاحقاً لتدريب الموديل.

التعليمات الأساسية لتشغيل البرنامج:
1. اختر الحرف العربي الذي تريد تسجيله بالضغط على الأزرار المحددة في لوحة المفاتيح.
2. ضع يدك أمام الكاميرا بشكل واضح حتى تظهر النقاط الملونة.
3. اضغط على زر المسافة (SPACE) لحفظ اللقطة الحالية. (يفضل جمع 200 لقطة على الأقل لكل حرف).
4. اضغط على زر الهروب (ESC) للخروج من البرنامج وحفظ البيانات.
"""

import csv          # لقراءة وكتابة البيانات في ملف CSV
import copy         # لعمل نسخ مستقلة من الصور في الذاكرة لتجنب التعديل على الأصل
import cv2 as cv    # مكتبة OpenCV الشهيرة لمعالجة الصور وفيديو الكاميرا
import numpy as np  # للعمليات الرياضية على المصفوفات
import mediapipe as mp # مكتبة جوجل السحرية لتتبع اليد واستخراج نقاطها الـ 21
import os           # للتعامل مع مجلدات النظام وملفاته
from datetime import datetime  # للتعامل مع الوقت والتاريخ
import config       # ملف التهيئة الخاص بنا لربط المسارات والإعدادات

# =========================================================================
# 1. إعداد وتجهيز مكتبة MediaPipe لتعقب اليد
# =========================================================================
from mediapipe.python.solutions import hands as mp_hands_solutions  # الموديول المسؤول عن تعقب اليد
from mediapipe.python.solutions import drawing_utils as mp_drawing  # موديول مساعد لرسم الخطوط والنقاط على الصورة

# =========================================================================
# 2. تعريف الحروف العربية وأزرار الاختصار المقابلة لها
# =========================================================================
# قمنا برسم خريطة للوحة المفاتيح لكي نسهل عليك اختيار الحروف أثناء جمع البيانات بيد واحدة
ARABIC_LETTERS = {
    '1': 'أ', '2': 'ب', '3': 'ت', '4': 'ث', '5': 'ج',
    '6': 'ح', '7': 'خ', '8': 'د', '9': 'ذ', '0': 'ر',
    'q': 'ز', 'w': 'س', 'e': 'ش', 'r': 'ص', 't': 'ض',
    'y': 'ط', 'u': 'ظ', 'i': 'ع', 'o': 'غ', 'p': 'ف',
    'a': 'ق', 's': 'ك', 'd': 'ل', 'f': 'م', 'g': 'ن',
    'h': 'ه', 'j': 'و', 'k': 'ي', 'l': 'لا'
}

# قواميس مساعدة للتحويل السريع بين المفتاح، الترتيب (Index)، والحرف العربي
KEY_TO_INDEX = {k: i for i, k in enumerate(ARABIC_LETTERS.keys())}
INDEX_TO_LETTER = {i: v for i, v in enumerate(ARABIC_LETTERS.values())}

# جلب مسارات حفظ البيانات من ملف الإعدادات المركزي (config.py)
DATA_DIR = config.DATA_DIR
CSV_PATH = config.TRAIN_CSV
LABELS_PATH = config.LABELS_CSV

# التأكد من وجود المجلد
os.makedirs(DATA_DIR, exist_ok=True)

# إذا لم يكن ملف تصنيفات الحروف (Labels File) موجوداً، نقوم بإنشائه فوراً وكتابة أرقام وأسماء الحروف بداخله
if not os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i, letter in INDEX_TO_LETTER.items():
            writer.writerow([i, letter])
    print(f"تم إنشاء ملف تصنيفات الحروف تلقائياً في: {LABELS_PATH}")

# =========================================================================
# 3. دالة استخراج وتطبيع نقاط اليد (Landmark Normalization)
# =========================================================================
def extract_landmarks(hand_landmarks, image_shape):
    """
    هذه الدالة هي عقل معالجة البيانات!
    الـ MediaPipe يعطينا نقاط اليد بنسب عشرية بالنسبة لأبعاد الصورة (من 0 إلى 1).
    نحن نقوم بالآتي:
    1. تحويل النقاط العشرية إلى إحداثيات بكسل حقيقية بناءً على حجم الصورة (العرض والارتفاع).
    2. جعل النقاط "نسبية" لنقطة معصم اليد (نقطة الصفر)، بحيث نطرح إحداثيات المعصم من كل النقاط. 
       هذا يضمن أنه لو تحركت يدك يميناً أو يساراً في الشاشة، تظل القيم ثابتة ولا تتغير!
    3. نقوم بعملية "تطبيع" (Normalization) بقسمة جميع الإحداثيات على أقصى قيمة مسافة. 
       هذا يجعل البيانات ثابتة حتى لو اقتربت يدك من الكاميرا أو ابتعدت!
    """
    h, w = image_shape[:2]
    landmark_list = []
    
    # 1. حساب البكسلات الحقيقية لكل نقطة
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        landmark_list.append([x, y])

    # 2. تحويل النقاط لتكون نسبية لمعصم اليد (الذي يحمل الترتيب رقم 0)
    base_x, base_y = landmark_list[0]
    rel_landmarks = []
    for x, y in landmark_list:
        rel_landmarks.extend([x - base_x, y - base_y])

    # 3. تطبيع القيم لتكون بين -1 و 1
    max_val = max(abs(v) for v in rel_landmarks) or 1
    normalized = [v / max_val for v in rel_landmarks]
    return normalized

# =========================================================================
# 4. حساب العينات المسجلة حالياً في ملف البيانات
# =========================================================================
def count_samples():
    """
    تقوم بفتح ملف الـ CSV وقراءة الأسطر المخزنة سابقاً لتعرف كم عينة قمنا بجمعها 
    لكل حرف من الحروف الـ 29، حتى نعرض لك التقدم المحرز مباشرة على الشاشة.
    """
    counts = {i: 0 for i in range(len(ARABIC_LETTERS))}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    label = int(row[0])
                    counts[label] = counts.get(label, 0) + 1
    return counts

# =========================================================================
# 5. رسم واجهة المستخدم الرسومية على شاشة الفيديو (HUD / UI Rendering)
# =========================================================================
def draw_ui(image, current_key, current_letter, sample_counts, recorded_this_session, landmark_detected):
    """
    تقوم هذه الدالة برسم نصوص ومعلومات توضيحية على شاشة الكاميرا مباشرة لمساعدة المجمع، مثل:
    - الحرف المحدد حالياً وكم عينة تم جمعها له.
    - حالة تعقب اليد (هل الكاميرا ترى يدك حالياً أم لا).
    - عدد العينات التي قمت بتسجيلها في هذه الجلسة الحالية.
    - شريط ونسبة التقدم الإجمالية للمشروع.
    """
    h, w = image.shape[:2]

    # رسم مستطيل خلفية أسود شفاف في الأعلى لجعل النصوص سهلة القراءة
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # عرض الحرف الحالي المحدد
    if current_letter:
        total = sample_counts.get(KEY_TO_INDEX.get(current_key, -1), 0)
        # إذا جمعنا 200 عينة أو أكثر، يظهر الحرف باللون الأخضر (مكتمل)، وإلا باللون البرتقالي
        status_color = (0, 255, 0) if total >= 200 else (0, 165, 255)
        cv.putText(image, f"Current Letter: {current_letter} | Samples: {total}/200",
                   (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    else:
        cv.putText(image, "Press a key to select letter",
                   (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # عرض حالة تعقب اليد باللون الأخضر أو الأحمر
    hand_status = "Hand Detected" if landmark_detected else "No Hand Detected"
    hand_color = (0, 255, 0) if landmark_detected else (0, 0, 255)
    cv.putText(image, hand_status, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

    # عرض العينات المسجلة في الجلسة الحالية
    cv.putText(image, f"Recorded this session: {recorded_this_session}",
               (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # عرض تعليمات الأزرار
    cv.putText(image, "SPACE: Record | ESC: Exit | Keys 1-9,0,q-l: Select Letter",
               (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # حساب الحروف التي اكتملت بنسبة 200 عينة أو أكثر
    completed = sum(1 for c in sample_counts.values() if c >= 200)
    progress_text = f"Progress: {completed}/29 letters completed"
    cv.putText(image, progress_text, (w - 380, 35),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    return image

# =========================================================================
# 6. البرنامج الرئيسي وتشغيل الحلقة اللانهائية (Main Loop)
# =========================================================================
def main():
    print("=" * 50)
    print("  أداة جمع بيانات لغة الإشارة العربية الموحدة")
    print("=" * 50)
    print("\nالحروف المتاحة في لوحة المفاتيح:")
    for key, letter in ARABIC_LETTERS.items():
        idx = KEY_TO_INDEX[key]
        print(f"  [{key}] = {letter}", end="  ")
        if (idx + 1) % 5 == 0:
            print()
    print("\n")

    # 1. فتح وتجهيز كاميرا الويب الافتراضية (الرقم 0 يعني الكاميرا المدمجة)
    cap = cv.VideoCapture(0)
    # ضبط أبعاد نافذة العرض لتكون بدقة عرض مناسبة وعالية (960 عرض × 540 ارتفاع)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    # 2. ضبط إعدادات تعقب اليد في مكتبة MediaPipe
    hands = mp_hands_solutions.Hands(
        static_image_mode=False,        # تعقب مستمر للفيديو وليس صور ثابتة منفصلة
        max_num_hands=1,                # نهتم بتتبع يد واحدة فقط لجمع البيانات بشكل نقي
        min_detection_confidence=0.7,   # مستوى الثقة الأدنى للكشف عن وجود يد لأول مرة
        min_tracking_confidence=0.5,    # مستوى الثقة الأدنى لتتبع حركة اليد في الإطارات التالية
    )

    # متغيرات للتحكم في حالة البرنامج
    current_key = None
    current_letter = None
    recorded_this_session = 0
    sample_counts = count_samples()
    landmark_detected = False
    current_landmarks = None

    print("البرنامج جاهز للعمل! اختر الحرف، ثم اضغط زر المسافة (SPACE) للتسجيل.")

    # حلقة لانهائية لمعالجة كل إطار (Frame) يأتي من الكاميرا
    while True:
        ret, image = cap.read()
        if not ret:
            print("خطأ: لم نتمكن من قراءة الفيديو من الكاميرا!")
            break

        # قلب الصورة أفقياً مثل المرآة (Mirror Effect) لتسهيل توجيه يدك أمام الشاشة
        image = cv.flip(image, 1)
        # أخذ نسخة من الصورة للرسم عليها وعرضها كواجهة
        debug_image = copy.deepcopy(image)

        # تحويل صيغة ألوان الصورة من BGR (الافتراضية في OpenCV) إلى RGB (المطلوبة في MediaPipe)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # إيقاف إمكانية الكتابة على مصفوفة الصورة مؤقتاً لتسريع المعالجة
        rgb_image.flags.writeable = False
        results = hands.process(rgb_image)
        rgb_image.flags.writeable = True

        landmark_detected = False
        current_landmarks = None

        # إذا اكتشف البرنامج وجود يد في الكاميرا
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_detected = True
                # استخراج النقاط بعد تصفيتها وتطبيعها
                current_landmarks = extract_landmarks(hand_landmarks, image.shape)

                # رسم هيكل ونقاط اليد على الشاشة
                mp_drawing.draw_landmarks(debug_image, hand_landmarks,
                                          mp_hands_solutions.HAND_CONNECTIONS)

        # رسم واجهة المعلومات على إطار الفيديو
        debug_image = draw_ui(debug_image, current_key, current_letter,
                               sample_counts, recorded_this_session, landmark_detected)

        # عرض الفيديو النهائي في نافذة مخصصة
        cv.imshow('Arabic Sign Language - Data Collection', debug_image)

        # ننتظر 10 ملي ثانية لرصد أي زر يقوم المستخدم بالضغط عليه في لوحة المفاتيح
        key = cv.waitKey(10) & 0xFF

        # إذا ضغط المستخدم على زر ESC (الرمز 27) يخرج من الحلقة وينتهي البرنامج
        if key == 27:
            break
            
        # إذا ضغط المستخدم على زر المسافة (SPACE - الرمز 32)
        elif key == 32:
            if current_letter and landmark_detected and current_landmarks:
                label_idx = KEY_TO_INDEX[current_key]
                
                # فتح ملف الـ CSV وإضافة سطر جديد يحتوي على [رقم الحرف، نقاط اليد الـ 42...]
                with open(CSV_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label_idx] + current_landmarks)

                # تحديث العدادات
                sample_counts[label_idx] = sample_counts.get(label_idx, 0) + 1
                recorded_this_session += 1

                total = sample_counts[label_idx]
                print(f"[{current_letter}] تم التسجيل بنجاح! الإجمالي لهذا الحرف: {total}/200", end="\r")

                # وميض أخضر سريع (Flash Effect) على الشاشة لتنبيه المستخدم برمجياً بأن التسجيل تم بنجاح
                flash = debug_image.copy()
                cv.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (0, 255, 0), 20)
                cv.addWeighted(flash, 0.3, debug_image, 0.7, 0, debug_image)
                cv.imshow('Arabic Sign Language - Data Collection', debug_image)
                cv.waitKey(100)  # الانتظار لمدة 100 ملي ثانية لرؤية الوميض

            elif not current_letter:
                print("\nتنبيه: الرجاء اختيار الحرف أولاً من لوحة المفاتيح!")
            elif not landmark_detected:
                print("\nتنبيه: لم يتم العثور على يد في الإطار! تأكد من الإضاءة جيداً.")
        else:
            # التحقق إذا كان الزر المضغوط يطابق أحد حروفنا العربية المعرفة
            char = chr(key).lower() if key < 128 else None
            if char and char in ARABIC_LETTERS:
                current_key = char
                current_letter = ARABIC_LETTERS[char]
                idx = KEY_TO_INDEX[char]
                count = sample_counts.get(idx, 0)
                print(f"\nتم اختيار حرف: {current_letter} | عدد العينات الحالية له: {count}/200")

    # إغلاق الكاميرا والنوافذ وإغلاق مكتبة MediaPipe بأمان لتحرير موارد النظام
    cap.release()
    cv.destroyAllWindows()
    hands.close()

    # طباعة ملخص شامل في الـ Terminal للجلسة الحالية وما تم إنجازه
    print("\n" + "=" * 50)
    print("  ملخص جلسة جمع البيانات")
    print("=" * 50)
    sample_counts = count_samples()
    total_samples = sum(sample_counts.values())
    completed = sum(1 for c in sample_counts.values() if c >= 200)
    print(f"إجمالي عدد العينات التي تم جمعها في الملف: {total_samples}")
    print(f"عدد الحروف المكتملة (200 عينة أو أكثر): {completed}/29")
    print(f"\nتفاصيل حالة كل حرف:")
    for i, letter in INDEX_TO_LETTER.items():
        count = sample_counts.get(i, 0)
        # رسم شريط تقدم نصي مبسط
        bar = "█" * min(count // 10, 20)
        status = "[مكتمل]" if count >= 200 else "[ناقص ]"
        print(f"  {status} {letter}: {count:3d}/200 {bar}")

if __name__ == '__main__':
    main()
