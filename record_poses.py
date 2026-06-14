#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
أهلاً بك يا صديقي الذكي في عالم الأبعاد الثلاثية ورسوم الكرتون المتحركة! 🎮✨

هذا الملف الرائع (record_poses.py) هو "مسجّل وضعيات اليد" (Hand Pose Recorder).
تخيل أن موقع الويب الخاص بنا يحتوي على يد افتراضية ثلاثية الأبعاد (GLB Model) تقوم بمحاكاة حركة يدك لتعليم المستخدمين لغة الإشارة.
لكي نعرف كيف نحرك عظام اليد الافتراضية بشكل واقعي، نستخدم هذا الملف لتسجيل "الإحداثيات الثلاثية الأبعاد" (X, Y, Z) لليد لكل حرف عربي.
البرنامج يقوم بالآتي:
1. يفتح الكاميرا ويتتبع يدك بأبعادها الثلاثية (بما فيها العمق Z) باستخدام MediaPipe.
2. يتيح لك بث هذه البيانات مباشرة عبر بروتوكول الويب الحديث (WebSockets) لرؤية الحركة فوراً في المتصفح!
3. يحفظ هذه الوضعيات في ملف جافا سكريبت اسمه (static/poses.js) لتستخدمه صفحة الترجمة مباشرة لاحقاً.

الأزرار والتحكم:
- الأزرار [1 إلى 9، 0، ومن q إلى l] في لوحة المفاتيح: لاختيار الحرف العربي المطلوب.
- زر المسافة (SPACE): لتسجيل وحفظ الوضعية الحالية لليد للحرف المحدد.
- زر الحرف (D): لحذف الوضعية المسجلة للحرف الحالي.
- زر الحرف (S): لحفظ التغييرات يدوياً (يتم الحفظ تلقائياً عند الخروج).
- زر الهروب (ESC): للخروج من البرنامج وحفظ البيانات بأمان.
"""

import cv2 as cv        # مكتبة معالجة الصور والفيديو OpenCV
import numpy as np      # للعمليات الحسابية والمصفوفات
import json             # للتعامل مع البيانات وحفظها بصيغة JSON
import os               # للتعامل مع مجلدات النظام
import math             # للعمليات الرياضية الهندسية
import config           # ملف إعدادات المسارات الموحد للمشروع
from collections import defaultdict
import asyncio          # للبرمجة غير المتزامنة (مهمة لتشغيل خادم الويب سوكيت دون تعليق الكاميرا)
import threading        # لتشغيل خادم الويب سوكيت في خلفية النظام (Background Thread)
import time             # للتعامل مع الوقت وقياس سرعة الفريمات

# =========================================================================
# 1. إعداد وتجهيز مكتبة تعقب اليد من MediaPipe
# =========================================================================
import mp
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands_solutions

# نحاول استيراد مكتبة websockets لإرسال البيانات الفورية للمتصفح
# وإذا لم تكن مثبتة، نتفادى توقف البرنامج وننبه المطور
try:
    import websockets
except Exception:
    websockets = None

# تعريف خريطة الحروف العربية الموحدة (28 حرف + تركيب لا)
ARABIC_LETTERS = {
    '1':'أ','2':'ب','3':'ت','4':'ث','5':'ج',
    '6':'ح','7':'خ','8':'د','9':'ذ','0':'ر',
    'q':'ز','w':'س','e':'ش','r':'ص','t':'ض',
    'y':'ط','u':'ظ','i':'ع','o':'غ','p':'ف',
    'a':'ق','s':'ك','d':'ل','f':'م','g':'ن',
    'h':'ه','j':'و','k':'ي','l':'لا'
}

# مسار تصدير ملف الوضعيات
OUTPUT_PATH = config.POSES_JS

# =========================================================================
# 2. إعداد بث البيانات المباشر عبر الويب سوكيت (WebSocket Streaming)
# =========================================================================
clients = set()                 # مجموعة لحفظ المتصفحات المتصلة حالياً بالبرنامج
_ws_loop = None                 # حلقة الأحداث الخاصة بالويب سوكيت (Asyncio Event Loop)
_ws_last_send_ts = 0.0          # الوقت الزمني لآخر إرسال (للتحكم في سرعة البث)
_ws_send_interval_s = 1.0 / 30.0 # نرسل البيانات 30 مرة في الثانية تقريباً (~30 FPS) لتكون الحركة ناعمة جداً

async def ws_handler(websocket):
    """دالة تتعامل مع كل اتصال ويب سوكيت جديد وتضيفه لقائمة المستقبلين"""
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # نبقى على الاتصال مفتوحاً ونستمع لأي رسالة
    finally:
        clients.discard(websocket)  # عند قطع الاتصال، نحذفه من المجموعة لتنظيف الذاكرة

async def broadcast(data):
    """إرسال مصفوفة نقاط اليد إلى جميع المتصفحات المتصلة في نفس اللحظة"""
    if not clients:
        return
    msg = json.dumps(data)
    # نقوم بالإرسال للجميع بشكل متوازٍ وسريع جداً
    await asyncio.gather(
        *(c.send(msg) for c in list(clients)),
        return_exceptions=True
    )

def _start_ws_server_in_background(host="localhost", port=8765):
    """
    تشغيل خادم الويب سوكيت في خيط معالجة خلفي (Thread) مستقل.
    هذه خطوة ذكية جداً لأن كود OpenCV يحتاج أن يعمل بأقصى سرعة لعرض الكاميرا،
    ولو قمنا بتشغيل الويب سوكيت في نفس الخيط سيتسبب ذلك في بطء وتقطيع شديد (Lagging) للفيديو.
    """
    if websockets is None:
        print("تنبيه: مكتبة websockets مفقودة. لتفعيل البث المباشر لليد الافتراضية شغل: pip install websockets")
        return

    async def _serve():
        async with websockets.serve(ws_handler, host, port):
            print(f"خادم البث المباشر للويب سوكيت يعمل الآن على: ws://{host}:{port}")
            await asyncio.Future()  # تشغيل الخادم للأبد دون توقف

    def _runner():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        _ws_loop.run_until_complete(_serve())

    # تشغيل الخيط الخلفي وجعله خيطاً خادماً (Daemon Thread) ينتهي تلقائياً عند إغلاق البرنامج الرئيسي
    threading.Thread(target=_runner, daemon=True).start()

def _maybe_broadcast_bone_rots(bone_rots):
    """
    إرسال بيانات النقاط الحالية إلى المتصفح إذا كانت هناك نافذة مفتوحة وتستمع للبث،
    مع الالتزام بحد أقصى للسرعة (30 إطار في الثانية) لمنع الضغط غير الضروري على المعالج.
    """
    global _ws_last_send_ts
    if not bone_rots:
        return
    if _ws_loop is None or not _ws_loop.is_running():
        return
    now = time.time()
    if now - _ws_last_send_ts < _ws_send_interval_s:
        return
    _ws_last_send_ts = now
    try:
        # إرسال البيانات بأمان لخيط العمل الخلفي الخاص بـ asyncio
        asyncio.run_coroutine_threadsafe(broadcast(bone_rots), _ws_loop)
    except Exception:
        pass

# =========================================================================
# 3. توضيح معالم ونقاط اليد الـ 21 (MediaPipe Landmarks)
# =========================================================================
# المعصم (Wrist) يحمل الرقم 0
# الإبهام (Thumb): من 1 إلى 4
# السبابة (Index): من 5 إلى 8
# الوسطى (Middle): من 9 إلى 12
# البنصر (Ring): من 13 إلى 16
# الخنصر (Pinky): من 17 إلى 20
#
# نقوم برسم عظام اليد بربط النقاط ببعضها كالتالي:
BONE_SEGMENTS = [
    ('thumb_trapez',  0,  1),
    ('thumb_meta',    1,  2),
    ('thumb_prox',    2,  3),
    ('thumb_dist',    3,  4),

    ('index_meta',    0,  5),
    ('index_prox',    5,  6),
    ('index_midd',    6,  7),
    ('index_dist',    7,  8),

    ('midd_meta',     0,  9),
    ('midd_prox',     9, 10),
    ('midd_midd',    10, 11),
    ('midd_dist',    11, 12),

    ('ring_meta',     0, 13),
    ('ring_prox',    13, 14),
    ('ring_midd',    14, 15),
    ('ring_dist',    15, 16),

    ('pinky_meta',    0, 17),
    ('pinky_prox',   17, 18),
    ('pinky_midd',   18, 19),
    ('pinky_dist',   19, 20),
]

# =========================================================================
# 4. دوال تنعيم حركة اليد وتقليل الرعشة (Smoothing & Jitter Reduction)
# =========================================================================
def smooth_points(prev, current, alpha=0.7):
    """
    تنعيم إحداثيات النقاط بين الفريمات باستخدام مرشح تمرير منخفض بسيط (Low-Pass Filter).
    المعادلة: القيمة الجديدة = (القيمة السابقة × alpha) + (القيمة الحالية × (1 - alpha))
    هذا يقلل من اهتزاز ورعشة الكاميرا ويجعل حركة اليد تبدو ناعمة جداً وسلسة!
    """
    return [prev[i] * alpha + current[i] * (1 - alpha) for i in range(len(current))]

def adaptive_alpha(prev, curr):
    """
    معامل تنعيم متكيف وذكي جداً!
    يقيس سرعة حركة يدك الحقيقية:
    - إذا كانت يدك تتحرك ببطء أو ثابتة، نستخدم معامل تنعيم عالٍ جداً (0.85) لمنع الاهتزاز الطفيف لليد تماماً.
    - إذا حركت يدك بسرعة، نقوم بخفض التنعيم تلقائياً إلى (0.6) حتى تتبع الكاميرا يدك بسرعة وبدون تأخير (Lag).
    """
    diff = sum(
        abs(curr[i][0] - prev[i][0]) +
        abs(curr[i][1] - prev[i][1]) +
        abs(curr[i][2] - prev[i][2])
        for i in range(21)
    ) / 21.0
    return 0.85 if diff < 0.01 else 0.6

def landmarks_to_points(lms_3d):
    """
    تحويل قائمة النقاط الثلاثية الأبعاد (21 نقطة) إلى قائمة واحدة مسطحة (1D Array) 
    تحتوي على 63 قيمة (x1, y1, z1, x2, y2, z2...) مع تقريب الأرقام لـ 5 خانات عشرية لتصغير حجم الملف.
    """
    pts = []
    for x, y, z in lms_3d:
        pts.extend([round(x, 5), round(y, 5), round(z, 5)])
    return pts

# =========================================================================
# 5. إدارة وحفظ وقراءة ملف الوضعيات (JSON Storage in JavaScript File)
# =========================================================================
def load_existing_poses():
    """
    تقوم بفتح وقراءة ملف (poses.js) إن كان موجوداً من قبل.
    بما أن الملف مكتوب بصيغة جافا سكريبت لتسهيل قراءته في المتصفح هكذا: const POSES = {...};
    فنحن نقوم برياضيات النصوص لاستخراج جزء الـ JSON الموجود بين القوسين { } وتحميله كقاموس بايثون.
    """
    if not os.path.exists(OUTPUT_PATH):
        return {}
    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        start = content.find('{')
        end   = content.rfind('}') + 1
        if start < 0 or end <= start:
            return {}
        return json.loads(content[start:end])
    except Exception as e:
        print(f'تنبيه: لم نتمكن من قراءة ملف الوضعيات poses.js ({e})')
        return {}

def save_poses(poses):
    """حفظ جميع وضعيات الحروف العربية في ملف poses.js بالصيغة المطلوبة للمتصفح"""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    js = 'const POSES = ' + json.dumps(poses, ensure_ascii=False, indent=2) + ';\n'
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(js)
    print(f'تم الحفظ بنجاح في -> {OUTPUT_PATH}  (إجمالي الحروف المسجلة: {len(poses)} حرف)')

# =========================================================================
# 6. رسم اليد والنصوص العربية الرسومية (OpenCV & PIL Rendering)
# =========================================================================
def draw_landmarks(img, hand_lms, h, w):
    """
    رسم نقاط اليد ومفاصلها بألوان زاهية وجميلة على شاشة الفيديو.
    نستخدم تدرجات لونية مختلفة لكل إصبع لجعل الواجهة تبدو احترافية وجذابة جداً للمستخدم!
    """
    conns = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    # لوحة ألوان نيون رائعة (Neon Colors Palette)
    colors = [(0,255,170),(0,200,255),(170,0,255),(255,100,180),(80,180,255)]
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms.landmark]
    
    # رسم خطوط العظام
    for i,(a,b) in enumerate(conns):
        c = colors[min(max(a,b)//5, 4)]
        cv.line(img, pts[a], pts[b], c, 2, cv.LINE_AA)
        
    # رسم دوائر المفاصل
    for i,p in enumerate(pts):
        c = colors[min(i//5, 4)]
        cv.circle(img, p, 5, c, -1, cv.LINE_AA)
        cv.circle(img, p, 5, (255,255,255), 1, cv.LINE_AA)

from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont

def _load_fonts():
    """تحميل خطوط القاهرة الأنيقة (Cairo Fonts) لتظهر الكلمات العربية بشكل بشري مقروء وجميل جداً"""
    base = os.path.join(os.path.dirname(__file__) or '.', 'fonts')
    paths = {
        'bold':    os.path.join(base, 'Cairo-Bold.ttf'),
        'regular': os.path.join(base, 'Cairo-Regular.ttf'),
        'semibold': os.path.join(base, 'Cairo-SemiBold.ttf'),
    }
    fonts = {}
    for name, path in paths.items():
        if os.path.exists(path):
            fonts[name] = path
        else:
            fonts[name] = None
    return fonts

_FONT_PATHS = _load_fonts()

def _pil_font(name, size):
    path = _FONT_PATHS.get(name)
    if path:
        return PILFont.truetype(path, size)
    return PILFont.load_default()

def put_arabic(img_bgr, text, pos, font_name='bold', size=28, color=(0,212,170), anchor='la'):
    """
    دالة سحرية لكتابة نصوص عربية صحيحة وغير مقطعة على صور OpenCV.
    بما أن مكتبة OpenCV لا تدعم اللغة العربية افتراضياً (تظهر الحروف مقلوبة ومقطعة)،
    نستخدم مكتبة arabic_reshaper و bidi لإعادة تشكيل النص العربي وكتابته عبر مكتبة PIL الرائعة،
    ثم نعيد الصورة إلى OpenCV.
    """
    import arabic_reshaper
    from bidi.algorithm import get_display

    text = str(text)

    # إعادة تشكيل وترتيب النص العربي ليقرأ من اليمين لليسار بشكل متصل
    if len(text) > 1:
        text = arabic_reshaper.reshape(text)
        text = get_display(text)

    # تحويل الصورة لمصفوفة PIL للرسم عليها بالخطوط المخصصة
    pil_img = PILImage.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    draw    = PILDraw.Draw(pil_img)
    font    = _pil_font(font_name, size)

    draw.text(pos, text, font=font, fill=color, anchor=anchor)

    # إعادة الصورة لصيغة BGR الخاصة بـ OpenCV
    img_bgr[:] = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

def draw_ui(img, letter, letter_count, status_msg, recorded_letters):
    """
    رسم واجهة المستخدم الرسومية لبرنامج تسجيل الوضعيات.
    تحتوي على:
    - شريط علوي أنيق باسم البرنامج.
    - نافذة جانبية مميزة تعرض الحرف المختار وعدد وضعياته.
    - أزرار تفاعلية ملونة في الأسفل مع توضيح وظيفة كل زر باللغة العربية.
    - رسائل حالة تفاعلية في الزاوية تخبرك بنجاح التسجيل أو الحذف.
    """
    h, w = img.shape[:2]

    # 1. شريط علوي شفاف وجميل
    overlay = img.copy()
    cv.rectangle(overlay, (0,0), (w, 52), (8,12,25), -1)
    cv.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv.putText(img, "Hand Pose Recorder  |  Arabic Sign Language",
               (12, 33), cv.FONT_HERSHEY_SIMPLEX, 0.56, (0,212,170), 1, cv.LINE_AA)

    # 2. نافذة معلومات الحرف المختار (أعلى اليمين)
    box_w, box_h = min(160, w//4), 95
    box_x = w - box_w - 5
    box_y = 58
    cv.rectangle(img, (box_x, box_y), (box_x+box_w, box_y+box_h), (12,18,38), -1)
    cv.rectangle(img, (box_x, box_y), (box_x+box_w, box_y+box_h), (0,212,170), 1)

    if letter:
        put_arabic(img, letter,
                   (box_x + box_w - 10, box_y + 10),
                   font_name='bold', size=46,
                   color=(167,243,208), anchor='ra')
        put_arabic(img, f'مسجّل: {letter_count}',
                   (box_x + box_w - 8, box_y + box_h - 10),
                   font_name='regular', size=14,
                   color=(107,114,128), anchor='ra')
    else:
        put_arabic(img, 'اختر حرفاً',
                   (box_x + box_w - 8, box_y + box_h//2 + 6),
                   font_name='regular', size=14,
                   color=(75,85,99), anchor='ra')

    # 3. شريط الأزرار والتلميحات التفاعلية (أسفل اليسار)
    hints = [
        ('SPACE', 'تسجيل'),
        ('D',     'حذف'),
        ('S',     'حفظ'),
        ('ESC',   'خروج'),
    ]
    bx = 8
    btn_w, btn_h = 68, 32
    margin_b = 8
    for key, label_ar in hints:
        y1 = h - btn_h - margin_b
        y2 = h - margin_b
        cv.rectangle(img, (bx, y1), (bx+btn_w, y2), (18,26,52), -1)
        cv.rectangle(img, (bx, y1), (bx+btn_w, y2), (50,62,95), 1)
        cv.putText(img, key, (bx+5, y1+btn_h//2-2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.38, (0,212,170), 1, cv.LINE_AA)
        put_arabic(img, label_ar,
                   (bx + btn_w - 3, y2 - 4),
                   font_name='regular', size=13,
                   color=(148,163,184), anchor='ra')
        bx += btn_w + 5

    # عرض عدد الحروف الإجمالية التي قمنا بحفظ وضعياتها
    saved_count = len(recorded_letters)
    put_arabic(img, f'محفوظ: {saved_count}/29',
               (bx + 6, h - margin_b - 4),
               font_name='semibold', size=14,
               color=(0,212,170), anchor='la')

    # 4. عرض رسائل الحالة الديناميكية في الأسفل
    if status_msg:
        clean = status_msg.replace('[OK]','').replace('[!]','').replace('[DEL]','').replace('[SAVE]','').strip()
        is_ok = any(k in status_msg for k in ('[OK]','تم','حُفظ','سُجّل'))
        color_ar = (0,230,140) if is_ok else (120,180,255)
        put_arabic(img, clean,
                   (w - 10, h - btn_h - margin_b - 14),
                   font_name='regular', size=15,
                   color=color_ar, anchor='ra')

    return img

# =========================================================================
# 7. حلقة المعالجة والتشغيل الرئيسية (Main Application Loop)
# =========================================================================
def main():
    print("\n=== مسجّل وضعيات اليد ثلاثي الأبعاد ===")
    print("التحكم: [1..l] اختر الحرف العربي | [SPACE] سجّل الوضعية | [D] احذف | [S] احفظ التغييرات | [ESC] خروج\n")

    # تشغيل خادم الويب سوكيت الخلفي
    _start_ws_server_in_background("localhost", 8765)

    # تحميل الوضعيات الحالية المخزنة
    poses = load_existing_poses()
    print(f"تم تحميل {len(poses)} حرفاً من ملف poses.js بنجاح.\n")

    # فتح الكاميرا
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("خطأ: لم نتمكن من فتح الكاميرا!")
        return

    # إعداد دقة كاميرا عالية
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,  720)

    # تهيئة كاشف اليد من MediaPipe
    hands = mp_hands_solutions.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.55
    )

    # متغيرات تتبع الحالة
    current_letter = None
    status_msg     = 'اختر حرفاً ثم ثبّت يدك أمام الكاميرا واضغط SPACE للتسجيل'
    prev_lms_3d    = None
    prev_stream_bone_rots = None
    bone_rots_raw = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # قلب الصورة أفقياً لتشبه المرآة
        frame = cv.flip(frame, 1)
        h, w  = frame.shape[:2]
        
        # تحويل الألوان لصيغة MediaPipe
        rgb   = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False
        lms_3d = None

        # إذا التقطت الكاميرا يداً
        if results.multi_hand_landmarks:
            for hls in results.multi_hand_landmarks:
                hand_detected = True
                # رسم هيكل اليد الملون على الإطار
                draw_landmarks(frame, hls, h, w)

                # استخراج الإحداثيات ثلاثية الأبعاد وجعلها نسبية لنقطة المعصم
                lms_3d = [(lm.x - hls.landmark[0].x,
                           lm.y - hls.landmark[0].y,
                           lm.z - hls.landmark[0].z)
                          for lm in hls.landmark]
                
                # قياس وتطبيع الحجم الكلي لليد لتفادي تغير القيم عند الاقتراب أو الابتعاد عن الكاميرا
                scale = max(abs(v) for pt in lms_3d for v in pt) or 1
                lms_3d = [(x/scale, y/scale, z/scale) for (x, y, z) in lms_3d]
                
                # تطبيق التنعيم التكيفي الذكي
                if prev_lms_3d is not None and lms_3d is not None:
                    alpha = adaptive_alpha(prev_lms_3d, lms_3d)
                    lms_3d = [
                        (
                            prev_lms_3d[i][0] * alpha + lms_3d[i][0] * (1 - alpha),
                            prev_lms_3d[i][1] * alpha + lms_3d[i][1] * (1 - alpha),
                            prev_lms_3d[i][2] * alpha + lms_3d[i][2] * (1 - alpha),
                        )
                        for i in range(21)
                    ]
                prev_lms_3d = lms_3d

                # تسوية النقاط تمهيداً للبث المباشر عبر الويب سوكيت للمتصفح
                points_raw = landmarks_to_points(lms_3d)
                if prev_stream_bone_rots is not None:
                    points_stream = smooth_points(prev_stream_bone_rots, points_raw, alpha=0.7)
                else:
                    points_stream = points_raw

                prev_stream_bone_rots = points_stream
                # بث النقاط الحالية فورياً عبر الويب سوكيت ليراها مجسم اليد ثلاثي الأبعاد بالمتصفح يتحرك في نفس اللحظة!
                _maybe_broadcast_bone_rots(points_stream)

        # إعادة تهيئة قيم التنعيم عند اختفاء اليد من الكاميرا
        if not hand_detected:
            prev_lms_3d = None
            prev_stream_bone_rots = None
            bone_rots_raw = None

        # رسم الواجهة الرسومية التفاعلية
        letter_count = 1 if (current_letter and current_letter in poses) else 0
        frame = draw_ui(frame, current_letter, letter_count, status_msg, poses)

        # رسم نقطة مضيئة خضراء في الأعلى للإشارة إلى حالة تشغيل الكاميرا وتوفر تعقب اليد
        dot_color = (0,255,100) if hand_detected else (60,60,80)
        cv.circle(frame, (w-200, 32), 7, dot_color, -1)
        if not hand_detected:
            cv.putText(frame, 'No hand', (w-188, 37),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1, cv.LINE_AA)

        # عرض نافذة البرنامج
        cv.imshow('Hand Pose Recorder', frame)

        # الاستماع للوحة المفاتيح
        key = cv.waitKey(1) & 0xFF

        # إذا ضغط ESC للخروج
        if key == 27:
            break

        # إذا اختار حرفاً عربياً
        elif chr(key) in ARABIC_LETTERS if key < 128 else False:
            current_letter = ARABIC_LETTERS[chr(key)]
            in_poses = '[مكتمل]' if current_letter in poses else '[فارغ]'
            status_msg = f'{in_poses} تم تحديد حرف: {current_letter} — اضغط SPACE لتسجيله'

        # إذا ضغط زر المسافة لتسجيل لقطة اليد الحالية للحرف
        elif key == ord(' '):
            if not current_letter:
                status_msg = '[!] تنبيه: الرجاء تحديد الحرف أولاً!'
            elif not hand_detected or lms_3d is None:
                status_msg = '[!] تنبيه: الكاميرا لا ترى أي يد حالياً!'
            else:
                points = points_raw or landmarks_to_points(lms_3d)
                # إذا كانت هناك وضعية سابقة قمنا بحفظها لهذا الحرف، نقوم بعمل تنعيم متوسط معها لكي لا تكون الحركة مفاجئة
                if current_letter in poses:
                    prev = poses[current_letter]
                    points = smooth_points(prev, points, alpha=0.5)
                poses[current_letter] = points
                _maybe_broadcast_bone_rots(points)
                status_msg = f'[OK] تم تسجيل وضعية الحرف ({current_letter}) بنجاح! الإجمالي: {len(poses)} حرفاً'
                print(f'  تم تسجيل الحرف: {current_letter}  |  63 إحداثي ثلاثي أبعاد لليد')

        # إذا ضغط D لحذف وضعية الحرف الحالي
        elif key == ord('d') or key == ord('D'):
            if current_letter and current_letter in poses:
                del poses[current_letter]
                status_msg = f'[DEL] تم حذف وضعية الحرف: {current_letter}'
            else:
                status_msg = '[!] تنبيه: لا توجد وضعية مسجلة لهذا الحرف لحذفها'

        # إذا ضغط S لحفظ كافة التعديلات في الملف فوراً
        elif key == ord('s') or key == ord('S'):
            save_poses(poses)
            status_msg = f'[SAVE] تم حفظ الحروف المسجلة ({len(poses)}) في ملف: {OUTPUT_PATH}'

    # حفظ تلقائي كإجراء أمان إضافي عند الخروج من البرنامج
    save_poses(poses)
    
    # إغلاق الكاميرا والنوافذ بأمان
    cap.release()
    hands.close()
    cv.destroyAllWindows()
    print("\nتم إغلاق مسجل الوضعيات بنجاح. ملف poses.js أصبح جاهزاً للاستخدام في صفحة الترجمة بالموقع!")

if __name__ == '__main__':
    main()