#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================
  مسجّل وضعيات اليد → عظام موديل GLB
=========================================================
  الاستخدام:
    python record_poses.py

  الأزرار:
    1..9, 0, q..l  → اختيار الحرف (نفس collect_data.py)
    SPACE          → تسجيل الوضعية الحالية
    D              → حذف آخر وضعية مسجلة للحرف الحالي
    S              → حفظ الكل الآن (يحفظ تلقائياً عند الخروج)
    ESC            → خروج وحفظ

  الناتج:
    static/poses.js  ← يُستخدم مباشرةً في صفحة translate
                       لتشغيل موديل GLB بعظام حقيقية
=========================================================
"""

import cv2 as cv
import numpy as np
import json, os, math
from collections import defaultdict

# ── MediaPipe ──────────────────────────────────────────
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands_solutions

# ── الحروف ─────────────────────────────────────────────
ARABIC_LETTERS = {
    '1':'أ','2':'ب','3':'ت','4':'ث','5':'ج',
    '6':'ح','7':'خ','8':'د','9':'ذ','0':'ر',
    'q':'ز','w':'س','e':'ش','r':'ص','t':'ض',
    'y':'ط','u':'ظ','i':'ع','o':'غ','p':'ف',
    'a':'ق','s':'ك','d':'ل','f':'م','g':'ن',
    'h':'ه','j':'و','k':'ي','l':'لا'
}

OUTPUT_PATH = os.path.join('static', 'poses.js')

# ── ربط MediaPipe landmarks بعظام GLB ─────────────────
#
#  كل bone في GLB محوره المحلي +Y يشير من المفصل للطرف.
#  لحساب الدوران: نأخذ اتجاه الـ segment من landmark[start]
#  إلى landmark[end] ونحول الـ bone إليه.
#
#  MediaPipe landmark indices:
#   0=wrist
#   1-4  thumb  (cmc, mcp, ip, tip)
#   5-8  index  (mcp, pip, dip, tip)
#   9-12 middle (mcp, pip, dip, tip)
#  13-16 ring   (mcp, pip, dip, tip)
#  17-20 pinky  (mcp, pip, dip, tip)

BONE_SEGMENTS = [
    # (bone_name,  lm_start, lm_end)   – +Y of bone points from start→end
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

# Rest-pose quaternions من GLB (x, y, z, w)
# هذه هي الدوران المحلي لكل bone في وضع الراحة
GLB_REST_QUAT = {
    'pinky_dist':   (-0.0054,  0.0630,  0.0240, 0.9977),
    'pinky_midd':   ( 0.0073,  0.1188, -0.0181, 0.9927),
    'pinky_prox':   (-0.0014, -0.0316, -0.0600, 0.9977),
    'pinky_meta':   ( 0.3648,  0.0301,  0.0858, 0.9266),
    'ring_dist':    ( 0.0116,  0.0010,  0.0164, 0.9998),
    'ring_midd':    ( 0.0055,  0.0001,  0.0071, 1.0000),
    'ring_prox':    ( 0.0146, -0.0339, -0.0610, 0.9975),
    'ring_meta':    ( 0.2504,  0.0215,  0.0740, 0.9651),
    'midd_dist':    (-0.0086, -0.0004, -0.0035, 1.0000),
    'midd_midd':    ( 0.0084,  0.0009,  0.0023, 1.0000),
    'midd_prox':    ( 0.0998, -0.0451, -0.0817, 0.9906),
    'midd_meta':    ( 0.1136,  0.0327,  0.0839, 0.9894),
    'index_dist':   ( 0.0014,  0.0044, -0.0016, 1.0000),
    'index_midd':   ( 0.0505, -0.0029,  0.0088, 0.9987),
    'index_prox':   ( 0.1318, -0.0205, -0.0798, 0.9878),
    'index_meta':   (-0.0107,  0.0182,  0.0636, 0.9978),
    'thumb_dist':   ( 0.0372,  0.1587,  0.0443, 0.9856),
    'thumb_prox':   ( 0.0696,  0.1937,  0.1351, 0.9692),
    'thumb_meta':   ( 0.2204,  0.1092, -0.1566, 0.9565),
    'thumb_trapez': (-0.4081,  0.0179,  0.0802, 0.9093),
    'radius_ulna':  (-0.2708, -0.0012, -0.0205, 0.9624),
}

# ── Quaternion helpers ─────────────────────────────────

def quat_mul(q1, q2):
    """ضرب quaternions (x,y,z,w)"""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )

def quat_conj(q):
    x,y,z,w = q
    return (-x,-y,-z,w)

def quat_from_two_vecs(v_from, v_to):
    """
    أقل دوران من v_from إلى v_to → quaternion (x,y,z,w)
    """
    v_from = v_from / (np.linalg.norm(v_from) + 1e-9)
    v_to   = v_to   / (np.linalg.norm(v_to)   + 1e-9)
    dot = float(np.clip(np.dot(v_from, v_to), -1, 1))

    if dot > 0.9999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.9999:
        # 180° rotation – أي محور عمودي
        perp = np.array([1,0,0]) if abs(v_from[0]) < 0.9 else np.array([0,1,0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return (float(axis[0]), float(axis[1]), float(axis[2]), 0.0)

    axis = np.cross(v_from, v_to)
    s    = math.sqrt((1 + dot) * 2)
    inv  = 1.0 / s
    return (float(axis[0]*inv), float(axis[1]*inv), float(axis[2]*inv), float(s*0.5))

def rotate_vec_by_quat(v, q):
    """دوران vector بـ quaternion"""
    x,y,z,w = q
    # q * (0,v) * q^-1
    vq = (v[0], v[1], v[2], 0.0)
    r  = quat_mul(quat_mul(q, vq), quat_conj(q))
    return np.array([r[0], r[1], r[2]])

# ── تحويل Landmarks → Bone Rotations ──────────────────

def landmarks_to_bone_rotations(lms_3d):
    """
    lms_3d: list of 21 × (x, y, z) في فضاء الكاميرا
    يرجع: dict { bone_name: (qx, qy, qz, qw) } quaternion محلي لكل bone
    """
    pts = np.array(lms_3d, dtype=np.float64)  # (21, 3)

    # --- حساب إطار المرجع للكف ---
    # v_palm_y: من المعصم (0) إلى قاعدة الإصبع الأوسط (9)
    palm_y = pts[9]  - pts[0]
    palm_y_n = palm_y / (np.linalg.norm(palm_y) + 1e-9)

    # v_palm_x: من الخنصر (17) إلى السبابة (5)
    palm_x = pts[5]  - pts[17]
    # جعلها عمودية تماماً على palm_y
    palm_x -= np.dot(palm_x, palm_y_n) * palm_y_n
    palm_x_n = palm_x / (np.linalg.norm(palm_x) + 1e-9)

    palm_z_n = np.cross(palm_x_n, palm_y_n)  # عمودي على سطح الكف

    # مصفوفة دوران الكف في فضاء العالم (cols = محاور)
    R_palm = np.column_stack([palm_x_n, palm_y_n, palm_z_n])  # (3×3)

    # --- دوران radius_ulna (جذر العظام) ---
    # في GLB، +Y لـ radius_ulna يشير للأعلى (اتجاه الكف)
    rest_y_world = rotate_vec_by_quat(np.array([0,1,0]), GLB_REST_QUAT['radius_ulna'])
    target_y     = palm_y_n
    delta_root   = quat_from_two_vecs(rest_y_world, target_y)

    result = {}
    result['radius_ulna'] = tuple(round(v,6) for v in delta_root)

    # --- لكل bone: حساب الدوران المحلي ---
    for bone_name, lm_start, lm_end in BONE_SEGMENTS:
        seg = pts[lm_end] - pts[lm_start]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            result[bone_name] = (0.0, 0.0, 0.0, 1.0)
            continue

        # الاتجاه المطلوب في فضاء العالم
        target_dir_world = seg / seg_len

        # الـ rest +Y للـ bone في فضاء العالم
        rest_q = GLB_REST_QUAT[bone_name]
        rest_y_world = rotate_vec_by_quat(np.array([0.0, 1.0, 0.0]), rest_q)

        # delta = الدوران من rest_y_world → target_dir_world
        delta = quat_from_two_vecs(rest_y_world, target_dir_world)

        # الـ quaternion النهائي = delta * rest
        final_q = quat_mul(delta, rest_q)
        # normalize
        n = math.sqrt(sum(v*v for v in final_q)) + 1e-12
        final_q = tuple(v/n for v in final_q)

        result[bone_name] = tuple(round(v, 6) for v in final_q)

    return result

# ── تحميل الوضعيات المحفوظة ────────────────────────────

def load_existing_poses():
    if not os.path.exists(OUTPUT_PATH):
        return {}
    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        # استخراج JSON من: const POSES = {...};
        start = content.find('{')
        end   = content.rfind('}') + 1
        if start < 0 or end <= start:
            return {}
        return json.loads(content[start:end])
    except Exception as e:
        print(f'⚠ تحذير: تعذّر تحميل poses.js ({e})')
        return {}

def save_poses(poses):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    js = 'const POSES = ' + json.dumps(poses, ensure_ascii=False, indent=2) + ';\n'
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(js)
    print(f'✅ حُفظ → {OUTPUT_PATH}  ({len(poses)} حرف)')

# ── واجهة OpenCV ───────────────────────────────────────

def draw_landmarks(img, hand_lms, h, w):
    conns = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    colors = [(0,255,170),(0,200,255),(170,0,255),(255,100,180),(80,180,255)]
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms.landmark]
    for i,(a,b) in enumerate(conns):
        c = colors[min(max(a,b)//5, 4)]
        cv.line(img, pts[a], pts[b], c, 2, cv.LINE_AA)
    for i,p in enumerate(pts):
        c = colors[min(i//5, 4)]
        cv.circle(img, p, 5, c, -1, cv.LINE_AA)
        cv.circle(img, p, 5, (255,255,255), 1, cv.LINE_AA)

from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont

# تحميل الخطوط مرة واحدة عند البدء
def _load_fonts():
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
    """رسم نص عربي على صورة OpenCV باستخدام PIL"""
    pil_img = PILImage.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    draw    = PILDraw.Draw(pil_img)
    font    = _pil_font(font_name, size)
    draw.text(pos, text, font=font, fill=color, anchor=anchor)
    img_bgr[:] = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

def draw_ui(img, letter, letter_count, status_msg, recorded_letters):
    h, w = img.shape[:2]

    # ── شريط علوي ──────────────────────────────────────
    overlay = img.copy()
    cv.rectangle(overlay, (0,0), (w, 52), (8,12,25), -1)
    cv.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv.putText(img, "Hand Pose Recorder  |  Arabic Sign Language",
               (12, 33), cv.FONT_HERSHEY_SIMPLEX, 0.62, (0,212,170), 1, cv.LINE_AA)

    # ── بوكس الحرف الحالي (يمين) ───────────────────────
    box_x, box_y, box_w, box_h = w - 175, 60, 165, 100
    cv.rectangle(img, (box_x, box_y), (box_x+box_w, box_y+box_h), (12,18,38), -1)
    cv.rectangle(img, (box_x, box_y), (box_x+box_w, box_y+box_h), (0,212,170), 1)

def draw_ui(img, letter, letter_count, status_msg, recorded_letters):
    h, w = img.shape[:2]

    # ── شريط علوي ──────────────────────────────────────
    overlay = img.copy()
    cv.rectangle(overlay, (0,0), (w, 52), (8,12,25), -1)
    cv.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv.putText(img, "Hand Pose Recorder  |  Arabic Sign Language",
               (12, 33), cv.FONT_HERSHEY_SIMPLEX, 0.56, (0,212,170), 1, cv.LINE_AA)

    # ── بوكس الحرف الحالي (يمين) ───────────────────────
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

    # ── شريط الأزرار (أسفل) ────────────────────────────
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

    saved_count = len(recorded_letters)
    put_arabic(img, f'محفوظ: {saved_count}/29',
               (bx + 6, h - margin_b - 4),
               font_name='semibold', size=14,
               color=(0,212,170), anchor='la')

    # ── رسالة الحالة ────────────────────────────────────
    if status_msg:
        clean = status_msg.replace('✅','').replace('⚠','').replace('🗑','').replace('💾','').strip()
        is_ok = any(k in status_msg for k in ('✅','تم','حُفظ','سُجّل'))
        color_ar = (0,230,140) if is_ok else (120,180,255)
        put_arabic(img, clean,
                   (w - 10, h - btn_h - margin_b - 14),
                   font_name='regular', size=15,
                   color=color_ar, anchor='ra')

    return img

# ── الحلقة الرئيسية ────────────────────────────────────

def main():
    print("\n=== مسجّل وضعيات اليد ===")
    print("الأزرار: [1..l] اختر الحرف | [SPACE] سجّل | [D] احذف | [S] احفظ | [ESC] خروج\n")

    poses = load_existing_poses()
    print(f"✔ حُمّل {len(poses)} حرف من poses.js\n")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ لا يمكن فتح الكاميرا")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,  720)

    hands = mp_hands_solutions.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.55
    )

    current_letter = None
    status_msg     = 'اختر حرفاً ثم ثبّت يدك واضغط SPACE'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False
        lms_3d = None

        if results.multi_hand_landmarks:
            for hls in results.multi_hand_landmarks:
                hand_detected = True
                draw_landmarks(frame, hls, h, w)

                # استخرج 3D landmarks (x,y,z) — z نسبي للعمق
                lms_3d = [(lm.x - hls.landmark[0].x,
                           lm.y - hls.landmark[0].y,
                           lm.z - hls.landmark[0].z)
                          for lm in hls.landmark]

        letter_count = 1 if (current_letter and current_letter in poses) else 0
        frame = draw_ui(frame, current_letter, letter_count, status_msg, poses)

        # مؤشر الكاميرا
        dot_color = (0,255,100) if hand_detected else (60,60,80)
        cv.circle(frame, (w-200, 32), 7, dot_color, -1)
        if not hand_detected:
            cv.putText(frame, 'No hand', (w-188, 37),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1, cv.LINE_AA)

        cv.imshow('Hand Pose Recorder', frame)

        key = cv.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif chr(key) in ARABIC_LETTERS if key < 128 else False:
            current_letter = ARABIC_LETTERS[chr(key)]
            in_poses = '✅' if current_letter in poses else '○'
            status_msg = f'{in_poses} الحرف: {current_letter} — ثبّت يدك واضغط SPACE'

        elif key == ord(' '):  # SPACE → تسجيل
            if not current_letter:
                status_msg = '⚠ اختر حرفاً أولاً'
            elif not hand_detected or lms_3d is None:
                status_msg = '⚠ لم تُكتشف يد'
            else:
                bone_rots = landmarks_to_bone_rotations(lms_3d)
                poses[current_letter] = bone_rots
                status_msg = f'✅ تم تسجيل حرف ({current_letter}) — {len(poses)} حرف إجمالاً'
                print(f'  ✅ سُجّل: {current_letter}  |  {len(bone_rots)} bone')

        elif key == ord('d') or key == ord('D'):  # حذف
            if current_letter and current_letter in poses:
                del poses[current_letter]
                status_msg = f'🗑 حُذف: {current_letter}'
            else:
                status_msg = '⚠ لا يوجد وضعية لهذا الحرف'

        elif key == ord('s') or key == ord('S'):  # حفظ
            save_poses(poses)
            status_msg = f'💾 حُفظ {len(poses)} حرف → {OUTPUT_PATH}'

    # حفظ تلقائي عند الخروج
    save_poses(poses)
    cap.release()
    hands.close()
    cv.destroyAllWindows()
    print("\n✅ انتهى. poses.js جاهز للاستخدام في صفحة translate.")

if __name__ == '__main__':
    main()