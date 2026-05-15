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
import asyncio
import threading
import time

# ── MediaPipe ──────────────────────────────────────────
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands_solutions

try:
    import websockets
except Exception:
    websockets = None

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

# ── WebSocket streaming (bone quaternions) ──────────────
clients = set()
_ws_loop = None
_ws_last_send_ts = 0.0
_ws_send_interval_s = 1.0 / 30.0  # ~30 FPS

async def ws_handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)

async def broadcast(data):
    if not clients:
        return
    msg = json.dumps(data)
    await asyncio.gather(
        *(c.send(msg) for c in list(clients)),
        return_exceptions=True
    )

def _start_ws_server_in_background(host="localhost", port=8765):
    """
    Run a small WS server in a background thread so the OpenCV loop
    never blocks on asyncio.
    """
    if websockets is None:
        print("Warning: websockets dependency missing. Run: pip install websockets")
        return

    async def _serve():
        async with websockets.serve(ws_handler, host, port):
            print(f"WebSocket server running at ws://{host}:{port}")
            await asyncio.Future()  # run forever

    def _runner():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        _ws_loop.run_until_complete(_serve())

    threading.Thread(target=_runner, daemon=True).start()

def _maybe_broadcast_bone_rots(bone_rots):
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
        asyncio.run_coroutine_threadsafe(broadcast(bone_rots), _ws_loop)
    except Exception:
        # Safe: never crash the tracking loop due to WS issues.
        pass

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

# تم حذف GLB_REST_QUAT لأننا سنرسل النقاط الثلاثية الأبعاد (Points) مباشرة
def smooth_points(prev, current, alpha=0.7):
    # تنعيم الإحداثيات (Points) بين الفريمات
    return [prev[i] * alpha + current[i] * (1 - alpha) for i in range(len(current))]

def adaptive_alpha(prev, curr):
    diff = sum(
        abs(curr[i][0] - prev[i][0]) +
        abs(curr[i][1] - prev[i][1]) +
        abs(curr[i][2] - prev[i][2])
        for i in range(21)
    ) / 21.0
    return 0.85 if diff < 0.01 else 0.6

# ── تحويل Landmarks → 3D Points ──────────────────

def landmarks_to_points(lms_3d):
    """
    lms_3d: list of 21 × (x, y, z) في فضاء الكاميرا
    يرجع: قائمة مسطحة (1D Array) تحتوي على 63 قيمة للإحداثيات
    """
    pts = []
    for x, y, z in lms_3d:
        pts.extend([round(x, 5), round(y, 5), round(z, 5)])
    return pts

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
        print(f'Warning: could not load poses.js ({e})')
        return {}

def save_poses(poses):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    js = 'const POSES = ' + json.dumps(poses, ensure_ascii=False, indent=2) + ';\n'
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(js)
    print(f'Saved -> {OUTPUT_PATH}  ({len(poses)} letters)')

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
    import arabic_reshaper
    from bidi.algorithm import get_display

    text = str(text)

    # Arabic text reshaping
    if len(text) > 1:
        text = arabic_reshaper.reshape(text)
        text = get_display(text)

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
        clean = status_msg.replace('[OK]','').replace('[!]','').replace('[DEL]','').replace('[SAVE]','').strip()
        is_ok = any(k in status_msg for k in ('[OK]','تم','حُفظ','سُجّل'))
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

    _start_ws_server_in_background("localhost", 8765)

    poses = load_existing_poses()
    print(f"Loaded {len(poses)} letters from poses.js\n")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
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
    prev_lms_3d    = None
    prev_stream_bone_rots = None
    bone_rots_raw = None

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
                scale = max(abs(v) for pt in lms_3d for v in pt) or 1
                lms_3d = [(x/scale, y/scale, z/scale) for (x, y, z) in lms_3d]
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

                # Live points (for WebSocket streaming)
                points_raw = landmarks_to_points(lms_3d)
                if prev_stream_bone_rots is not None:
                    points_stream = smooth_points(prev_stream_bone_rots, points_raw, alpha=0.7)
                else:
                    points_stream = points_raw

                prev_stream_bone_rots = points_stream
                _maybe_broadcast_bone_rots(points_stream)

        if not hand_detected:
            prev_lms_3d = None
            prev_stream_bone_rots = None
            bone_rots_raw = None

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
            in_poses = '[OK]' if current_letter in poses else 'o'
            status_msg = f'{in_poses} الحرف: {current_letter} — ثبّت يدك واضغط SPACE'

        elif key == ord(' '):  # SPACE → تسجيل
            if not current_letter:
                status_msg = '[!] اختر حرفاً أولاً'
            elif not hand_detected or lms_3d is None:
                status_msg = '[!] لم تُكتشف يد'
            else:
                points = points_raw or landmarks_to_points(lms_3d)
                if current_letter in poses:
                    prev = poses[current_letter]
                    points = smooth_points(prev, points, alpha=0.5)
                poses[current_letter] = points
                _maybe_broadcast_bone_rots(points)
                status_msg = f'[OK] تم تسجيل حرف ({current_letter}) — {len(poses)} حرف إجمالاً'
                print(f'  Recorded: {current_letter}  |  63 points')

        elif key == ord('d') or key == ord('D'):  # حذف
            if current_letter and current_letter in poses:
                del poses[current_letter]
                status_msg = f'[DEL] حُذف: {current_letter}'
            else:
                status_msg = '[!] لا يوجد وضعية لهذا الحرف'

        elif key == ord('s') or key == ord('S'):  # حفظ
            save_poses(poses)
            status_msg = f'[SAVE] حُفظ {len(poses)} حرف -> {OUTPUT_PATH}'

    # حفظ تلقائي عند الخروج
    save_poses(poses)
    cap.release()
    hands.close()
    cv.destroyAllWindows()
    print("\nDone. poses.js is ready for use in the translate page.")

if __name__ == '__main__':
    main()