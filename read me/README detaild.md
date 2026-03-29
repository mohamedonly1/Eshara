# مشروع التعرف على لغة الإشارة العربية الموحدة
# Arabic Sign Language Recognition System

---

## 📋 نظرة عامة على المشروع

هذا المشروع نظام للتعرف على لغة الإشارة العربية الموحدة في الوقت الفعلي (Real-time) باستخدام الكاميرا. يقوم النظام بتحليل حركات اليد والتعرف على الحروف الأبجدية العربية الـ28 وتجميعها في كلمات.

### الهدف
بناء نظام يساعد في التواصل بين مجتمع الصم والبكم والمجتمع العام عن طريق ترجمة إشارات اليد إلى نص عربي مقروء.

---

## 🏗️ معمارية النظام

```
الكاميرا (OpenCV) → MediaPipe (21 نقطة) → موديل TFLite (MLP) → عرض النتيجة (نص عربي)
```

### شرح المراحل
1. **الكاميرا**: تلتقط الفيديو في الوقت الفعلي عبر OpenCV
2. **MediaPipe**: يستخرج 21 نقطة من اليد (landmarks) ويحوّلها لإحداثيات
3. **الموديل**: شبكة عصبية (MLP) تصنّف الإيماءة وتحدد الحرف
4. **العرض**: يظهر الحرف والكلمة بالنص العربي على الشاشة

---

## 📁 هيكل المشروع

```
arabic-sign-language/
│
├── collect_data.py          ← أداة جمع بيانات التدريب
├── train_model.py           ← سكريبت تدريب الموديل
├── app_arabic.py            ← التطبيق النهائي (Real-time)
├── Cairo-Regular.ttf        ← فونت عربي للعرض
│
├── arabic_data/             ← بيانات التدريب (تتولد تلقائياً)
│   ├── arabic_keypoints.csv   ← إحداثيات نقاط اليد لكل عينة
│   └── arabic_labels.csv      ← أسماء الحروف وأرقامها
│
└── arabic_model/            ← الموديل المدرَّب (يتولد بعد التدريب)
    ├── arabic_sign_model.h5       ← الموديل بصيغة Keras
    ├── arabic_sign_model.tflite   ← الموديل المضغوط للنشر
    ├── best_model.h5              ← أفضل نسخة أثناء التدريب
    ├── confusion_matrix.png       ← مصفوفة الأخطاء
    └── training_curves.png        ← منحنيات الدقة والخسارة
```

---

## 📚 المكتبات المستخدمة وشرح دورها

### 1. MediaPipe (mediapipe==0.10.11)
**الدور**: المكتبة الأساسية لاكتشاف اليد واستخراج نقاطها.

تقوم باكتشاف اليد في الصورة وإرجاع **21 نقطة** (landmarks) تمثل مفاصل الأصابع وراحة اليد:

```
النقاط: المعصم (0) + إبهام (1-4) + سبابة (5-8) + وسطى (9-12) + بنصر (13-16) + خنصر (17-20)
```

```python
from mediapipe.python.solutions import hands as mp_hands_solutions
hands = mp_hands_solutions.Hands(
    max_num_hands=1,               # يد واحدة فقط
    min_detection_confidence=0.7,  # دقة اكتشاف 70%
    min_tracking_confidence=0.5    # دقة تتبع 50%
)
```

---

### 2. TensorFlow / Keras (tensorflow==2.10.1)
**الدور**: بناء وتدريب الشبكة العصبية، وتحويلها لصيغة TFLite.

**معمارية الموديل (MLP - Multi-Layer Perceptron)**:
```
Input (42)  →  Dense(128) + BatchNorm + Dropout(0.3)
            →  Dense(256) + BatchNorm + Dropout(0.3)
            →  Dense(128) + BatchNorm + Dropout(0.2)
            →  Dense(64)  + Dropout(0.2)
            →  Output(28) Softmax
```

- **Input**: 42 قيمة = 21 نقطة × (x, y)
- **BatchNormalization**: يسرّع التدريب ويستقر النتائج
- **Dropout**: يمنع الـ Overfitting (حفظ البيانات بدلاً من تعلمها)
- **Softmax**: يحوّل النتائج لاحتماليات لكل حرف (مجموعها = 1)

---

### 3. OpenCV (opencv-contrib-python)
**الدور**: التعامل مع الكاميرا ورسم العناصر على الشاشة.

```python
cap = cv.VideoCapture(0)                           # فتح الكاميرا
image = cv.flip(image, 1)                          # عكس الصورة (مرآة)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)       # تحويل الألوان لـ MediaPipe
cv.rectangle(image, pt1, pt2, color, thickness)   # رسم مستطيل حول اليد
cv.imshow('window', image)                         # عرض الصورة
```

---

### 4. NumPy (numpy==1.26.4)
**الدور**: العمليات الحسابية على المصفوفات.

```python
# تطبيع إحداثيات النقاط
max_val = max(abs(v) for v in rel_landmarks)
normalized = [v / max_val for v in rel_landmarks]

# تحضير البيانات للموديل
input_data = np.array([landmarks], dtype=np.float32)
```

---

### 5. Pillow + arabic-reshaper + python-bidi
**الدور**: عرض النص العربي بشكل صحيح على الصورة.

**المشكلة**: OpenCV لا يدعم النص العربي مباشرة لأن:
- العربية تُكتب من اليمين لليسار (RTL)
- الحروف تتشكل حسب موقعها في الكلمة

**الحل بثلاث خطوات**:
```python
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image

# 1. arabic_reshaper: يصحح شكل الحروف (مثلاً: ب تصبح بـ أو ـبـ حسب الموقع)
reshaped = arabic_reshaper.reshape("مرحبا")

# 2. python-bidi: يعكس اتجاه الكتابة من اليسار لليمين
bidi_text = get_display(reshaped)

# 3. Pillow: يرسم النص على الصورة بفونت Cairo
img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype('Cairo-Regular.ttf', 40)
draw.text(position, bidi_text, font=font, fill=color)
```

---

### 6. scikit-learn
**الدور**: تقسيم البيانات وتقييم الموديل.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# تقسيم 80% تدريب، 20% اختبار مع الحفاظ على توزيع الفئات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

### 7. Matplotlib + Seaborn
**الدور**: رسم نتائج التدريب وتحليل الأداء.

```python
# Confusion Matrix: تُظهر الحروف المتشابهة والأخطاء
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)

# Training Curves: منحنيات الدقة والخسارة عبر الـ Epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

---

## 🔄 شرح تدفق البيانات

### مرحلة جمع البيانات (collect_data.py)

**خطوات المعالجة**:
1. قراءة إطار من الكاميرا
2. MediaPipe يكتشف اليد ويرجع 21 نقطة (x, y لكل نقطة)
3. **تحويل لإحداثيات نسبية**: طرح إحداثيات المعصم من كل نقطة
4. **تطبيع**: قسمة على أكبر قيمة مطلقة (النتيجة بين -1 و 1)
5. حفظ الـ 42 قيمة مع رقم الحرف في ملف CSV

```python
def extract_landmarks(hand_landmarks, image_shape):
    # استخراج الإحداثيات الخام
    for lm in hand_landmarks.landmark:
        x = int(lm.x * width)
        y = int(lm.y * height)

    # تحويل لإحداثيات نسبية (مستقلة عن موقع اليد في الصورة)
    base_x, base_y = landmark_list[0]  # المعصم كنقطة مرجعية
    rel = [x - base_x, y - base_y for x, y in landmark_list]

    # تطبيع (مستقل عن حجم اليد)
    max_val = max(abs(v) for v in rel)
    normalized = [v / max_val for v in rel]
```

**لماذا التطبيع مهم؟**
- يجعل الموديل يتعرف على الإيماءة بغض النظر عن **موقع اليد** في الصورة
- يجعله مستقلاً عن **حجم اليد** (يد كبيرة أو صغيرة)
- يحسن دقة التعرف بشكل كبير

---

### مرحلة التدريب (train_model.py)

**الـ Callbacks المستخدمة**:

| Callback | الدور |
|----------|-------|
| EarlyStopping | يوقف التدريب إذا توقف التحسن (patience=20 epoch) |
| ReduceLROnPlateau | يقلل معدل التعلم عند التوقف (factor=0.5) |
| ModelCheckpoint | يحفظ أفضل نسخة من الموديل تلقائياً |

---

### مرحلة التطبيق (app_arabic.py)

**نظام تأكيد الحرف (History Buffer)**:
```python
HISTORY_LENGTH = 25    # عدد الإطارات المطلوبة
AGREEMENT_RATIO = 0.85 # 85% من الإطارات يجب أن تتفق

prediction_history = deque(maxlen=25)

# يسجل الحرف فقط لو 85% من آخر 25 إطار متفقين
most_common = Counter(prediction_history).most_common(1)[0]
if most_common[1] >= 25 * 0.85:
    confirmed_letter = labels_dict[most_common[0]]
```

**لماذا هذا النظام؟**
- يمنع تسجيل حروف خاطئة بسبب حركة عابرة
- يعطي وقتاً كافياً للمستخدم لثبيت الإيماءة
- الشريط الأخضر على الشاشة يُظهر مدى اكتمال التأكيد

**نظام الـ Cooldown**:
```python
COOLDOWN_FRAMES = 40  # 40 إطار تقريباً 1.3 ثانية

# بعد تسجيل حرف، انتظر قبل تسجيل التالي
if confirmed_letter:
    current_word.append(confirmed_letter)
    cooldown = COOLDOWN_FRAMES
```

---

## 🗂️ تنسيق البيانات

### ملف arabic_keypoints.csv
```
label, x0, y0, x1, y1, ..., x20, y20
0, 0.0, 0.0, 0.15, -0.08, ...   ← حرف أ
1, 0.0, 0.0, 0.12, -0.11, ...   ← حرف ب
```
- **العمود الأول**: رقم الحرف (0-28)
- **الأعمدة التالية**: 42 قيمة (إحداثيات 21 نقطة بعد التطبيع)

### ملف arabic_labels.csv
```
0, أ
1, ب
2, ت
...
28, لا
```

---

## 🔤 الحروف المدعومة ومفاتيحها

| مفتاح | حرف |
|-------|------|
|   1   |  أ   |
|   2   |  ب   |
|   3   |  ت   |
|   4   |  ث   |
|   5   |  ج   |
|   6   |  ح   |
|   7   |  خ   |
|   8   |  د   |
|   9   |  ذ   |
|   0   |  ر   |
|   q   |  ز   |
|   w   |  س   |
|   e   |  ش   |
|   r   |  ص   |
|   t   |  ض   |
|   y   |  ط   |
|   u   |  ظ   |
|   i   |  ع   |
|   o   |  غ   |
|   p   |  ف   |
|   a   |  ق   |
|   s   |  ك   |
|   d   |  ل   |
|   f   |  م   |
|   g   |  ن   |
|   h   |  ه   |
|   j   |  و   |
|   k   |  ي   |
|   l   |  لا   |
---

## ⚙️ متطلبات التشغيل

### البيئة
```
Python 3.10
Windows 10/11
كاميرا ويب
```

### تثبيت المكتبات
```bash
python -m venv
venv310\Scripts\activate

pip install mediapipe==0.10.11
pip install tensorflow==2.10.1
pip install numpy==1.26.4
pip install opencv-contrib-python
pip install Pillow arabic-reshaper python-bidi
pip install scikit-learn matplotlib seaborn
pip install python-dotenv
pip install flask
pip install flask-limiter
```



# Web
flask==3.0.2

# Core AI stack (لازم الإصدارات دي بالظبط)
tensorflow==2.10.1
numpy==1.26.4
mediapipe==0.10.11

# Computer Vision (نسخة متوافقة مع numpy 1.x)
opencv-contrib-python==4.8.0.76

# Image & Arabic text
Pillow==10.3.0
arabic-reshaper==3.0.0
python-bidi==0.4.2

# ML & Visualization
scikit-learn==1.3.2
matplotlib==3.7.5
seaborn==0.13.2
---





## 🚀 تشغيل المشروع

### 1. جمع البيانات
```bash
python collect_data.py
```
- اضغط مفتاح الحرف (مثال: 1 لـ أ)
- ثبّت يدك بشكل الإشارة
- اضغط SPACE لتسجيل عينة
- الهدف: **200 عينة لكل حرف على الأقل**

### 2. تدريب الموديل
```bash
python train_model.py
```
- ينتهي تلقائياً عند توقف التحسن
- يحفظ الموديل في مجلد arabic_model/

### 3. التطبيق النهائي
```bash
python app_arabic.py
```

| زر | الوظيفة |
|----|---------|
| SPACE | إضافة مسافة بين الكلمات |
| BACKSPACE | حذف آخر حرف |
| C | مسح الكلمة كاملاً |
| ESC | إغلاق التطبيق |

---

## 📊 نتائج التدريب (10 حروف - المرحلة الأولى)

| الحرف | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| أ | 0.95 | 1.00 | 0.98 |
| ب | 0.87 | 0.98 | 0.92 |
| ت | 0.98 | 1.00 | 0.99 |
| ث | 1.00 | 0.98 | 0.99 |
| ج | 1.00 | 0.85 | 0.92 |
| ح | 0.85 | 1.00 | 0.92 |
| خ | 1.00 | 0.97 | 0.99 |
| د | 0.95 | 0.53 | 0.68 |
| ذ | 0.95 | 0.95 | 0.95 |
| ر | 0.67 | 0.85 | 0.75 |
| **المتوسط** | **0.92** | **0.91** | **0.91** |

**الدقة الكلية: 91.2%** على 2044 عينة

### تحليل الأخطاء
- **د و ر**: أكثر الحروف تشابهاً في الإشارة — يحتاجان بيانات إضافية (300+ عينة)
- باقي الحروف تجاوزت 92% وهو مستوى ممتاز لمشروع تخرج

---

## 🔧 تحسينات مقترحة

### تحسين الدقة
- زيادة عينات التدريب لـ **300-500 عينة** لكل حرف
- جمع بيانات من **أشخاص مختلفين** (جنس، حجم يد، لون بشرة)
- **Data Augmentation**: إضافة تشويش طفيف على الإحداثيات أثناء التدريب

### تحسين الموديل
- تجربة **CNN** بدلاً من MLP للحصول على دقة أعلى
- إضافة **LSTM** للتعرف على الكلمات كاملة وليس حرفاً بحرف

### تحسين التطبيق
- إضافة **Text-to-Speech** لتحويل النص لصوت عربي
- دعم **كلتا اليدين** معاً
- نشر التطبيق على **الموبايل** باستخدام TFLite

---

## 📖 المراجع

- [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- المشروع الأصلي: [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
