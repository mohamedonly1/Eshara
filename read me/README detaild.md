# مشروع إشارة - التعرف على لغة الإشارة العربية الموحدة
# Ishara - Arabic Sign Language Recognition System

---

## 📋 نظرة عامة على المشروع

هذا المشروع هو منظومة متكاملة للتعرف على لغة الإشارة العربية الموحدة في الوقت الفعلي (Real-time). تطور المشروع من أداة حاسوبية بسيطة ليصبح **نظاماً ثلاثي الأبعاد (3D System)** للترجمة العكسية، و**تطبيق هاتف ذكي (Android App)** للترجمة الفورية عبر كاميرا الهاتف.

### الهدف
بناء جسر تواصل متكامل بين مجتمع الصم والبكم والمجتمع العام عبر ثلاث منصات رئيسية تتيح الترجمة الفورية بالاتجاهين (إشارة ← نص، ونص/صوت ← إشارة).

---

## 🏗️ معمارية النظام الشاملة

```
المنظومة تتكون من 3 أقسام متكاملة:

1. الذكاء الاصطناعي (AI Pipeline)
   OpenCV → MediaPipe Landmarks → Data Normalization → TFLite MLP Model

2. تطبيق الأندرويد (Android Mobile App)
   CameraX (Frames) → Image Rotation/Correction → MediaPipe Tasks Vision → TFLite Inference → Canvas Overlay

3. منصة الويب والـ 3D (Web Dashboard & Inverse Kinematics)
   Speech/Text Input → Flask Backend → Three.js (GLTF Model) → Quaternion/Euler IK Math → 3D Hand Animation
```

---

## 📱 القسم الأول: تطبيق الأندرويد للترجمة الفورية

تطبيق هواتف ذكية مبني بلغة **Kotlin** يعمل كعدسة مترجمة في الوقت الفعلي باستخدام الكاميرا.

### 📚 المكتبات وتقنيات الأندرويد المستخدمة

#### 1. CameraX
**الدور**: إدارة الكاميرا والتقاط الإطارات (Frames) بكفاءة عالية بدون إرهاق الذاكرة.
```kotlin
val cameraProvider = cameraProviderFuture.get()
val imageAnalysis = ImageAnalysis.Builder()
    .setTargetResolution(Size(480, 640))
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build()
```

#### 2. MediaPipe Tasks Vision (الإصدار الأحدث للأندرويد)
**الدور**: اكتشاف اليد واستخراج 21 نقطة مفصلية بسرعة فائقة تناسب الهواتف.
```kotlin
val baseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
val options = HandLandmarker.HandLandmarkerOptions.builder()
    .setBaseOptions(baseOptions)
    .setRunningMode(RunningMode.IMAGE)
    .setNumHands(1)
    .build()
```

#### 3. TensorFlow Lite (TFLite)
**الدور**: تشغيل المودل المدرب (MLP) محلياً (On-Device) لترجمة الإحداثيات إلى حروف دون الحاجة لإنترنت.

### ⚙️ تحديات تقنية معقدة تم حلها في الأندرويد:

**مشكلة الدوران ونسبة الأبعاد (Rotation & Aspect Ratio)**
تم تدريب المودل على صور كاميرا الويب (شاشة أفقية 640x480). عند تشغيل التطبيق على الهاتف (شاشة عمودية)، اختلفت الأبعاد وانقلبت الزوايا، مما أدى لانهيار الدقة تماماً.

**الحل الرياضي الجذري (Foolproof Mapping):**
```kotlin
// 1. تدوير الصورة فعلياً في الذاكرة لتصبح معتدلة تماماً (Upright) قبل إرسالها للذكاء الاصطناعي
if (bitmap.width > bitmap.height) {
    val matrix = Matrix()
    matrix.postRotate(rotationDegrees.toFloat())
    bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

// 2. تطبيع الإحداثيات لمحاكاة شاشة التدريب (Aspect Ratio Scaling)
val scaleX = bitmapWidth / 640f
val scaleY = bitmapHeight / 480f
val points = landmarks.map { Pair(it.x() * scaleX, it.y() * scaleY) }
```
هذا الحل ضمن أن المودل يستقبل الإحداثيات الدقيقة التي تدرب عليها، لتعود الدقة لنسبة تفوق الـ 90%.

---

## 🌐 القسم الثاني: منصة الويب والأنيميشن 3D

تطبيق ويب يتيح ترجمة الصوت أو النص إلى لغة إشارة، وعرضها باستخدام مجسم يد ثلاثي الأبعاد.

### 📚 المكتبات وتقنيات الويب المستخدمة

#### 1. Flask & Flask-Limiter (Python)
**الدور**: خادم الويب الأساسي (Backend) لإدارة الجلسات ولوحة التحكم.
```python
@app.route('/update_sign', methods=['POST'])
def update_sign():
    data = request.json
    # تحديث وتخزين الزوايا الدقيقة لكل مفصل في قاعدة البيانات
```

#### 2. Three.js & GLTF Loader
**الدور**: تشغيل ومعالجة مجسم اليد ثلاثي الأبعاد (Avatar) في المتصفح.

#### 3. الحركيات العكسية (Inverse Kinematics - IK) ورياضيات العظام
**المشكلة**: عند تحريك أصابع الـ 3D Model بناءً على زوايا ثابتة، كانت العظام تتشوه (Deformation) وتنكمش (Squishing).
**الحل**: تم بناء نظام يحاكي تشريح اليد برمجياً، يعتمد على تحويل الزوايا إلى `Quaternions` لتجنب مشكلة الـ (Gimbal Lock) وتصفير زوايا المحاور بذكاء عند الانتقال بين الإشارات.
```javascript
// الحساب الديناميكي للدوران بدون تشوه
const axis = new THREE.Vector3(1, 0, 0); // محور الثني
const quaternion = new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(angle));
bone.quaternion.copy(quaternion); // استبدال الـ Euler بالـ Quaternion لضمان استقرار العظم
```

#### 4. Web Speech API
**الدور**: تحويل الكلام الصوتي المسموع (عربي) إلى نص مقروء يترجمه المجسم لإشارات.

---

## 🧠 القسم الثالث: نظام الذكاء الاصطناعي وتدريب الموديل (AI Pipeline)

النواة الأساسية التي تم بناء المشروع عليها للتعرف على اليد.

### 📚 مكتبات الـ AI المستخدمة وشرح دورها

#### 1. TensorFlow / Keras (tensorflow==2.10.1)
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

#### 2. MediaPipe (mediapipe==0.10.11)
تقوم باكتشاف اليد في الصورة وإرجاع **21 نقطة** (landmarks) تمثل مفاصل الأصابع وراحة اليد.

#### 3. OpenCV (opencv-contrib-python)
**الدور**: التعامل مع الكاميرا ورسم العناصر على الشاشة أثناء مرحلة جمع البيانات للتدريب.

#### 4. NumPy (numpy==1.26.4)
**الدور**: العمليات الحسابية وتطبيع الإحداثيات (Normalization) لجعل المودل مستقلاً عن موقع وحجم اليد في الصورة.

```python
# تطبيع إحداثيات النقاط
max_val = max(abs(v) for v in rel_landmarks)
normalized = [v / max_val for v in rel_landmarks]
```

#### 5. Pillow + arabic-reshaper + python-bidi
**الدور**: عرض النص العربي بشكل صحيح داخل نافذة OpenCV لأن المكتبة لا تدعم رسم الحروف العربية المتصلة من اليمين لليسار.

#### 6. scikit-learn
**الدور**: تقسيم البيانات (80% تدريب - 20% اختبار) وتقييم الموديل وإنشاء `confusion_matrix`.

---

## 🔄 شرح تدفق بيانات الذكاء الاصطناعي (Data Flow)

### 1. مرحلة جمع البيانات (collect_data.py)
**خطوات المعالجة**:
1. قراءة إطار من الكاميرا
2. MediaPipe يكتشف اليد ويرجع 21 نقطة (x, y لكل نقطة)
3. **تحويل لإحداثيات نسبية**: طرح إحداثيات المعصم من كل نقطة
4. **تطبيع**: قسمة على أكبر قيمة مطلقة (النتيجة بين -1 و 1)
5. حفظ الـ 42 قيمة مع رقم الحرف في ملف CSV

### 2. مرحلة التدريب (train_model.py)
تستخدم **Callbacks** ذكية لتحسين التدريب:
- `EarlyStopping`: يوقف التدريب إذا توقف التحسن (patience=20 epoch).
- `ReduceLROnPlateau`: يقلل معدل التعلم عند التوقف.
- `ModelCheckpoint`: يحفظ أفضل نسخة تلقائياً.

### 3. مرحلة التطبيق السطحي للكمبيوتر (app_arabic.py)
**نظام تأكيد الحرف (History Buffer)**:
يستخدم قائمة بحجم 25 إطاراً، ولا يسجل الحرف إلا إذا تكرر بنسبة 85% لضمان عدم الطباعة العشوائية مع حركة اليد العابرة.

---

## 🗂️ تنسيق البيانات

### ملف arabic_keypoints.csv
```
label, x0, y0, x1, y1, ..., x20, y20
0, 0.0, 0.0, 0.15, -0.08, ...   ← حرف أ
1, 0.0, 0.0, 0.12, -0.11, ...   ← حرف ب
```

### ملف arabic_labels.csv
```
0, أ
1, ب
...
28, لا
```

---

## 🔤 الحروف المدعومة ومفاتيح التدريب

| مفتاح | حرف | | مفتاح | حرف |
|-------|------|-|-------|------|
|   1   |  أ   | |   y   |  ط   |
|   2   |  ب   | |   u   |  ظ   |
|   3   |  ت   | |   i   |  ع   |
|   4   |  ث   | |   o   |  غ   |
|   5   |  ج   | |   p   |  ف   |
|   6   |  ح   | |   a   |  ق   |
|   7   |  خ   | |   s   |  ك   |
|   8   |  د   | |   d   |  ل   |
|   9   |  ذ   | |   f   |  م   |
|   0   |  ر   | |   g   |  ن   |
|   q   |  ز   | |   h   |  ه   |
|   w   |  س   | |   j   |  و   |
|   e   |  ش   | |   k   |  ي   |
|   r   |  ص   | |   l   |  لا  |
|   t   |  ض   | |       |      |

---

## ⚙️ متطلبات التشغيل الأساسية للبيئة

```bash
# إنشاء البيئة الافتراضية
python -m venv venv310
venv310\Scripts\activate

# تنزيل مكتبات الويب والذكاء الاصطناعي الأساسية
pip install mediapipe==0.10.11 tensorflow==2.10.1 numpy==1.26.4
pip install opencv-contrib-python Pillow arabic-reshaper python-bidi
pip install scikit-learn matplotlib seaborn
pip install flask flask-limiter python-dotenv
```

---

## 🚀 كيفية تشغيل المنصات المختلفة

### 1. تشغيل خادم الويب والمجسم ثلاثي الأبعاد
```bash
python server.py
# سيتم فتح لوحة التحكم على http://127.0.0.1:5000
```

### 2. تشغيل تطبيق الأندرويد
1. قم بفتح مجلد `Ishara/` باستخدام برنامج **Android Studio**.
2. انتظر حتى يقوم البرنامج بتحميل وتزامن ملفات الـ `Gradle`.
3. قم بتوصيل هاتفك أو تشغيل المحاكي واضغط على زر التثبيت والتشغيل (Run).

### 3. تدريب النموذج أو إضافة إشارات جديدة (اختياري)
1. جمع البيانات: `python collect_data.py`
2. تدريب المودل: `python train_model.py`
3. سيتم توليد مودل جديد، انسخ ملف `arabic_sign_model.tflite` من مجلد `arabic_model/` وضعه في المسار `Ishara/app/src/main/assets/` في تطبيق الأندرويد ليتم تحديثه.

---

## 📊 نتائج التدريب (المرحلة الأساسية)

| الحرف | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| أ | 0.95 | 1.00 | 0.98 |
| ب | 0.87 | 0.98 | 0.92 |
| ت | 0.98 | 1.00 | 0.99 |
| خ | 1.00 | 0.97 | 0.99 |
| ذ | 0.95 | 0.95 | 0.95 |
| **المتوسط الكلي** | **0.92** | **0.91** | **0.91** |

**الدقة الكلية: 91.2%** 

### تحليل الأخطاء
- الحروف المتقاربة جداً حركياً (مثل الدال والراء) حصلت على تقييم أضعف وتحتاج إلى عينات إضافية وزوايا تصوير متنوعة.
- أغلب الحروف الباقية تجاوزت الدقة فيها 95% مما يجعل النظام ممتازاً للعمل الحي.

---

## 🔧 تحسينات وتوسعات مقترحة للمشروع
- **AI**: إضافة نموذج `LSTM` لتحليل الحركة عبر الزمن، بحيث يفهم الكلمات كاملة (حركة متصلة) بدلاً من الحروف المتقطعة.
- **Data Augmentation**: تطبيق قلب وتشويه بسيط للبيانات أثناء التدريب برمجياً للحصول على موديل أصلب ومقاوم لأخطاء التصوير.
- **Android**: إضافة مزامنة سحابية (Cloud Sync) بين تطبيق الأندرويد ومنصة الويب لجلب تحديثات قاموس الـ 3D للإشارات مباشرة للتطبيق.

---

## 📖 المراجع العلمية والتقنية
- [MediaPipe Hand Landmarker Guide](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [TensorFlow Lite for Android](https://www.tensorflow.org/lite/android)
- [Three.js Quaternions and Rotations](https://threejs.org/docs/#api/en/math/Quaternion)
- المشروع الأساسي مبني على إلهام من أبحاث `Kazuhito00` لتقنيات تتبع اليدين.
