# مشروع إشارة - التعرف على لغة الإشارة العربية الموحدة (Detailed Documentation)

---

## 🌟 نظرة عامة شاملة

"إشارة" هو مشروع تخرج متكامل ومتطور للتعرف الفوري على لغة الإشارة العربية. تم بناء النظام ليعمل عبر منصات متعددة لتقديم أقصى فائدة عملية لمجتمع الصم والبكم.

تتكون المنظومة من 3 مكونات برمجية رئيسية:
1. **نظام الذكاء الاصطناعي (AI & Machine Learning)**: مسؤول عن معالجة البيانات، بناء الشبكة العصبية، وتدريب نماذج التعرف.
2. **منصة الويب التفاعلية (Web Platform & 3D IK System)**: خادم Flask مع واجهة تفاعلية تضم قاموساً للإشارات ومجسماً ثلاثي الأبعاد (Avatar) يترجم النصوص والصوتيات إلى إيماءات باستخدام تقنيات Inverse Kinematics.
3. **تطبيق الأندرويد (Mobile Application)**: تطبيق مخصص يعمل في الوقت الفعلي (Real-time) باستخدام كاميرا الهاتف للترجمة الفورية.

---

## 📱 1. تطبيق الأندرويد المتقدم (Ishara Android App)

تم تطوير تطبيق الهاتف بلغة Kotlin باستخدام أحدث تقنيات نظام الأندرويد لضمان كفاءة عالية (High Performance) أثناء المعالجة الحية للفيديو.

### التقنيات المستخدمة:
- **CameraX**: لإدارة دورة حياة الكاميرا بسلاسة والتقاط الإطارات بدقة.
- **MediaPipe Tasks Vision**: الإصدار الأحدث من مكتبة جوجل (HandLandmarker) لاكتشاف وتتبع مفاصل اليد بدقة وسرعة على الأجهزة المحمولة.
- **TensorFlow Lite (TFLite)**: لتشغيل النموذج المدرب محلياً على الجهاز بدون الحاجة للإنترنت.

### 🔧 التحديات التقنية التي تم حلها (Technical Breakthroughs):

#### أ. مشكلة توافق الأبعاد وتدوير الكاميرا (Aspect Ratio & Rotation Mapping)
- **المشكلة**: تم تدريب موديل TFLite باستخدام كاميرا ويب (Landscape) بنسبة أبعاد 640x480. عند تشغيل الموديل على كاميرا الهاتف المحمول (Portrait)، تختلف نسبة الأبعاد وزوايا الدوران (Rotation Degrees)، مما أدى لانهيار دقة الاستنتاج وتشوه رسم الخطوط (Overlay) فوق اليد.
- **الحل (Foolproof Rotation)**:
  - تمت برمجة الكود ليقوم بتدوير الصورة المستلمة من `ImageProxy` بشكل آلي ودقيق (باستخدام `Matrix.postRotate`) بناءً على دوران مستشعر الهاتف، وذلك لضمان تسليم صورة معتدلة (Upright) لـ MediaPipe بشكل دائم.
  - تم عمل **تطبيع إحداثيات رياضي (Mathematical Coordinate Normalization)** حيث تُضغط الإحداثيات لمحاكاة نسبة أبعاد كاميرا الويب (ScaleX = width/640, ScaleY = height/480).
  - تم إصلاح نظام الرسم (Overlay Canvas) ليدعم الكاميرات الأمامية والخلفية (Front/Back) مع معالجة تأثير المرآة (Mirror Effect).

---

## 🌐 2. منصة الويب والأنيميشن ثلاثي الأبعاد (Web Dashboard & 3D System)

### التقنيات المستخدمة:
- **Backend**: Python, Flask, Flask-Limiter.
- **Frontend**: HTML, CSS, JavaScript, Three.js.
- **Speech-to-Text**: Web Speech API للترجمة الصوتية.

### 🔧 التحديات التقنية التي تم حلها:

#### أ. نظام الحركيات العكسية (Inverse Kinematics - IK) للمجسم الثلاثي
- تم الانتقال من نظام زوايا ثابت (Pre-calculated Rotations) إلى نظام ديناميكي يقوم بحساب مواقع العظام بناءً على بيانات MediaPipe في متصفح الويب.
- **منع تشوه العظام (Skeletal Deformation Fix)**: تم حل مشكلة تشوه المجسم عند الانتقال بين الإشارات من خلال ضمان إعادة تهيئة العظام وحساب محاور الدوران الدقيقة باستخدام خوارزمياترياضية متقدمة داخل بيئة `Three.js` (مزامنة Quaternion و Euler Angles).
- **لوحة التحكم (Dashboard)**: نظام متكامل لإدارة الإيماءات وإجراء تعديلات دقيقة (Calibration) لمحاور X, Y, Z وحفظها برمجياً لاستخدامها في النظام المباشر، مع ميزة التحقق البصري وحفظ التاريخ (Timestamp).

---

## 🧠 3. نظام الذكاء الاصطناعي (AI Pipeline)

### معمارية تصنيف الإشارات (MLP Model)
تم استخدام شبكة عصبية عميقة (Multi-Layer Perceptron) تتميز بسرعتها وخفتها.

```
Input (42 Features)  →  Dense(128) + BatchNorm + Dropout(0.3)
                     →  Dense(256) + BatchNorm + Dropout(0.3)
                     →  Dense(128) + BatchNorm + Dropout(0.2)
                     →  Dense(64)  + Dropout(0.2)
                     →  Output(28 Classes) Softmax
```

### معالجة البيانات (Data Preprocessing):
- يتم التقاط 21 نقطة من MediaPipe، وتُحول إلى إحداثيات نسبية (Relative Coordinates) نسبةً إلى موقع مفصل المعصم (Wrist) لجعل التعرف مستقلاً عن موقع اليد في الشاشة.
- يتم تطبيق عملية "تطبيع" (Normalization) بالقسمة على أكبر قيمة مطلقة لجعل التعرف مستقلاً عن المسافة بين اليد والكاميرا.

---

## 🚀 كيفية التثبيت والتشغيل (Quick Start)

### متطلبات النظام الأساسية:
- Python 3.10
- Android Studio 

### 1. إعداد بيئة بايثون (Web & AI)
```bash
python -m venv venv310
venv310\Scripts\activate

pip install mediapipe==0.10.11 tensorflow==2.10.1 numpy==1.26.4
pip install opencv-contrib-python Pillow arabic-reshaper python-bidi
pip install scikit-learn matplotlib seaborn
pip install flask flask-limiter python-dotenv
```

### 2. تشغيل خادم الويب (Dashboard)
```bash
python server.py
# أو app.py بناءً على ملف التشغيل الرئيسي
```

### 3. تدريب النموذج (إن رغبت في إضافة إشارات)
1. تشغيل `python collect_data.py` لجمع العينات عبر الكاميرا.
2. تشغيل `python train_model.py` لبدء التدريب وتوليد ملف `arabic_sign_model.tflite` الجديد.
3. يتم نسخ الملف الجديد إلى مجلد الأصول `assets` في مشروع الأندرويد.

### 4. تشغيل تطبيق الأندرويد
قم بفتح المجلد `Ishara` عبر Android Studio، ثم انتظر حتى يكتمل تحميل Gradle، وقم بتشغيله على هاتفك الموصول أو على المحاكي.

---

## 📊 دقة النظام (System Accuracy)
وصلت الدقة الكلية للنظام (Accuracy) إلى **93.8%** على مجموعة البيانات الكبيرة، حيث تعمل الطبقات الإضافية كـ BatchNormalization و Dropout على تجنب الحفظ الأعمى (Overfitting) واستقرار الأوزان.

---

## 🛠 التوسعات المستقبلية (Future Enhancements)
- إضافة دعم لاكتشاف واستنتاج كلتا اليدين (Two-handed signs).
- دعم معمارية Long Short-Term Memory (LSTM) لمعرفة تسلسل الكلمات بدلاً من الحروف المتقطعة.l mediapipe==0.10.11
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





## تشغيل المشروع

### 1. جمع البيانات
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

## نتائج التدريب (10 حروف - المرحلة الأولى)

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

## تحسينات مقترحة

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

## المراجع

- [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- المشروع الأصلي: [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)


























