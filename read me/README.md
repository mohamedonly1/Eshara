# مشروع لغة الإشارة العربية الموحدة
## Arabic Sign Language Recognition

---

## هيكل المشروع
```
arabic-sign-language/
│
├── collect_data.py        ← جمع البيانات
├── train_model.py         ← تدريب الموديل
├── app_arabic.py          ← التطبيق النهائي
├── Cairo-Regular.ttf      ← فونت عربي (حمّله)
│
├── arabic_data/           ← 
│   ├── arabic_keypoints.csv
│   └── arabic_labels.csv
│
└── arabic_model/          ← 
    ├── arabic_sign_model.h5
    ├── arabic_sign_model.tflite
    ├── confusion_matrix.png
    └── training_curves.png
```

---

## خطوات التشغيل

### 1. تثبيت المكتبات
```bash
venv310\Scripts\activate
pip install mediapipe==0.10.11
pip install numpy==1.26.4
pip install tensorflow==2.10.1
pip install opencv-contrib-python
pip install Pillow arabic-reshaper python-bidi
pip install scikit-learn matplotlib seaborn
```

### 3. جمع البيانات
```bash
python collect_data.py
```
**الهدف:** 200 عينة لكل حرف (5800 عينة إجمالاً)

**مفاتيح الحروف:**| مفتاح | حرف |
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
**SPACE** = تسجيل إيماءة

### 4. تدريب الموديل
```bash
python train_model.py
```

### 5. تشغيل التطبيق
```bash
python app_arabic.py
```

---

## نصائح لجمع بيانات جيدة
- ✅ إضاءة كويسة وثابتة
- ✅ خلفية سادة (أبيض أو رمادي أفضل)
- ✅ يد واحدة في الإطار
- ✅ جرّب زوايا مختلفة قليلاً لكل حرف
- ✅ لو معاك ناس تانية، اجمع منهم كمان (بيحسّن الدقة)
- ❌ متسجلش في إضاءة ضعيفة أو ضد الضوء

---