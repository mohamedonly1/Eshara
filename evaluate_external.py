#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
مرحباً بك يا صديقي الذكي! في هذا الملف، سنتعلم كيف نقوم بـ "تقييم أداء النموذج على بيانات خارجية".

تخيل أنك قمت بتدريب نموذج ذكاء اصطناعي، وتريد اختباره على بيانات جديدة تماماً لم يراها من قبل (بيانات خارجية) 
للتأكد من أنه مستعد للعمل في العالم الحقيقي وليس فقط داخل معمل التدريب (وهذا ما نسميه بتجنب Overfitting).
هذا الملف (evaluate_external.py) يقوم بتحميل نموذج TFLite، وقراءة ملف اختبار خارجي، ثم تشغيل توقعات النموذج وحساب مقاييس النجاح 
مثل: الدقة (Accuracy)، والضبط (Precision)، والاستدعاء (Recall)، ومقياس F1-Score، ثم يرسم لنا رسمة بيانية تسمى "مصفوفة الالتباس" (Confusion Matrix).

دعنا نستكشف كيف يعمل الكود بالتفصيل:
"""

import os         # للتعامل مع مجلدات وملفات النظام
import csv        # لفتح وقراءة ملفات الجداول CSV
import json       # لحفظ تقرير التقييم النهائي بصيغة JSON المرنة
import argparse   # مكتبة رائعة تسمح لنا بتشغيل هذا الملف من سطر الأوامر (Terminal) وتمرير إعدادات مختلفة له
from datetime import datetime  # لتسجيل وقت وتاريخ التقييم
import numpy as np             # للعمليات الرياضية على المصفوفات
import tensorflow as tf        # لتشغيل موديل TFLite المصغر الخاص بالهواتف والأجهزة الذكية
import matplotlib.pyplot as plt # لرسم الأشكال البيانية
import seaborn as sns          # مكتبة تعتمد على matplotlib لرسم مصفوفة الالتباس بشكل ملون وجذاب
# استيراد دوال قياس الأداء الشهيرة من مكتبة تعلم الآلة scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config  # ملف إعدادات المسارات الموحد للمشروع

# إعداد مسجل التقارير وحفظ مخرجات التقييم في ملف server.log
logger = config.get_file_logger('evaluation', config.SERVER_LOG)

def parse_args():
    """
    هذه الدالة تقوم بتعريف "معاملات سطر الأوامر" (Command Line Arguments).
    تسمح للمطور بكتابة شيء مثل: python evaluate_external.py --model path/to/model.tflite --dataset path/to/test.csv
    وإذا لم يحدد المطور هذه القيم، ستقوم الدالة باستخدام القيم الافتراضية المحددة في ملف config.py تلقائياً!
    """
    parser = argparse.ArgumentParser(description="Evaluate a TFLite model on an external test dataset.")
    
    # المعامل الأول: مسار ملف الموديل المراد تقييمه
    parser.add_argument(
        "--model", 
        type=str, 
        default=config.PRODUCTION_MODEL_TFLITE,
        help="Path to the TFLite model file to evaluate."
    )
    
    # المعامل الثاني: مسار ملف بيانات الاختبار الخارجي
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=config.TEST_CSV,
        help="Path to the CSV dataset (expects: tester_id, label, landmarks...)."
    )
    
    # المعامل الثالث: مجلد المخرجات حيث سنحفظ التقارير والرسومات البيانية
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=config.REPORTS_DIR,
        help="Directory where evaluation reports will be saved."
    )
    return parser.parse_args()

def load_labels(labels_path: str) -> dict:
    """
    تقوم بقراءة ملف التسميات (Labels CSV) لربط رقم التنبؤ بالحرف العربي المقابل له.
    المدخلات: مسار ملف التسميات.
    المخرجات: قاموس (Dictionary) يحتوي على الترتيب كـ مفتاح واسم الحرف كـ قيمة.
    """
    labels = {}
    if not os.path.exists(labels_path):
        logger.error("Labels CSV file not found: %s", labels_path)
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if row:
                labels[int(row[0])] = row[1]
    return labels

def load_test_dataset(dataset_path: str):
    """
    دالة ذكية جداً لقراءة بيانات الاختبار من ملف الـ CSV.
    تستطيع الدالة كشف التنسيق تلقائياً ودعمه:
    1. تنسيق المختبرين الجدد: (معرف المختبر، رقم الحرف الصحيح، 42 نقطة لليد) -> إجمالي 44 عموداً.
    2. تنسيق الأعضاء والمسجلين: (معرف المستخدم، رقم الحرف الصحيح، 42 نقطة لليد) -> إجمالي 44 عموداً.
    3. التنسيق الكلاسيكي القديم: (رقم الحرف الصحيح، 42 نقطة لليد) -> إجمالي 43 عموداً.
    """
    if not os.path.exists(dataset_path):
        logger.error("Dataset CSV file not found: %s", dataset_path)
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    X, y = [], []  # X للمدخلات (نقاط اليد) و y للإجابات الصحيحة (رقم الحرف)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            # التنسيق الجديد (44 عموداً): يحتوي على معرف المستخدم/المختبر في العمود الأول
            if len(row) == 44:  
                try:
                    label = int(row[1])  # رقم الحرف في العمود الثاني
                    landmarks = [float(v) for v in row[2:]]  # النقاط الـ 42 تبدأ من العمود الثالث
                    X.append(landmarks)
                    y.append(label)
                except ValueError:
                    continue
            # التنسيق القديم (43 عموداً): يبدأ برقم الحرف مباشرة في العمود الأول
            elif len(row) == 43:  
                try:
                    label = int(row[0])  # رقم الحرف في العمود الأول
                    landmarks = [float(v) for v in row[1:]]  # النقاط تبدأ من العمود الثاني
                    X.append(landmarks)
                    y.append(label)
                except ValueError:
                    continue
                    
    # نحول القوائم إلى مصفوفات Numpy ذات كفاءة رياضية عالية ونوع بيانات مناسب للذكاء الاصطناعي (float32 و int32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def evaluate_tflite_model(model_path: str, X: np.ndarray) -> np.ndarray:
    """
    دالة تشغيل التنبؤ على موديل TFLite.
    تقوم بتهيئة مفسر TFLite (Interpreter)، ثم تمرير كل عينة يد له، وتشغيل المعالجة،
    ثم قراءة النتائج ومعرفة الحرف الذي توقع له الموديل أعلى نسبة احتمال.
    """
    if not os.path.exists(model_path):
        logger.error("TFLite model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # تحميل ملف الموديل في مفسر Tensorflow Lite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  # حجز الذاكرة اللازمة للمصفوفات الداخلية للنموذج
    
    # جلب تفاصيل مدخلات ومخرجات الموديل (معرفة الأبعاد ونوع البيانات المطلوبة)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []  # قائمة لحفظ توقعات الموديل لكل عينة
    for sample in X:
        # تحويل عينة النقاط الفردية إلى مصفوفة وإضافة بعد إضافي (Batch Size = 1)
        inp = np.array([sample], dtype=np.float32)
        
        # وضع البيانات في بوابة المدخلات للموديل
        interpreter.set_tensor(input_details[0]['index'], inp)
        
        # استدعاء تشغيل النموذج للتخمين
        interpreter.invoke()
        
        # قراءة النواتج من بوابة المخرجات (تحتوي على احتمالات لكل حرف من الـ 29 حرفاً)
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # معرفة ترتيب الحرف صاحب الاحتمال الأكبر وحفظه في قائمة التوقعات
        y_pred.append(np.argmax(output))
    
    return np.array(y_pred, dtype=np.int32)

def main():
    # 1. جلب الإعدادات من سطر الأوامر والتأكد من وجود مجلد حفظ التقارير
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting evaluation of model: %s", args.model)
    logger.info("Using dataset: %s", args.dataset)

    try:
        # 2. تحميل أسماء الحروف العربية
        labels_dict = load_labels(config.LABELS_CSV)
        
        # 3. تحميل بيانات الاختبار
        X, y_true = load_test_dataset(args.dataset)
        
        if len(X) == 0:
            logger.error("No valid samples found in dataset.")
            return

        # 4. تشغيل الموديل والحصول على توقعاته
        y_pred = evaluate_tflite_model(args.model, X)

        # 5. حساب مقاييس الأداء الرياضية:
        # - الدقة (Accuracy): عدد التوقعات الصحيحة مقسوماً على إجمالي المحاولات.
        # - الضبط (Precision): مدى موثوقية الموديل عندما يتوقع حرفاً معيناً (تجنب الإشارات الكاذبة).
        # - الاستدعاء (Recall): قدرة الموديل على إيجاد وكشف كل الحالات الحقيقية لهذا الحرف.
        # - F1-Score: المتوسط المتناغم للضبط والاستدعاء، وهو أفضل مقياس لتوازن الأداء.
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        # طباعة المقاييس الكلية في شاشة الـ Terminal وحفظها في السجلات
        logger.info("Overall Accuracy: %0.2f%%", acc * 100)
        logger.info("Weighted Precision: %0.2f%%", precision * 100)
        logger.info("Weighted Recall: %0.2f%%", recall * 100)
        logger.info("Weighted F1-Score: %0.2f%%", f1 * 100)

        letter_names = [labels_dict.get(i, str(i)) for i in range(len(labels_dict))]
        
        # 6. كتابة تقرير نصي مفصل ومقروء بالأرقام لكل حرف وحفظه في ملف classification_report.txt
        cls_report_str = classification_report(y_true, y_pred, target_names=letter_names, zero_division=0)
        cls_report_path = os.path.join(args.output_dir, 'classification_report.txt')
        with open(cls_report_path, 'w', encoding='utf-8') as f:
            f.write(cls_report_str)
        logger.info("Saved classification report to: %s", cls_report_path)

        # 7. رسم وحفظ مصفوفة الالتباس (Confusion Matrix):
        # وهي جدول يوضح الحرف الحقيقي في المحور الرأسي، والحرف المتوقع في المحور الأفقي.
        # يسهل هذا الجدول معرفة أي الحروف يتلخبط الموديل بينها (مثلاً: يظن حرف دال أنه ذال).
        cm = confusion_matrix(y_true, y_pred)
        
        # الرسمة الأولى: مصفوفة الالتباس العادية (بالأرقام الصحيحة)
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
        plt.title('Confusion Matrix - External Test Set', fontsize=16)
        plt.ylabel('Actual (الحرف الحقيقي)')
        plt.xlabel('Predicted (الحرف المتوقع)')
        plt.tight_layout()
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # الرسمة الثانية: مصفوفة الالتباس المطبعة (بالنسب المئوية)
        # تفيد لو كان عدد العينات لكل حرف غير متساوٍ، لتظهر نسب الخطأ والصواب بوضوح
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=letter_names, yticklabels=letter_names)
        plt.title('Normalized Confusion Matrix - External Test Set', fontsize=16)
        plt.ylabel('Actual (الحرف الحقيقي)')
        plt.xlabel('Predicted (الحرف المتوقع)')
        plt.tight_layout()
        cm_norm_path = os.path.join(args.output_dir, 'normalized_confusion_matrix.png')
        plt.savefig(cm_norm_path, dpi=150)
        plt.close()

        logger.info("Saved confusion matrix plots to output directory.")

        # 8. حفظ تقرير تفصيلي بصيغة JSON:
        # يحتوي على التاريخ والوقت، اسم الموديل، واسم مجموعة البيانات، والنتائج الرقمية بالكامل 
        # ليسهل استهلاكها برمجياً في لوحة تحكم الإدارة (Admin Dashboard).
        cls_report_dict = classification_report(y_true, y_pred, target_names=letter_names, output_dict=True, zero_division=0)
        
        training_report = {
            'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_evaluated': args.model,
            'dataset_evaluated': args.dataset,
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': cls_report_dict
        }
        
        report_json_path = os.path.join(args.output_dir, 'training_report.json')
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(training_report, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON metrics report to: %s", report_json_path)

        logger.info("External evaluation complete.")

    except Exception as exc:
        # في حال حدوث أي خطأ مفاجئ، نقوم بتسجيله مع تتبع الخطوات (Traceback) لحل المشكلة بسهولة
        logger.error("Error during evaluation script execution: %s", exc, exc_info=True)

if __name__ == "__main__":
    main()
