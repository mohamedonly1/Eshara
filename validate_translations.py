# -*- coding: utf-8 -*-
"""
مرحباً بك مجدداً يا صديقي البطل! 

هذا الملف الرائع اسمه (validate_translations.py) ووظيفته الأساسية هي "التدقيق والتحقق من التراجم".
تخيل أن تطبيقنا يدعم لغات متعددة (العربية، الإنجليزية، الفرنسية)، ولعرض النصوص بلغات مختلفة نستخدم مفاتيح (keys) في قوالب الـ HTML.
أحياناً، قد يكتب المبرمج كوداً في الواجهات ويطلب ترجمة كلمة معينة، ولكنه ينسى إضافتها في ملفات الترجمة (ملفات الـ JSON الخاصة باللغات).
هذا الملف يقوم بالبحث التلقائي في جميع ملفات الواجهات (Templates) ويقارن المفاتيح المستخدمة مع ملفات الترجمة، ليخبرنا بالكلمات الناقصة أو غير المستخدمة.

دعنا نتتبع الكود البسيط والذكي:
"""

import sys      # للتعامل مع مدخلات ومخرجات نظام التشغيل
import os      # للتعامل مع المجلدات والملفات في الهارد ديسك
import re      # مكتبة التعبيرات النمطية (Regular Expressions) - أداة قوية جداً للبحث عن نصوص معينة داخل الملفات
import json    # للتعامل مع ملفات الترجمة التي تُحفظ بصيغة JSON

# نعدل ترميز نظام الإخراج لطباعة الحروف العربية بشكل سليم في ويندوز
sys.stdout.reconfigure(encoding='utf-8')

# نحدد مجلق الواجهات (HTML) ومجلد ملفات الترجمة (JSON)
TEMPLATES_DIR = r'templates'
TRANSLATIONS_DIR = r'translations'

def load_json(path):
    """
    دالة مساعدة لفتح ملف JSON وقراءة محتوياته بأمان.
    إذا لم يكن الملف موجوداً، ترجع قاموساً فارغاً {} لتجنب توقف البرنامج.
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def scan_templates():
    """
    هذه الدالة تقوم بفتح كل ملف HTML في المجلد والبحث عن أي استدعاء للترجمة.
    الترجمة في القوالب تستدعى عادةً هكذا: t('home_title') أو t("welcome_msg").
    """
    used_keys = set()  # نستخدم Set (مجموعة) بدلاً من List لأن الـ Set تمنع تكرار القيم تلقائياً!
    
    # هذا هو النمط (Pattern) الذي نبحث عنه:
    # نبحث عن حرف t يليه قوس مفتوح ( ثم علامة تنصيص فردية أو زوجية ثم نص يحتوي على حروف وأرقام ثم علامة التنصيص والقوس المغلق )
    pattern = re.compile(r'(?:^|[^a-zA-Z0-9_])t\(\s*[\'"]([a-zA-Z0-9_]+)[\'"]\s*\)')
    
    # إذا لم يكن مجلد الواجهات موجوداً أصلاً، نتوقف ونرجع قائمة فارغة
    if not os.path.exists(TEMPLATES_DIR):
        print(f"لم يتم العثور على مجلد الواجهات: {TEMPLATES_DIR}")
        return used_keys

    # نمر على كل الملفات الموجودة داخل مجلد الواجهات
    for filename in os.listdir(TEMPLATES_DIR):
        # نهتم فقط بملفات الواجهات التي تنتهي بـ '.html'
        if filename.endswith('.html'):
            path = os.path.join(TEMPLATES_DIR, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # نبحث عن جميع الكلمات التي تطابق النمط الذي حددناه بالأعلى
                matches = pattern.findall(content)
                for m in matches:
                    used_keys.add(m)  # نضيف المفتاح المكتشف لمجموعة المفاتيح المستخدمة
    return used_keys

def main():
    # 1. نحدد مسارات ملفات الترجمة للغات الثلاثة
    ar_path = os.path.join(TRANSLATIONS_DIR, 'ar.json')
    en_path = os.path.join(TRANSLATIONS_DIR, 'en.json')
    fr_path = os.path.join(TRANSLATIONS_DIR, 'fr.json')

    # 2. نقوم بتحميل المفاتيح المعرفة في كل لغة (عبر جلب مفاتيح القاموس .keys())
    ar_keys = set(load_json(ar_path).keys())
    en_keys = set(load_json(en_path).keys())
    fr_keys = set(load_json(fr_path).keys())
    
    # 3. نقوم بمسح الواجهات لنعرف ما هي المفاتيح المستخدمة فعلياً في الموقع
    used_keys = scan_templates()
    
    # 4. نطبع ملخصاً سريعاً للأعداد في الـ Terminal
    print("=== ملخص فحص التراجم ===")
    print(f"إجمالي المفاتيح المطلوبة في الواجهات: {len(used_keys)}")
    print(f"المفاتيح الموجودة في ملف اللغة العربية (ar.json): {len(ar_keys)}")
    print(f"المفاتيح الموجودة في ملف اللغة الإنجليزية (en.json): {len(en_keys)}")
    print(f"المفاتيح الموجودة في ملف اللغة الفرنسية (fr.json): {len(fr_keys)}")
    
    # 5. نبحث عن "المفاتيح المفقودة":
    # وهي المفاتيح التي استعملناها في كود الـ HTML ولكننا نسينا كتابتها وترجمتها في ملفات الـ JSON
    # نستخدم عملية الطرح بين المجموعات (used_keys - lang_keys) لمعرفة العناصر الموجودة في الأولى وغير موجودة في الثانية
    print("\n=== مفاتيح مستخدمة في الواجهات ولكنها ناقصة في ملفات الترجمة ===")
    missing_in_ar = sorted(list(used_keys - ar_keys))
    missing_in_en = sorted(list(used_keys - en_keys))
    missing_in_fr = sorted(list(used_keys - fr_keys))
    
    print(f"ناقصة في ملف اللغة العربية ({len(missing_in_ar)}):")
    for k in missing_in_ar:
        print(f"  - {k}")
        
    print(f"\nناقصة في ملف اللغة الإنجليزية ({len(missing_in_en)}):")
    for k in missing_in_en:
        print(f"  - {k}")
        
    print(f"\nناقصة في ملف اللغة الفرنسية ({len(missing_in_fr)}):")
    for k in missing_in_fr:
        print(f"  - {k}")
        
    # 6. نبحث عن "المفاتيح غير المستخدمة":
    # وهي مفاتيح قمنا بترجمتها ووضعها في ملفات الـ JSON، ولكننا قمنا بحذفها من كود الـ HTML ولم نعد نستخدمها
    # (عملية الطرح العكسية: lang_keys - used_keys)
    print("\n=== مفاتيح معرفة في ملفات الترجمة ولكنها غير مستخدمة في أي واجهة ===")
    unused_in_ar = sorted(list(ar_keys - used_keys))
    unused_in_en = sorted(list(en_keys - used_keys))
    unused_in_fr = sorted(list(fr_keys - used_keys))
    
    print(f"غير مستخدمة في العربية ({len(unused_in_ar)}): {unused_in_ar}")
    print(f"غير مستخدمة في الإنجليزية ({len(unused_in_en)}): {unused_in_en}")
    print(f"غير مستخدمة في الفرنسية ({len(unused_in_fr)}): {unused_in_fr}")

if __name__ == '__main__':
    main()
