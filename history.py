#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import uuid
from datetime import datetime

HISTORY_DIR = 'arabic_data/users'

def _path(user_id):
    return os.path.join(HISTORY_DIR, f'{user_id}_history.json')

def get_entries(user_id):
    path = _path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_entry(user_id, text):
    entries = get_entries(user_id)
    entries.insert(0, {
        'id': uuid.uuid4().hex[:16],
        'text': text,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'date': datetime.now().strftime('%Y-%m-%d')
    })
    # احتفظ بآخر 100 فقط
    entries = entries[:100]
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(_path(user_id), 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def delete_entry(user_id, entry_id):
    entries = get_entries(user_id)
    entries = [e for e in entries if e.get('id') != entry_id]
    with open(_path(user_id), 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
