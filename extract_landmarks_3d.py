#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
حساب متوسط نقاط اليد (2D) لكل حرف وحفظها في ملف JavaScript.

يقرأ:
  - arabic_data/arabic_keypoints.csv  (label_index, 42 قيمة)
  - arabic_data/arabic_labels.csv     (index, letter)

ويكتب:
  - static/hand_landmarks.js
"""

import csv
import os
from collections import defaultdict


DATA_DIR = "arabic_data"
KEYPOINTS_PATH = os.path.join(DATA_DIR, "arabic_keypoints.csv")
LABELS_PATH = os.path.join(DATA_DIR, "arabic_labels.csv")
OUTPUT_JS_PATH = os.path.join("static", "hand_landmarks.js")


def load_labels():
    """Load label_index -> arabic letter from labels CSV."""
    labels = {}
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                idx = int(row[0])
                letter = row[1]
                labels[idx] = letter
            except (ValueError, IndexError):
                continue
    return labels


def load_keypoints():
    """Load all samples from keypoints CSV grouped by label index."""
    sums = defaultdict(lambda: [0.0] * 42)
    counts = defaultdict(int)

    if not os.path.exists(KEYPOINTS_PATH):
        print(f"Keypoints file not found: {KEYPOINTS_PATH}")
        return sums, counts

    with open(KEYPOINTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                if len(row) == 44:
                    label_idx = int(row[1])
                    values = [float(v) for v in row[2:]]
                elif len(row) == 43:
                    label_idx = int(row[0])
                    values = [float(v) for v in row[1:]]
                else:
                    continue
                if len(values) != 42:
                    continue
            except ValueError:
                continue

            counts[label_idx] += 1
            acc = sums[label_idx]
            for i, v in enumerate(values):
                acc[i] += v

    return sums, counts


def compute_means(labels, sums, counts):
    """Compute mean vector (42 values) per letter."""
    letter_means = {}
    for idx, letter in labels.items():
        c = counts.get(idx, 0)
        if c == 0:
            # لا عينات لهذا الحرف
            letter_means[letter] = []
        else:
            acc = sums[idx]
            means = [v / c for v in acc]
            letter_means[letter] = means
    return letter_means


def write_js(letter_means):
    """Write JS file with HAND_LANDMARKS constant."""
    os.makedirs(os.path.dirname(OUTPUT_JS_PATH), exist_ok=True)

    lines = []
    lines.append("const HAND_LANDMARKS = {")

    first = True
    for letter, values in letter_means.items():
        # تعليق بعد السطر يوضح عدد القيم
        if first:
            first = False
        # تنسيق القيم في سطر واحد مع تقريب معقول
        if values:
            vals_str = ", ".join(f"{v:.6f}" for v in values)
        else:
            vals_str = ""
        lines.append(f'  "{letter}": [{vals_str}],')

    lines.append("};")
    lines.append("")  # newline at end

    with open(OUTPUT_JS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    labels = load_labels()
    sums, counts = load_keypoints()
    letter_means = compute_means(labels, sums, counts)

    write_js(letter_means)

    # طباعة عدد العينات لكل حرف
    print("\nSamples used per letter:")
    for idx, letter in labels.items():
        c = counts.get(idx, 0)
        print(f"  {letter}: {c} sample(s)")

    print(f"\nOutput file created: {OUTPUT_JS_PATH}")


if __name__ == "__main__":
    main()

