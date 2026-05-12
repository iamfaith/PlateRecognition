#!/usr/bin/env python3
"""
Convert YOLO txt labels (class x y w h) to extended format with 4 points (8 values).
This script assumes you have some method to generate landmarks (e.g., from manual labels or another tool).
It provides a helper to append landmark values to each label line; for missing landmarks it pads with -1.

Usage:
    python convert_labels_to_landmark.py --src labels/ --dst labels_landmark/ --add example.txt

"""
import os
import argparse


def convert_file(src_file, dst_file, landmarks_for_image=None):
    # landmarks_for_image: optional dict mapping image basename -> [lmk0x, lmk0y, ..., lmk3x, lmk3y]
    with open(src_file, 'r') as f_in, open(dst_file, 'w') as f_out:
        for line in f_in:
            vals = line.strip().split()
            if not vals:
                continue
            imgname = os.path.splitext(os.path.basename(src_file))[0]
            lm = landmarks_for_image.get(imgname) if landmarks_for_image else None
            if lm and len(lm) == 8:
                f_out.write(line.strip() + ' ' + ' '.join(str(x) for x in lm) + '\n')
            else:
                # pad with -1 for missing landmarks
                f_out.write(line.strip() + ' ' + ' '.join(['-1'] * 8) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source labels directory')
    parser.add_argument('--dst', required=True, help='destination labels directory')
    parser.add_argument('--landmarks', help='optional JSON mapping image->landmarks')
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    landmarks_map = {}
    if args.landmarks:
        import json
        with open(args.landmarks, 'r') as f:
            landmarks_map = json.load(f)
    for fn in os.listdir(args.src):
        if not fn.endswith('.txt'):
            continue
        convert_file(os.path.join(args.src, fn), os.path.join(args.dst, fn), landmarks_map)
