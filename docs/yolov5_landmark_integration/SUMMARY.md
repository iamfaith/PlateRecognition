YOLOv5 Landmark Integration — Patch Summary

Modified components (patches included):
- models/yolo.py: Detect layer extended to predict `num_landmarks` points (2*M channels)
- utils/datasets.py: label parsing extended to read landmark coordinates and note to apply augment transforms
- utils/loss.py: compute landmark loss (SmoothL1) for positive anchors and weight by `lambda_lmk`
- export.py: note to include new output channel count when exporting to ONNX

Patch application
- Copy the `patches/*.patch` files into your YOLOv5 repo root and run:

```bash
# from YOLOv5 root
git apply /path/to/PlateRecognition/docs/yolov5_landmark_integration/patches/modify_detect.patch
git apply /path/to/PlateRecognition/docs/yolov5_landmark_integration/patches/modify_datasets.patch
git apply /path/to/PlateRecognition/docs/yolov5_landmark_integration/patches/modify_loss.patch
git apply /path/to/PlateRecognition/docs/yolov5_landmark_integration/patches/modify_export.patch
```

Testing checklist
- Run one training iteration and print shapes of preds and t_lmk:
  - verify preds per-anchor `no = 5 + 2*M + nc`
  - print `p_lmk.shape` and `t_lmk.shape` (should match)
- Export ONNX and verify output last dim equals `no`.
- Visualize landmark points after decoding back to image coords.

Notes
- Augmentations must be updated to transform landmark coords identically.
- Choose representation (grid-relative vs. normalized) and keep consistent across model, target, and decoding.

Contact
- If you want, I can apply these patches directly to a YOLOv5 checkout in this workspace and run a smoke test.

Commands to apply patches
-------------------------
Below are concrete commands you can copy/paste to apply the patches included in this repo to a local YOLOv5 checkout. Replace `~/yolov5` with your YOLOv5 path if different.

1) (Optional) Clone YOLOv5 if you don't already have it:
```bash
cd ~
git clone https://github.com/ultralytics/yolov5.git yolov5
```

2) Create and switch to a feature branch:
```bash
cd ~/yolov5
git checkout -b feat/landmark-integration
```

3) Copy patch files from this repo into YOLOv5 root:
```bash
cp /home/faith/PlateRecognition/docs/yolov5_landmark_integration/patches/*.patch .
ls -l *.patch
```

4) Check patches can apply cleanly:
```bash
git apply --check modify_detect.patch modify_datasets.patch modify_loss.patch modify_export.patch
# no output means the check passed
```

5) Preview patch summary (optional):
```bash
git apply --stat modify_detect.patch modify_datasets.patch modify_loss.patch modify_export.patch
git apply --numstat modify_detect.patch modify_datasets.patch modify_loss.patch modify_export.patch
```

6) Apply the patches:
```bash
git apply modify_detect.patch
git apply modify_datasets.patch
git apply modify_loss.patch
git apply modify_export.patch
```

7) Inspect, then commit changes:
```bash
git status
git add -A
git commit -m "feat: add landmark prediction support (detect/dataset/loss/export patches)"
```

8) If a patch fails to apply, use reject mode and manually fix `.rej` files:
```bash
git apply --reject --whitespace=fix modify_detect.patch
# open and fix the .rej files then git add/commit
```

9) Basic smoke test (install deps and run 1-epoch dry run):
```bash
# optional: create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run a tiny training job (CPU example)
python train.py --img 640 --batch 2 --epochs 1 --data data/coco128.yaml --weights yolov5s.pt --device cpu
```

10) Export ONNX and check output dims:
```bash
python export.py --weights runs/train/exp/weights/best.pt --img 640 --include onnx
python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession('best.onnx')
print(sess.get_outputs()[0].shape)
PY
```

11) Revert/rollback (if needed):
```bash
# if not committed
git reset --hard

# if committed and you want to abandon the branch
git checkout master
git branch -D feat/landmark-integration
```

Troubleshooting tips
- If shapes mismatch, inspect `models/yolo.py` `self.no` and the loss code to ensure `2*M` channels are handled consistently.
- Ensure `utils/datasets.py` augmentations apply identical transforms to landmarks.
- If using GPU, adjust `--device` and ensure CUDA/CuDNN are available.

If you'd like, I can now try to locate a YOLOv5 checkout in this workspace and apply the patches automatically.
