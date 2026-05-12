# Usage: Training & Inference (with and without landmarks)

This guide explains how to run YOLOv5 v6.2 after applying the landmark patches. It covers both modes:
- No-landmark (backward compatible): `num_landmarks=0` (default)
- Landmark-enabled: set `num_landmarks > 0` (e.g., 4 → 8 dims)

Files referenced below are in the YOLOv5 repository (e.g. `~/yolov5`). Adapt paths if different.

---

## 1. Data format

- No-landmark (existing YOLO format): each label `.txt` line:
  `class x_center y_center width height`  (normalized 0..1)

- With-landmark (4 points → 8 values): each label `.txt` line:
  `class x_center y_center width height lmk0_x lmk0_y lmk1_x lmk1_y lmk2_x lmk2_y lmk3_x lmk3_y`
  - All coordinates normalized to [0,1] relative to original image width/height
  - If a landmark is missing, set its x/y to `-1` (and the dataset code/mask will ignore it)

Use `docs/yolov5_landmark_integration/patches/convert_labels_to_landmark.py` to add/pad landmark columns.

---

## 2. Enabling / disabling landmarks in the model

Two options to control the model-side behavior.

Option A (quick, edit file):
- Open `models/yolo.py` and set the `Detect` default argument `num_landmarks`:
  - No landmarks (default): `def __init__(..., num_landmarks=0, ...)`
  - With landmarks (4 points): `def __init__(..., num_landmarks=4, ...)`

Option B (per-model config, advanced):
- Modify your model YAML and `models/yolo.py` loader to pass `num_landmarks` when constructing `Detect`. This requires editing the code that parses the YAML and creates modules — see YOLOv5 model creation in `models/common.py`/`models/yolo.py`.

Notes:
- `num_landmarks=0` preserves original `no = 5 + nc` behavior.
- `num_landmarks=4` results in `no = 5 + 2*4 + nc = 13 + nc` (for nc=2 → 15).

---

## 3. Hyperparameters

- In your `hyp.yaml` (e.g. `data/hyp.scratch-low.yaml`) add a line:

```yaml
landmark_loss_gain: 1.0  # scale for landmark loss, tune as needed
```

- You can also add `cls_pw` or other existing hyperparameters as before.

---

## 4. Training commands

Basic training (no-landmark, normal YOLOv5):

```bash
cd ~/yolov5
python train.py --img 640 --batch 16 --epochs 50 --data data/plate.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt
```

Training with landmarks enabled (assumes you set `num_landmarks>0` in `models/yolo.py` or passed via model YAML):

1. Ensure label files have 13 values per line (1+4+8).
2. Ensure `hyp.yaml` contains `landmark_loss_gain`.
3. Run training same as before:

```bash
cd ~/yolov5
python train.py --img 640 --batch 16 --epochs 50 --data data/plate_landmark.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --hyp data/hyp.scratch-low.yaml
```

Notes:
- If you prefer a quick smoke-test, use `--epochs 1 --batch 2` to validate shapes and that no NaNs occur.
- Monitor training logs: after applying patches you should see the usual loss entries and an additional landmark-related loss (`llmk` or similar) if implemented in `ComputeLoss`.

---

## 5. Inference / detection

After training, detection output per prediction row will be:

```
[x_center, y_center, width, height, obj_conf, lmk0_x, lmk0_y, ..., lmk3_x, lmk3_y, cls0, cls1, ...]
```

Where the landmark slice exists only if `num_landmarks > 0`.

Run inference (standard):

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source data/images --conf 0.25
```

If you want to visualize landmarks after inference, use the snippet below to extract and transform coordinates back to the original image.

### Visualization snippet (PyTorch inference outputs)

```python
import cv2
import numpy as np

# preds: array of shape (N, no) from model (post-decoding by Detect), or use onnxruntime output
# assume no = 5 + 2*M + nc when landmarks present
M = 4  # number of points
no = preds.shape[1]
landmark_exists = no >= (5 + 2*M + 1)  # simple check

for row in preds:
    x, y, w, h = row[0:4]
    obj = row[4]
    if landmark_exists:
        lmk = row[5:5+2*M]
        # convert center xywh -> xyxy
        x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
        # lmk are absolute coords in input scale (Detect already decoded to input space if patches decode there)
        pts = []
        for i in range(0, len(lmk), 2):
            lx = lmk[i]; ly = lmk[i+1]
            pts.append((int(lx), int(ly)))
        # draw
        for (lx, ly) in pts:
            if lx >= 0 and ly >= 0:
                cv2.circle(img, (lx, ly), 3, (0, 255, 0), -1)
```

If using ONNX output where the model outputs are still in padded-input coordinates, you must reverse the letterbox transform (same as for bbox):

```python
# given pad_x, pad_y, scale (as in infer script):
orig_x = (lx - pad_x) / scale
orig_y = (ly - pad_y) / scale
```

---

## 6. Export to ONNX

After training, export weights to ONNX as usual. Verify the output channel count matches your chosen `num_landmarks`:

```bash
python export.py --weights runs/train/exp/weights/best.pt --img 640 --include onnx
python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession('best.onnx')
print(sess.get_outputs()[0].shape)
PY
```

Expected final dimension: `no = 5 + 2*M + nc` if landmarks enabled, otherwise `no = 5 + nc`.

---

## 7. How to run without modifying code repeatedly

- Quick toggle: keep `models/yolo.py` default `num_landmarks=0` and maintain two branches or two copies of the model file:
  - `models/yolo.py` (no landmarks)
  - `models/yolo_lmk.py` (with `num_landmarks=4`) — switch by renaming or editing the import in the model loader

- Preferred (clean): modify model YAML & loader so `num_landmarks` parameter is read from YAML; then you can provide different YAMLs per experiment:
  - `models/yolov5s_lmk.yaml` with `num_landmarks: 4`
  - `models/yolov5s.yaml` with `num_landmarks: 0`

---

## 8. Smoke-test checklist (quick)

1. Set `num_landmarks=4` in `models/yolo.py` or use a YAML that sets it.
2. Prepare one image and an associated label file with 13 values.
3. Run a 1-epoch 1-batch training:
   ```bash
   python train.py --img 640 --batch 2 --epochs 1 --data data/your_plate_landmark.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt
   ```
4. Confirm training starts and prints loss components and no NaN.
5. Run detection on same image and visualize landmarks.

---

If you want, I can also:
- Patch `train.py` or `detect.py` to add a `--num-landmarks` CLI switch that updates `Detect` dynamically at runtime; or
- Apply the patches to a local YOLOv5 checkout and run the smoke-test here and post the logs.

Which would you prefer next?