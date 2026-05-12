# YOLOv5 Landmark Integration — Documentation and Patches

This folder documents how to add landmark/keypoint prediction to a YOLOv5-style detection model, and provides patch files to apply to a standard YOLOv5 repository.

Overview
- Predict M landmarks per detection (each landmark is an (x,y) pair). Typical M = 4 for plate corner points.
 - Predict M landmarks per detection (each landmark is an (x,y) pair). Typical M = 4 for plate corner points.
 - Compatibility: patches are conditional — set `num_landmarks=0` to keep original behavior (no landmarks), or >0 to enable landmarks.
- Model output per anchor becomes: [x, y, w, h, obj, lmk0x, lmk0y, ..., lmk(M-1)x, lmk(M-1)y, cls0, cls1, ...]
- Landmarks are predicted in the model input (letterbox/padded) coordinate system and must be transformed back to original image coordinates by undoing pad/scale, same as bbox.

This README contains:
- patch descriptions and locations
- implementation notes and hyperparameters
- testing and export steps

Files in this folder
- patches/modify_detect_v6.2.patch — Detect layer changes for YOLOv5 v6.2 to add landmark channels
- patches/modify_datasets_v6.2.patch — Dataset parsing changes to support 4 points (8 values) and guidance for augment transforms
- patches/modify_loss_v6.2.patch — Loss changes to gather landmark targets and compute SmoothL1 landmark loss
- patches/modify_export_v6.2.patch — Notes to ensure ONNX export includes landmark outputs
- patches/convert_labels_to_landmark.py — helper script to convert/augment label files with landmark columns
- SUMMARY.md — concise checklist and commands

Important design choices
- Coordinate format: label files should contain normalized coordinates in YOLO format: `class x_center y_center width height lmk0x lmk0y ...` where all coordinates are in [0,1] relative to original image width/height.
- Prediction representation: two common choices
  1. Predict normalized absolute coords (0..1) for landmarks (apply `sigmoid` to outputs). Simpler to decode: `x_img = (sigmoid(pred)*input_w - pad_x)/scale`.
  2. Predict grid-cell-relative offsets (like xy head): more consistent with existing xy representation but more code changes in target build.
- Loss: use SmoothL1 or L1 on positive anchors only. Add `lambda_lmk` hyperparameter to weight the landmark loss relative to box/cls.

Testing
1. Apply patches to your YOLOv5 repo (see patch files in `patches/`).
2. Run a dry training step (one batch, few iterations) and print shapes for preds and targets. Ensure no NaNs.
3. Export to ONNX and verify output shape `C = 5 + 2*M + nc`.
4. Run inference and visualize landmarks on the image (draw small circles at landmark coords after mapping back to original image).

If you want, I can also attempt to apply these changes into a YOLOv5 checkout inside this workspace and run a smoke test. Otherwise, apply the patch files to your YOLOv5 repo with `git apply` or `git am`.

---

Next: see `patches/` for the actual diffs and example snippets.
