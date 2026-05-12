#!/usr/bin/env python3
"""
Simple ONNX Runtime inference script for plate detection.
Loads a detection ONNX model (default: yolov5plate.onnx in repo) and runs
inference on an input image, draws boxes and confidences on the image,
and writes the result to the output path.

Usage:
  python3 infer_plate_onnx.py --image path/to/img.jpg [--det-model path/to/yolov5plate.onnx] --output out.jpg
"""
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import os
import time


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)


def xywh2xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]


def non_max_suppression(boxes, scores, iou_thres=0.45):
    # boxes: list of [x1,y1,x2,y2]
    if len(boxes) == 0:
        return []
    rects = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
    idxs = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0, nms_threshold=iou_thres)
    if len(idxs) == 0:
        return []
    idxs = idxs.flatten().tolist()
    return idxs


def run_detection(session, img, conf_thres=0.25, iou_thres=0.45, input_size=640):
    # Prepare image exactly like C++ preprocess_img (letterbox with centered padding)
    h0, w0 = img.shape[:2]
    input_w = input_size
    input_h = input_size
    r_w = input_w / (w0 * 1.0)
    r_h = input_h / (h0 * 1.0)
    if r_h > r_w:
        new_w = input_w
        new_h = int(r_w * h0)
        pad_x = 0
        pad_y = int((input_h - new_h) / 2)
        scale = r_w
    else:
        new_w = int(r_h * w0)
        new_h = input_h
        pad_x = int((input_w - new_w) / 2)
        pad_y = 0
        scale = r_h

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_trans = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_trans})


    a = np.asarray(outputs[0])  # shape (1,25200,15)
    print('per-row len:', a.shape[-1])
    print('example cls_probs:', a.reshape(-1, a.shape[-1])[0,:])


    # Support outputs shaped (1, N, C) or (N, C) or multiple outputs
    arr = None
    for out in outputs:
        a = np.asarray(out)
        if a.ndim == 3 and a.shape[0] == 1:
            arr = a[0]
            break
        if a.ndim == 2:
            arr = a
            break
    if arr is None:
        arrs = [np.asarray(o) for o in outputs if np.asarray(o).ndim >= 2]
        if len(arrs) == 0:
            return []
        arr = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrs], axis=0)

    boxes = []
    scores = []
    dets_info = []

    # Follow C++ layout: [x,y,w,h,obj_conf, lmk0x..lmk7y (8 values), cls_scores...] 5+8+2=15 total per row
    for row in arr:
        if row.size < 6:
            continue
        x, y, w, h = row[0:4]
        obj_conf = float(row[4])
        lmk = row[5:13] if row.size >= 13 else np.zeros(8)
        cls_probs = row[13:]
        if cls_probs.size == 0:
            cls_conf = 1.0
            cls = 0
        else:
            cls = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls])
        conf = obj_conf * cls_conf
        if conf < conf_thres:
            continue

        # Convert bbox from model (center x,y,w,h in padded input space) to original image coords
        x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)
        # undo padding and scaling (reverse of canvas placement)
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale
        x1 = max(0, min(w0 - 1, x1))
        x2 = max(0, min(w0 - 1, x2))
        y1 = max(0, min(h0 - 1, y1))
        y2 = max(0, min(h0 - 1, y2))

        # map landmarks similarly
        keypoints = []
        for i in range(0, 8, 2):
            lx = lmk[i]
            ly = lmk[i + 1]
            # undo padding and scale
            lx = (lx - pad_x) / scale
            ly = (ly - pad_y) / scale
            keypoints.append((float(lx), float(ly)))

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        dets_info.append({'cls': cls, 'conf': conf, 'kps': keypoints})

    keep = non_max_suppression(boxes, scores, iou_thres=iou_thres)
    results = []
    for i in keep:
        b = boxes[i]
        info = dets_info[i]
        results.append((b, info['conf'], info['cls'], info['kps']))
    return results


def draw_results(img, results, out_path):
    # Use PIL to draw text supporting Chinese fonts
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_im)
    # font will be provided via args in main; fall back to default if not set
    font_path = getattr(draw_results, 'font_path', None)
    try:
        font = ImageFont.truetype(font_path, 20) if font_path and os.path.exists(font_path) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for item in results:
        box = item.get('bbox')
        score = item.get('score')
        cls = item.get('cls')
        plate = item.get('plate', '')
        color = item.get('color', '')
        rec_time = item.get('rec_time', None)
        x1, y1, x2, y2 = map(int, box)
        # draw rectangle with PIL
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        if rec_time is not None:
            label = f"{plate} {color} {score:.2f} {rec_time:.3f}s"
        else:
            label = f"{plate} {color} {score:.2f}"
        try:
            # Pillow >=8.0: textbbox is available
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            try:
                text_w, text_h = font.getsize(label)
            except Exception:
                text_w, text_h = (len(label) * 8, 16)
        draw.rectangle([(x1, y1 - text_h - 6), (x1 + text_w + 6, y1)], fill=(0, 255, 0))
        draw.text((x1 + 3, y1 - text_h - 2), label, fill=(0, 0, 0), font=font)

    out_bgr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)


def get_transformed_roi(src_img, keypoints):
    # keypoints: list of 4 (x,y) tuples corresponding to pts 0..3
    if keypoints is None or len(keypoints) < 4:
        return None
    pts = np.array(keypoints[:4], dtype=np.float32)
    w1 = pts[0] - pts[1]
    w2 = pts[2] - pts[3]
    width1 = np.linalg.norm(w1)
    width2 = np.linalg.norm(w2)
    maxWidth = int(max(width1, width2))

    h1 = pts[0] - pts[3]
    h2 = pts[1] - pts[2]
    height1 = np.linalg.norm(h1)
    height2 = np.linalg.norm(h2)
    maxHeight = int(max(height1, height2))

    if maxWidth <= 0 or maxHeight <= 0:
        return None

    pts_std = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, pts_std)
    dst = cv2.warpPerspective(src_img, M, (maxWidth, maxHeight))
    return dst


def get_split_merge(img):
    # Split double-layer plate and merge (upper | lower) like C++ get_split_merge
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    upper_h = int(5.0 / 12 * h)
    lower_y = int(1.0 / 3 * h)
    if upper_h <= 0 or lower_y >= h:
        return img
    upper = img[0:upper_h, 0:w]
    lower = img[lower_y:h, 0:w]
    if upper.size == 0 or lower.size == 0:
        return img
    # resize upper to lower size
    try:
        upper_resized = cv2.resize(upper, (lower.shape[1], lower.shape[0]), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return img
    out_h = lower.shape[0]
    out_w = lower.shape[1] + upper_resized.shape[1]
    out = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
    out[0:upper_resized.shape[0], 0:upper_resized.shape[1]] = upper_resized
    out[0:lower.shape[0], upper_resized.shape[1]:] = lower
    return out


def run_plate_recognition(session, roi_img):
    # Returns (plate_str, plate_color)
    if roi_img is None or roi_img.size == 0:
        return "", ""
    # preprocess to 168x48 and normalize same as C++: channel order B,G,R and (v/255 - 0.588)/0.193
    pr = cv2.resize(roi_img, (168, 48))
    pr = pr.astype(np.float32)
    H, W = 48, 168
    data = np.zeros((1, 3, H, W), dtype=np.float32)
    # C++ uses B,G,R order from OpenCV
    for row in range(H):
        for col in range(W):
            b = pr[row, col, 0]
            g = pr[row, col, 1]
            r = pr[row, col, 2]
            data[0, 0, row, col] = (b / 255.0 - 0.588) / 0.193
            data[0, 1, row, col] = (g / 255.0 - 0.588) / 0.193
            data[0, 2, row, col] = (r / 255.0 - 0.588) / 0.193

    input_name = session.get_inputs()[0].name
    outs = session.run(None, {input_name: data})
    # expect two outputs: prob1 (21*78) and prob2 (5)
    out1 = None
    out2 = None
    for o in outs:
        a = np.asarray(o)
        if a.size == 5:
            out2 = a.reshape(-1)
        else:
            out1 = a.reshape(-1)
    if out1 is None:
        return "", ""
    # decode color
    plate_color_list = ["黑色", "蓝色", "绿色", "白色", "黄色"]
    if out2 is None:
        plate_color = ""
    else:
        idx = int(np.argmax(out2))
        plate_color = plate_color_list[idx] if idx < len(plate_color_list) else ""

    # decode plate chars
    plate_chr = ["#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁",
    "新","学","警","港","澳","挂","使","领","民","航","危","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","险","品"]

    plate_str = ""
    # out1 expected length 21*78
    if out1.size >= 21 * 78:
        for j in range(21):
            slice_j = out1[j * 78:(j + 1) * 78]
            idx = int(np.argmax(slice_j))
            plate_str += str(idx) + ";"  # placeholder - we'll postprocess below
        # convert indices to chars and remove repeats and zeros like C++
        plate_indices = []
        for j in range(21):
            slice_j = out1[j * 78:(j + 1) * 78]
            idx = int(np.argmax(slice_j))
            plate_indices.append(idx)
        pre = 0
        plate_out = ""
        for idx in plate_indices:
            if idx != 0 and idx != pre:
                if idx < len(plate_chr):
                    plate_out += plate_chr[idx]
            pre = idx
        plate_str = plate_out
    else:
        plate_str = ""

    return plate_str, plate_color


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image", required=False, help="Input image path", default="/home/faith/PlateRecognition/PlateDetectionRecognition/test/data/double3.jpg")
    parser.add_argument("--image", required=False, help="Input image path", default="/home/faith/PlateRecognition/PlateDetectionRecognition/test/data/double4.png")

    # parser.add_argument("--image", required=False, help="Input image path", default="/home/faith/fux.png")

    parser.add_argument("--det-model", default="/home/faith/PlateRecognition/PlateDetectionRecognition/test/yolov5plate.onnx", help="Detection ONNX model path")
    parser.add_argument("--rec-model", default="/home/faith/PlateRecognition/PlateRecognition/test/plate_recognition_color.onnx", help="Plate recognition ONNX model path")
    parser.add_argument("--output", default="out.jpg", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--font", default="PlateDetectionRecognition/test/NotoSansCJK-Regular.otf", help="Path to TTF/OTF font for Chinese text")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Image not found:", args.image)
        return
    if not os.path.exists(args.det_model):
        print("Detection model not found:", args.det_model)
        return

    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image:", args.image)
        return

    sess = ort.InferenceSession(args.det_model, providers=["CPUExecutionProvider"]) 
    rec_sess = None
    if os.path.exists(args.rec_model):
        rec_sess = ort.InferenceSession(args.rec_model, providers=["CPUExecutionProvider"]) 
    else:
        print("Recognition model not found, continuing without recognition:", args.rec_model)

    det_start = time.time()
    dets = run_detection(sess, img, conf_thres=args.conf, iou_thres=args.iou, input_size=args.size)
    det_time = time.time() - det_start
    print(f"Detection inference time: {det_time:.3f}s for {len(dets)} detections")
    annotated = []
    rec_total = 0.0
    rec_count = 0
    for b, conf, cls, kps in dets:
        plate_str = ""
        plate_color = ""
        roi = None
        try:
            roi = get_transformed_roi(img, kps)
        except Exception:
            roi = None
        ####################################################################################
        # If model predicts double-layer plate (label==1), split+merge to normalize layout before recognition
        if cls == 1 and roi is not None and roi.size > 0:
            try:
                roi = get_split_merge(roi)
            except Exception:
                pass
        ############################################################################
        if (roi is None or roi.size == 0) and b is not None:
            x1, y1, x2, y2 = map(int, b)
            roi = img[y1:y2, x1:x2]
        rec_time = None
        if rec_sess is not None and roi is not None and roi.size > 0:
            rec_start = time.time()
            plate_str, plate_color = run_plate_recognition(rec_sess, roi)
            rec_time = time.time() - rec_start
            rec_total += rec_time
            rec_count += 1
        annotated.append({'bbox': b, 'score': conf, 'cls': cls, 'plate': plate_str, 'color': plate_color, 'rec_time': rec_time})

    # attach font path to draw_results function for use inside
    draw_results.font_path = args.font
    draw_results(img, annotated, args.output)
    print(f"Wrote output to {args.output}")
    if rec_count > 0:
        print(f"Total recognition time: {rec_total:.3f}s for {rec_count} plates, average {rec_total/rec_count:.3f}s")
    else:
        print("No recognition runs performed.")


if __name__ == '__main__':
    main()
