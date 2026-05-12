from onnx import shape_inference
import onnx
f = "/home/faith/PlateRecognition/PlateDetectionRecognition/test/yolov5plate.onnx"
# 执行 shape 推理
model_onnx = onnx.load(f)  # load onnx model
inferred_model = shape_inference.infer_shapes(model_onnx)

# # 保存新模型
# onnx.save(inferred_model, "your_model_inferred.onnx")

# Metadata
# d = {'stride': int(max(model.stride)), 'names': model.names}
# for k, v in d.items():
#     meta = inferred_model.metadata_props.add()
#     meta.key, meta.value = k, str(v)
f = "yolov5plate-shape.onnx"
onnx.save(inferred_model, f)