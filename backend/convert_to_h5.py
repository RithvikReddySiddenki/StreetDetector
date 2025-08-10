# convert_to_h5.py
import onnx
from onnx2keras import onnx_to_keras

onnx_path = r"yolov5\yolov5m.onnx"    # the ONNX you exported
print(f"Loading ONNX from: {onnx_path}")
model_onnx = onnx.load(onnx_path)

# Key changes:
# - name_policy='short'  -> avoids "/" in layer names
# - change_ordering=True -> convert to channels_last (Keras default)
# - verbose=True         -> helpful logs if anything else fails
k_model = onnx_to_keras(
    model_onnx,
    ['images'],
    name_policy='short',
    change_ordering=True,
    verbose=True
)

out_path = r"backend\data\yolo_weights.h5"
k_model.save(out_path)
print(f"Saved Keras model to: {out_path}")