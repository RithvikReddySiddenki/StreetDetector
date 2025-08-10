# backend/main.py
# FastAPI + ONNX Runtime backend for YOLOv5 (COCO 80 classes)
# POST /predict accepts an image; add ?render=true to get an annotated PNG image back.
# Otherwise you get JSON with boxes.

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
import os
from typing import List, Tuple

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(MODEL_DIR, "yolov5m.onnx")   # <- exported ONNX
IMG_SIZE = 640   # yolov5 default
CONF_THRES = 0.25
IOU_THRES = 0.45

# COCO classes
NAMES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# Will be set at startup
ort_session = None
ort_input_name = None

# -----------------------------
# Utils
# -----------------------------
def letterbox(im: np.ndarray, new_shape=IMG_SIZE, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize + pad to square while keeping aspect ratio. Returns (padded_img, gain, (pad_w, pad_h))."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = im.shape[:2]
    r = min(new_shape[1] / w0, new_shape[0] / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Simple NMS over one class. boxes: (N,4) xyxy; returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def run_onnx(img_bgr: np.ndarray) -> np.ndarray:
    """Run ONNX model and return raw prediction (N, 85) as in YOLOv5: [x,y,w,h,conf, 80 class confs]."""
    global ort_session, ort_input_name
    im, gain, pad = letterbox(img_bgr, IMG_SIZE)
    im = im[:, :, ::-1]  # BGR -> RGB
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # HWC -> CHW
    im = np.expand_dims(im, 0)        # add batch

    out = ort_session.run(None, {ort_input_name: im})[0]  # shape: (1, 25200, 85)
    pred = out[0]  # (25200, 85)

    # decode to xyxy on the padded image
    box = xywh2xyxy(pred[:, :4])
    conf = pred[:, 4:5]
    cls = pred[:, 5:]
    scores = conf * cls  # (25200,80)

    # Select best class per box
    class_ids = scores.argmax(axis=1)
    class_scores = scores.max(axis=1)

    # Filter by confidence
    mask = class_scores >= CONF_THRES
    box = box[mask]
    class_ids = class_ids[mask]
    class_scores = class_scores[mask]

    # Scale boxes back to original image size
    # Current box coords are in padded image coordinates [0, IMG_SIZE]
    # Undo padding and gain
    if box.size:
        box[:, [0, 2]] -= pad[0]   # x padding
        box[:, [1, 3]] -= pad[1]   # y padding
        box /= gain
        # clip
        h, w = img_bgr.shape[:2]
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 2] = np.clip(box[:, 2], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
        box[:, 3] = np.clip(box[:, 3], 0, h - 1)

    # NMS per class
    final_boxes = []
    final_scores = []
    final_cls = []
    for c in np.unique(class_ids):
        idxs = np.where(class_ids == c)[0]
        keep = nms_boxes(box[idxs], class_scores[idxs], IOU_THRES)
        final_boxes.append(box[idxs][keep])
        final_scores.append(class_scores[idxs][keep])
        final_cls.append(np.full(len(keep), c))

    if final_boxes:
        final_boxes = np.concatenate(final_boxes, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
        final_cls = np.concatenate(final_cls, axis=0).astype(int)
    else:
        final_boxes = np.zeros((0, 4), dtype=np.float32)
        final_scores = np.zeros((0,), dtype=np.float32)
        final_cls = np.zeros((0,), dtype=int)

    # Stack into (N, 6): x1,y1,x2,y2,score,cls
    if final_boxes.shape[0]:
        arr = np.concatenate(
            [final_boxes,
             final_scores.reshape(-1, 1),
             final_cls.reshape(-1, 1)],
            axis=1
        )
    else:
        arr = np.zeros((0, 6), dtype=np.float32)

    return arr  # (N,6)

def draw_boxes(img_bgr: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Draw boxes on a copy of the original image (preserves original look)."""
    out = img_bgr.copy()
    for x1, y1, x2, y2, score, cls in det:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{NAMES[cls]} {score:.2f}" if 0 <= cls < len(NAMES) else f"{cls} {score:.2f}"
        # green box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # text background
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

# -----------------------------
# Startup (load ONNX)
# -----------------------------
@app.on_event("startup")
def load_model():
    global ort_session, ort_input_name
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ONNX model not found at {MODEL_PATH}. Put yolov5m.onnx under backend/data/"
        )
    # Providers: CPUExecutionProvider by default
    ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    # Find first input name (usually 'images')
    ort_input_name = ort_session.get_inputs()[0].name

@app.get("/")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_PATH), "backend": "onnxruntime"}

# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    render: bool = Query(False, description="Return PNG image with boxes if true"),
    conf: float = Query(CONF_THRES, ge=0.0, le=1.0),
    iou: float = Query(IOU_THRES, ge=0.0, le=1.0),
):
    global CONF_THRES, IOU_THRES
    CONF_THRES = float(conf)
    IOU_THRES = float(iou)

    # Read image
    img_bytes = await file.read()
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Inference
    det = run_onnx(img_bgr)  # (N,6): x1,y1,x2,y2,score,cls

    h, w = img_bgr.shape[:2]

    if render:
        vis = draw_boxes(img_bgr, det)
        ok, buf = cv2.imencode(".png", vis)
        if not ok:
            return JSONResponse({"error": "render_failed"}, status_code=500)
        return Response(content=buf.tobytes(), media_type="image/png")

    # JSON path
    boxes = []
    for x1, y1, x2, y2, score, cls in det:
        cls = int(cls)
        boxes.append({
            "x1": int(max(0, round(x1))),
            "y1": int(max(0, round(y1))),
            "x2": int(min(w, round(x2))),
            "y2": int(min(h, round(y2))),
            "label": NAMES[cls] if 0 <= cls < len(NAMES) else str(cls),
            "score": float(round(score, 6)),
        })
    return {"boxes": boxes, "width": w, "height": h}
