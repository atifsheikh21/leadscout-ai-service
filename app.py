import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

app = FastAPI(title="AI Image Service", version="1.0.0")


_YOLO_NET: Optional[cv2.dnn.Net] = None
_YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8n.onnx")


def _get_yolo_net() -> Optional[cv2.dnn.Net]:
    global _YOLO_NET
    if _YOLO_NET is not None:
        return _YOLO_NET

    if not os.path.isfile(_YOLO_MODEL_PATH):
        return None

    try:
        _YOLO_NET = cv2.dnn.readNetFromONNX(_YOLO_MODEL_PATH)
        return _YOLO_NET
    except Exception:
        return None


class ProcessRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    settings: Dict[str, Any] = Field(default_factory=dict)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_image_with_exif_autorotate(input_path: str, auto_rotate: bool) -> Image.Image:
    try:
        img = Image.open(input_path)
    except Exception as e:
        raise RuntimeError(f"Unable to read image: {e}")

    if auto_rotate:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

    return img.convert("RGB")


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_auto_enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(out, 1.25, blur, -0.25, 0)

    return out


def _rotate_bound(bgr: np.ndarray, angle_degrees: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    cX, cY = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D((cX, cY), angle_degrees, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    m[0, 2] += (nW / 2) - cX
    m[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(bgr, m, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)


def _estimate_roll_angle_degrees(bgr: np.ndarray) -> Optional[float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=80, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return None

    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < 60:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize to [-90, 90]
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180

        # Keep near-horizontal lines only
        if abs(angle) > 25:
            continue

        angles.append(angle)
        weights.append(length)

    if not angles:
        return None

    a = np.array(angles, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    return float(np.sum(a * w) / np.sum(w))


def _auto_straighten(bgr: np.ndarray, max_abs_degrees: float = 12.0) -> np.ndarray:
    angle = _estimate_roll_angle_degrees(bgr)
    if angle is None:
        return bgr

    if abs(angle) < 0.6:
        return bgr

    angle = float(np.clip(angle, -max_abs_degrees, max_abs_degrees))
    # Rotate opposite direction to correct roll
    return _rotate_bound(bgr, -angle)


def _largest_subject_bbox(bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = bgr.shape[:2]
    img_area = float(h * w)

    best = None
    best_area = 0.0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = float(cw * ch)
        if area < img_area * 0.05:
            continue
        if area > best_area:
            best_area = area
            best = (int(x), int(y), int(cw), int(ch))

    return best


def _letterbox(img: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, left, top


def _yolov8_detect_vehicle_bbox(bgr: np.ndarray, conf_thres: float = 0.35, iou_thres: float = 0.45) -> Optional[tuple[int, int, int, int]]:
    net = _get_yolo_net()
    if net is None:
        return None

    img, r, pad_x, pad_y = _letterbox(bgr, new_shape=640)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)

    net.setInput(blob)
    out = net.forward()

    # Ultralytics YOLOv8 ONNX commonly outputs shape (1, 84, 8400)
    out = np.squeeze(out)
    if out.ndim != 2:
        return None
    if out.shape[0] < 6:
        return None

    # Ensure shape is (N, 84)
    if out.shape[0] <= out.shape[1]:
        preds = out.T
    else:
        preds = out

    # COCO ids for vehicles: car=2, motorcycle=3, bus=5, truck=7
    vehicle_ids = {2, 3, 5, 7}

    boxes = []
    scores = []

    for p in preds:
        x, y, w, h = p[0:4]
        cls_scores = p[4:]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if cls_id not in vehicle_ids:
            continue
        if score < conf_thres:
            continue

        # xywh in letterboxed image space -> xyxy
        x1 = float(x - w / 2)
        y1 = float(y - h / 2)
        x2 = float(x + w / 2)
        y2 = float(y + h / 2)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)

    if not boxes:
        return None

    idxs = cv2.dnn.NMSBoxes(
        bboxes=[
            [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            for b in boxes
        ],
        scores=scores,
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )

    if idxs is None or len(idxs) == 0:
        return None

    idx = int(idxs.flatten()[0])
    x1, y1, x2, y2 = boxes[idx]

    # Map back from letterbox to original image
    x1 = (x1 - pad_x) / r
    y1 = (y1 - pad_y) / r
    x2 = (x2 - pad_x) / r
    y2 = (y2 - pad_y) / r

    h0, w0 = bgr.shape[:2]
    x1 = int(max(0, min(w0 - 1, round(x1))))
    y1 = int(max(0, min(h0 - 1, round(y1))))
    x2 = int(max(0, min(w0, round(x2))))
    y2 = int(max(0, min(h0, round(y2))))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return (x1, y1, bw, bh)


def _auto_center_reframe(bgr: np.ndarray, padding_percent: float = 0.08, bbox: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    bbox = bbox or _largest_subject_bbox(bgr)
    if bbox is None:
        return bgr

    x, y, bw, bh = bbox
    h, w = bgr.shape[:2]
    pad = float(np.clip(padding_percent, 0.0, 0.35))

    # Expand bbox by padding
    px = int(round(bw * pad))
    py = int(round(bh * pad))
    x1 = x - px
    y1 = y - py
    x2 = x + bw + px
    y2 = y + bh + py

    # Keep same aspect ratio as original by expanding crop
    target_ar = w / float(h)
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_h <= 0 or crop_w <= 0:
        return bgr

    crop_ar = crop_w / float(crop_h)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if crop_ar > target_ar:
        # Too wide, increase height
        new_h = int(round(crop_w / target_ar))
        crop_h = new_h
    else:
        # Too tall, increase width
        new_w = int(round(crop_h * target_ar))
        crop_w = new_w

    x1 = int(round(cx - crop_w / 2.0))
    x2 = int(round(cx + crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    y2 = int(round(cy + crop_h / 2.0))

    # If crop goes out of bounds, pad with reflect
    left = max(0, -x1)
    top = max(0, -y1)
    right = max(0, x2 - w)
    bottom = max(0, y2 - h)

    if left or top or right or bottom:
        bgr = cv2.copyMakeBorder(bgr, top, bottom, left, right, borderType=cv2.BORDER_REFLECT101)
        x1 += left
        x2 += left
        y1 += top
        y2 += top

    return bgr[y1:y2, x1:x2]


def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _upscale(bgr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return bgr
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


@dataclass
class OutputInfo:
    width: int
    height: int


def _save_image(pil_img: Image.Image, output_path: str, fmt: str, quality: int) -> None:
    fmt = fmt.lower()
    if fmt not in {"jpg", "png", "webp"}:
        fmt = "jpg"

    params: Dict[str, Any] = {}
    if fmt == "jpg":
        params["quality"] = int(quality)
        params["optimize"] = True
        params["progressive"] = True
        pil_img.save(output_path, format="JPEG", **params)
    elif fmt == "webp":
        params["quality"] = int(quality)
        params["method"] = 6
        pil_img.save(output_path, format="WEBP", **params)
    else:
        params["optimize"] = True
        pil_img.save(output_path, format="PNG", **params)


def process_image(input_path: str, output_path: str, settings: Dict[str, Any]) -> OutputInfo:
    auto_rotate = bool(settings.get("auto_rotate", True))
    auto_enhance = bool(settings.get("auto_enhance", True))
    auto_straighten = bool(settings.get("auto_straighten", False))
    auto_center = bool(settings.get("auto_center", False))
    padding_percent = float(settings.get("padding_percent", 0.08))
    center_mode = str(settings.get("center_mode", "classic")).lower()
    target = str(settings.get("target", "original"))
    upscale = str(settings.get("upscale", "none"))
    output_quality = int(settings.get("output_quality", 85))
    output_format = str(settings.get("output_format", "jpg")).lower()

    output_quality = max(40, min(95, output_quality))

    if target == "1080p":
        max_side = 1920
    elif target == "4k":
        max_side = 3840
    else:
        max_side = 0

    if upscale == "2x":
        upscale_factor = 2
    elif upscale == "4x":
        upscale_factor = 4
    else:
        upscale_factor = 1

    pil_img = _read_image_with_exif_autorotate(input_path, auto_rotate=auto_rotate)
    bgr = _pil_to_bgr(pil_img)

    if auto_straighten:
        bgr = _auto_straighten(bgr)

    if auto_center:
        bbox = None
        if center_mode == "yolo":
            bbox = _yolov8_detect_vehicle_bbox(bgr)
        bgr = _auto_center_reframe(bgr, padding_percent=padding_percent, bbox=bbox)

    if auto_enhance:
        bgr = _apply_auto_enhance(bgr)

    if upscale_factor > 1:
        bgr = _upscale(bgr, upscale_factor)

    if max_side > 0:
        bgr = _resize_max_side(bgr, max_side)

    out_pil = _bgr_to_pil(bgr)

    _ensure_parent_dir(output_path)
    _save_image(out_pil, output_path, fmt=output_format, quality=output_quality)

    w, h = out_pil.size
    return OutputInfo(width=int(w), height=int(h))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    output_quality: int = Form(85),
    output_format: str = Form("jpg"),
    target: str = Form("original"),
    upscale: str = Form("none"),
    auto_rotate: bool = Form(True),
    auto_enhance: bool = Form(True),
    auto_straighten: bool = Form(False),
    auto_center: bool = Form(False),
    center_mode: str = Form("classic"),
    padding_percent: float = Form(0.08),
) -> Response:
    try:
        # Read uploaded file
        contents = await image.read()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or ".jpg")[1]) as input_tmp:
            input_tmp.write(contents)
            input_path = input_tmp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as output_tmp:
            output_path = output_tmp.name
        
        # Build settings
        settings = {
            "output_quality": output_quality,
            "output_format": output_format,
            "target": target,
            "upscale": upscale,
            "auto_rotate": auto_rotate,
            "auto_enhance": auto_enhance,
            "auto_straighten": auto_straighten,
            "auto_center": auto_center,
            "center_mode": center_mode,
            "padding_percent": padding_percent,
        }
        
        # Process image
        info = process_image(input_path, output_path, settings)
        
        # Read processed image
        with open(output_path, "rb") as f:
            processed_data = f.read()
        
        # Cleanup temp files
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass
        
        # Return processed image
        return Response(
            content=processed_data,
            media_type=f"image/{output_format}",
            headers={
                "X-Image-Width": str(info.width),
                "X-Image-Height": str(info.height),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
