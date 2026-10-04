"""Cài đặt chỉnh trên UI (tab ⚙ Cài đặt), lưu theo workspace / dự án ở `<workspace>/settings.json`.

configs/autolabel.yaml vẫn là mặc định chung; file này chỉ ghi các giá trị người dùng đổi và được áp đè lên khi đọc
config (routes.get_config, bước gán nhãn của dự án). Chỉ các trường trong FIELDS được đổi, mỗi trường có kiểu và
khoảng hợp lệ, nên UI không thể làm hỏng config.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.qa_config import AutoLabelConfig

SETTINGS_FILE = "settings.json"
REPO_ROOT = Path(__file__).resolve().parents[2]  # đường dẫn trọng số trong config tính từ gốc repo

# path trong config -> mô tả cho UI. kind: choice | bool | float | int
# label / choices / help: ngắn, hiện thẳng trên trang. detail: số đo làm căn cứ, chỉ hiện khi rê chuột vào ⓘ.
# group: tiêu đề nhóm trên trang. applies: now = có hiệu lực ngay | relabel = cần bấm "Áp dụng lại" cho frame cũ
FIELDS: dict[str, dict] = {
    "propagation.flow": {
        "kind": "choice",
        "group": "Lan truyền 2D",
        "label": "Dự đoán chuyển động",
        "choices": {
            "always": "Optical flow",
            "dam4sam": "DAM4SAM (cần GPU)",
            "flow+dam4sam": "Optical flow + DAM4SAM (cần GPU)",
            "missing": "Optical flow, chỉ ảnh chưa detect",
            "off": "Vận tốc không đổi",
        },
        "help": "Khuyên dùng: Optical flow.",
        "detail": "Held-out 20 scene (795 keyframe): optical flow ở mọi ảnh cho thêm 14% nhãn lan truyền đúng "
        "(2910 → 3329), đổi ID 265 → 97, mất dấu −22%; tốn ~1 s cho mỗi lần lan truyền 10 keyframe trên CPU. "
        "DAM4SAM + BoT-SORT: HOTA 0.573 so với 0.563, chậm hơn 3,4 lần.",
        "applies": "now",
    },
    "propagation.association": {
        "kind": "choice",
        "group": "Lan truyền 2D",
        "label": "Ghép với detection",
        "choices": {
            "byte": "ByteTrack",
            "botsort": "BoT-SORT",
            "single": "IoU một lượt",
        },
        "help": "Khuyên dùng: ByteTrack.",
        "detail": "ByteTrack ghép box score cao trước, box score thấp chỉ cho track còn thiếu. Dev: box lan truyền sai "
        "224 → 203, nhãn đúng giữ nguyên; held-out 4 scene: sai 329 → 300. BoT-SORT thêm ngoại hình (màu) và bù chuyển "
        "động camera, gần như không đổi khi đã có optical flow.",
        "applies": "now",
    },
    "propagation.oc_recover": {
        "kind": "bool",
        "group": "Lan truyền 2D",
        "label": "Nhận lại vật bị che (OC-SORT)",
        "help": "Không khuyên bật.",
        "detail": "Track đang mất được ghép lại theo box quan sát cuối. 20 scene test: box ghi ra đúng 3192 → 3215 "
        "(+0.7%) nhưng đổi ID 76 → 87, box sai 860 → 868.",
        "applies": "now",
    },
    "propagation.max_keyframes": {
        "kind": "int",
        "group": "Lan truyền 2D",
        "label": "Số keyframe tối đa mỗi lần",
        "min": 1,
        "max": 200,
        "help": "",
        "detail": "Lan truyền dừng sau bấy nhiêu keyframe, hoặc trước frame đã có người mở.",
        "applies": "now",
    },
    "propagation3d.max_misses": {
        "kind": "int",
        "group": "Lan truyền 3D",
        "label": "Số keyframe giữ track khi mất dấu",
        "min": 1,
        "max": 10,
        "help": "",
        "detail": "Vật bị che quá bấy nhiêu keyframe thì dừng. 2 → 4 (mặc định): test 24 scene nhãn đúng +1.6% "
        "(23355 → 23731), đổi ID 470 → 543.",
        "applies": "now",
    },
    "propagation3d.oc_recover": {
        "kind": "bool",
        "group": "Lan truyền 3D",
        "label": "Nhận lại theo vị trí cuối (OC-SORT)",
        "help": "Không khuyên bật.",
        "detail": "Thêm nhãn đúng trên test (23731 → 23872) nhưng không trên dev, và thêm đổi ID.",
        "applies": "now",
    },
    "detection.yoloe.weights": {
        "kind": "choice",
        "group": "Gán nhãn",
        "label": "Mô hình phát hiện 2D",
        "choices": {
            "weights/yoloe-26l-nuimages-full-1280.pt": "Fine-tune full",
            "weights/yoloe-26l-nuimages-lp-1280.pt": "Fine-tune mini",
            "yoloe-26l-seg.pt": "Original",
        },
        "help": "Khuyên dùng: Fine-tune full. Cần lớp ngoài 10 lớp nuScenes: chọn Original.",
        "detail": "Held-out nuScenes CAM_FRONT 4852 keyframe, mAP50 / recall: Original (YOLOE gốc, open-vocab) "
        "0.266 / 0.478, Fine-tune mini (linear probe) 0.305 / 0.530, Fine-tune full (toàn mạng) 0.585 / 0.782. Hai bản "
        "fine-tune (nuImages) chỉ nhận 10 lớp nuScenes, "
        "không nhận prompt chữ mới. Đổi mô hình thì 'Áp dụng lại' chạy lại detector trên các frame chưa ai sửa.",
        "applies": "relabel",
    },
    "detection.min_score": {
        "kind": "float",
        "group": "Gán nhãn",
        "label": "Ngưỡng giữ box",
        "min": 0.05,
        "max": 0.9,
        "step": 0.05,
        "help": "Thấp: ít sót, nhiều box phải xoá.",
        "detail": "Box có score dưới ngưỡng bị bỏ trước khi duyệt. Đổi ngưỡng không phải chạy lại detector.",
        "applies": "relabel",
    },
    "qa.temporal.rescore": {
        "kind": "choice",
        "group": "Gán nhãn",
        "label": "Tính lại score theo sweep",
        "choices": {
            "off": "Tắt",
            "mean": "Trung bình 5 ảnh",
            "linked": "Trung bình các lần thấy",
        },
        "help": "Khuyên dùng: Tắt.",
        "detail": "mAP gần như không đổi ở cả ba. Held-out 20 scene, 'Trung bình 5 ảnh': box phải xem tay 1707 → 648, "
        "box sai 2980 → 1860, tổng lỗi còn lại sau duyệt như cũ (4197 → 4190), nhưng vật sót phải vẽ thêm 2565 → 2791. "
        "'Trung bình các lần thấy': ít sót hơn, nhiều box sai hơn. Bật cùng so khớp bằng optical flow thì lỗi còn lại "
        "tăng ~9%.",
        "applies": "relabel",
    },
    "qa.temporal.flow": {
        "kind": "bool",
        "group": "Gán nhãn",
        "label": "So khớp sweep bằng optical flow",
        "help": "Không khuyên bật.",
        "detail": "Box ở sweep t−2…t+2 được dời về thời điểm keyframe trước khi so: cờ FLICKER đúng hơn (precision "
        "0.79 → 0.90) nhưng bắt ít lỗi hơn hẳn — lỗi lọt qua duyệt theo lô 1632 → 2082 (held-out 20 scene).",
        "applies": "relabel",
    },
}


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(obj, path: str, value) -> None:
    *parents, last = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, last, value)


def validate(values: dict) -> dict:
    """Giữ các trường biết, ép kiểu và chặn khoảng; ValueError nếu giá trị sai."""
    out = {}
    for path, v in values.items():
        spec = FIELDS.get(path)
        if spec is None:
            raise ValueError(f"Không chỉnh được '{path}' trên UI")
        kind = spec["kind"]
        if kind == "choice":
            if v not in spec["choices"]:
                raise ValueError(f"{spec['label']}: giá trị '{v}' không hợp lệ")
        elif kind == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{spec['label']}: cần true / false")
        else:
            try:
                v = int(v) if kind == "int" else round(float(v), 4)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{spec['label']}: cần một số") from e
            if not spec["min"] <= v <= spec["max"]:
                raise ValueError(f"{spec['label']}: trong khoảng {spec['min']}–{spec['max']}")
        out[path] = v
    return out


def missing_weights(values: dict) -> str | None:
    """File trọng số được chọn mà chưa có trên máy (None nếu đủ). Tên trần như yoloe-26l-seg.pt thì Ultralytics tự tải."""
    for path, v in values.items():
        if path.endswith(".weights") and "/" in str(v) and not (REPO_ROOT / v).exists():
            return str(v)
    return None


def load(root: str | Path) -> dict:
    path = Path(root) / SETTINGS_FILE
    if not path.exists():
        return {}
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, json.JSONDecodeError):
        return {}  # file hỏng / trường cũ: bỏ qua, dùng mặc định


def save(root: str | Path, values: dict) -> dict:
    values = validate(values)
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / SETTINGS_FILE).write_text(json.dumps(values, indent=1, ensure_ascii=False), encoding="utf-8")
    return values


def apply(config: AutoLabelConfig, overrides: dict) -> AutoLabelConfig:
    if not overrides:
        return config
    cfg = config.model_copy(deep=True)
    for path, v in overrides.items():
        _set(cfg, path, v)
    return cfg


def describe(base: AutoLabelConfig, overrides: dict) -> list[dict]:
    """Danh sách trường cho UI: giá trị đang dùng, mặc định (từ autolabel.yaml), mô tả."""
    cfg = apply(base, overrides)
    return [
        {"path": path, **spec, "value": _get(cfg, path), "default": _get(base, path), "changed": path in overrides}
        for path, spec in FIELDS.items()
    ]
