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

# path trong config -> mô tả cho UI. kind: choice | bool | float | int
FIELDS: dict[str, dict] = {
    "propagation.flow": {
        "kind": "choice",
        "label": "Lan truyền nhãn: dự đoán chuyển động",
        "choices": {
            "always": "Optical flow ở mọi ảnh (khuyên dùng)",
            "missing": "Optical flow chỉ ở ảnh chưa detect",
            "off": "Đoán theo vận tốc không đổi (cách cũ)",
        },
        "help": "Held-out 20 scene (795 keyframe): flow ở mọi ảnh cho thêm 14% nhãn lan truyền đúng (2910 → 3329), "
        "đổi ID 265 → 97, mất dấu −22%. Tốn ~1 s cho mỗi lần lan truyền 10 keyframe trên CPU.",
        "applies": "now",
    },
    "propagation.association": {
        "kind": "choice",
        "label": "Lan truyền nhãn: cách ghép với detection",
        "choices": {
            "byte": "Hai lượt kiểu ByteTrack — box rõ trước (khuyên dùng)",
            "single": "Một lượt với mọi box (cách cũ)",
        },
        "help": "Box score thấp nằm gần không còn 'cướp' track của vật có box rõ. Dev: box lan truyền sai 224 → 203, "
        "nhãn đúng giữ nguyên; held-out 4 scene: sai 329 → 300.",
        "applies": "now",
    },
    "propagation.oc_recover": {
        "kind": "bool",
        "label": "Lan truyền 2D: nhận lại vật bị che (OC-SORT)",
        "help": "Không khuyên bật. Track đang mất được ghép lại theo box quan sát cuối. 20 scene test: box ghi ra đúng "
        "3192 → 3215 (+0.7%) nhưng đổi ID 76 → 87, box sai 860 → 868.",
        "applies": "now",
    },
    "propagation3d.max_misses": {
        "kind": "int",
        "label": "Lan truyền 3D: số keyframe giữ track khi mất dấu",
        "min": 1,
        "max": 10,
        "help": "Vật bị che quá bấy nhiêu keyframe thì dừng. 2 → 4 (mặc định): test 24 scene nhãn đúng +1.6% "
        "(23355 → 23731), đổi ID 470 → 543.",
        "applies": "now",
    },
    "propagation3d.oc_recover": {
        "kind": "bool",
        "label": "Lan truyền 3D: nhận lại theo vị trí quan sát cuối (OC-SORT)",
        "help": "Tắt mặc định: thêm nhãn đúng trên test (23731 → 23872) nhưng không trên dev, và thêm đổi ID.",
        "applies": "now",
    },
    "propagation.max_keyframes": {
        "kind": "int",
        "label": "Số keyframe tối đa mỗi lần lan truyền",
        "min": 1,
        "max": 200,
        "help": "Lan truyền dừng sau bấy nhiêu keyframe (hoặc trước frame đã có người mở).",
        "applies": "now",
    },
    "qa.temporal.flow": {
        "kind": "bool",
        "label": "So khớp sweep t−2…t+2 bằng optical flow",
        "help": "Không khuyên dùng. Box sweep được dời về thời điểm keyframe trước khi so: cờ FLICKER đúng hơn "
        "(precision 0.79 → 0.90) nhưng bắt ít lỗi hơn hẳn — lỗi lọt qua duyệt theo lô 1632 → 2082 (held-out 20 scene).",
        "applies": "relabel",
    },
    "qa.temporal.rescore": {
        "kind": "choice",
        "label": "Tính lại score keyframe theo các sweep",
        "choices": {
            "off": "Tắt — dùng score detector (mặc định)",
            "mean": "Trung bình 5 ảnh — bớt ~60% box phải xem tay, sót thêm ~9% vật",
            "linked": "Trung bình các lần thấy — ít sót hơn, nhiều box sai hơn",
        },
        "help": "mAP gần như không đổi ở cả ba. Held-out 20 scene, 'Trung bình 5 ảnh' khi TẮT so khớp bằng flow: "
        "box phải xem tay 1707 → 648, box sai 2980 → 1860, tổng lỗi còn lại sau duyệt như cũ (4197 → 4190), "
        "nhưng vật bị sót phải vẽ thêm 2565 → 2791. Bật cùng flow thì lỗi còn lại tăng ~9%.",
        "applies": "relabel",
    },
    "detection.min_score": {
        "kind": "float",
        "label": "Ngưỡng giữ box sau detector",
        "min": 0.05,
        "max": 0.9,
        "step": 0.05,
        "help": "Box dưới ngưỡng bị bỏ trước khi duyệt. Thấp: ít sót, nhiều box phải xoá. Không phải detect lại.",
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
