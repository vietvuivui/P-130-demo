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
        "help": "Held-out 4 scene: flow ở mọi ảnh cho thêm 11% nhãn lan truyền đúng, đổi ID 26 → 11. "
        "Tốn ~0.05–0.1 s mỗi ảnh 12Hz trên CPU. Áp dụng ngay cho lần lan truyền sau.",
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
        "help": "Box sweep được dời về thời điểm keyframe trước khi so: cờ FLICKER ít và đúng hơn "
        "(precision 0.84 → 0.94) nhưng bắt ít lỗi hơn (lỗi lọt qua duyệt theo lô 277 → 327 trên held-out).",
        "applies": "relabel",
    },
    "qa.temporal.rescore": {
        "kind": "choice",
        "label": "Tính lại score keyframe theo các sweep",
        "choices": {
            "off": "Tắt — dùng score detector (mặc định)",
            "mean": "Trung bình 5 ảnh — bớt ~60% box phải xem tay, lỗi còn lại +5%",
            "linked": "Trung bình các lần thấy — thêm recall, thêm box sai",
        },
        "help": "mAP gần như không đổi ở cả ba; đây là đổi công duyệt lấy chất lượng. Nên bật cùng so khớp bằng flow.",
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
