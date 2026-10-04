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
# label / choices: ngắn, hiện thẳng trên trang. detail: tác dụng của mục và của từng lựa chọn + khuyên dùng, hiện khi rê
# chuột vào ⓘ (mỗi dòng một ý; không ghi số đo eval ở đây, số đo nằm trong eval/report).
# group: tiêu đề nhóm trên trang. applies: now = có hiệu lực ngay | relabel = cần bấm "Áp dụng lại" cho frame cũ
FIELDS: dict[str, dict] = {
    "propagation.flow": {
        "kind": "choice",
        "group": "Lan truyền 2D",
        "label": "Dự đoán chuyển động",
        "choices": {
            "always": "Optical flow",
            "dam4sam": "DAM4SAM",
            "flow+dam4sam": "Optical flow + DAM4SAM",
            "missing": "Optical flow, chỉ ảnh chưa detect",
            "off": "Vận tốc không đổi",
        },
        "help": "",
        "detail": "\n".join(
            [
                "Khi chép nhãn sang frame sau, hệ thống phải đoán vật đã di chuyển tới đâu. Mục này chọn cách đoán.",
                "• Optical flow: dời box theo chuyển động của điểm ảnh. Nhanh, chạy được trên CPU.",
                "• DAM4SAM: mô hình bám theo hình dạng từng vật. Chính xác hơn một chút, chậm hơn nhiều, cần GPU.",
                "• Optical flow + DAM4SAM: dùng optical flow, chỉ gọi DAM4SAM cho vật đang bị che.",
                "• Optical flow, chỉ ảnh chưa detect: chỉ tính ở ảnh chưa có kết quả phát hiện. Nhẹ hơn, kém chính xác hơn.",
                "• Vận tốc không đổi: coi vật đi đều như trước. Nhanh nhất, kém chính xác nhất.",
                "Khuyên dùng: Optical flow.",
            ]
        ),
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
        "help": "",
        "detail": "\n".join(
            [
                "Sau khi đoán vị trí, hệ thống ghép box dự đoán với box mô hình phát hiện ở frame mới để giữ đúng ID của vật.",
                "• ByteTrack: ghép box có độ tin cậy cao trước, rồi dùng box tin cậy thấp cho vật còn thiếu. Giữ được vật mờ, bị che.",
                "• BoT-SORT: như ByteTrack, thêm so màu sắc của vật và bù chuyển động camera.",
                "• IoU một lượt: ghép một lần theo độ chồng lấn giữa các box. Đơn giản nhất, dễ mất vật mờ.",
                "Khuyên dùng: ByteTrack.",
            ]
        ),
        "applies": "now",
    },
    "propagation.oc_recover": {
        "kind": "bool",
        "group": "Lan truyền 2D",
        "label": "Nhận lại vật bị che (OC-SORT)",
        "help": "",
        "detail": "\n".join(
            [
                "Vật bị che rồi hiện lại được nối vào ID cũ theo vị trí lần cuối nhìn thấy.",
                "• Bật: giữ thêm được một ít vật, nhưng dễ gán nhầm ID.",
                "• Tắt: chỉ nối lại theo vị trí dự đoán.",
                "Khuyên dùng: Tắt.",
            ]
        ),
        "applies": "now",
    },
    "propagation.max_keyframes": {
        "kind": "int",
        "group": "Lan truyền 2D",
        "label": "Số keyframe tối đa mỗi lần",
        "min": 1,
        "max": 200,
        "help": "",
        "detail": "\n".join(
            [
                "Mỗi lần lan truyền, nhãn được chép sang tối đa bấy nhiêu keyframe tiếp theo. Lan truyền cũng dừng trước frame đã có người mở.",
                "Số lớn: đỡ phải bấm nhiều lần, nhưng sai lệch dồn lại ở các frame xa.",
            ]
        ),
        "applies": "now",
    },
    "propagation3d.max_misses": {
        "kind": "int",
        "group": "Lan truyền 3D",
        "label": "Số keyframe giữ track khi mất dấu",
        "min": 1,
        "max": 10,
        "help": "",
        "detail": "\n".join(
            [
                "Số keyframe liên tiếp một vật 3D còn được theo dõi khi không thấy nữa (bị che). Quá số này thì dừng theo dõi vật đó.",
                "Số lớn: giữ được vật bị che lâu, nhưng dễ nối nhầm sang vật khác.",
            ]
        ),
        "applies": "now",
    },
    "propagation3d.oc_recover": {
        "kind": "bool",
        "group": "Lan truyền 3D",
        "label": "Nhận lại theo vị trí cuối (OC-SORT)",
        "help": "",
        "detail": "\n".join(
            [
                "Vật 3D bị che rồi hiện lại được nối vào ID cũ theo vị trí lần cuối nhìn thấy.",
                "• Bật: giữ thêm được một ít vật, nhưng dễ gán nhầm ID.",
                "• Tắt: chỉ nối lại theo vị trí dự đoán.",
                "Khuyên dùng: Tắt.",
            ]
        ),
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
        "help": "",
        "detail": "\n".join(
            [
                "Mô hình tìm vật trên ảnh.",
                "• Fine-tune full: huấn luyện lại toàn bộ mô hình trên ảnh đường phố. Chính xác nhất, chỉ nhận 10 lớp nuScenes.",
                "• Fine-tune mini: chỉ huấn luyện lại lớp cuối. Kém bản full, cũng chỉ nhận 10 lớp.",
                "• Original: YOLOE gốc. Kém chính xác hơn, nhưng nhận được lớp mới bằng prompt chữ.",
                "Khuyên dùng: Fine-tune full. Cần lớp ngoài 10 lớp nuScenes thì chọn Original.",
            ]
        ),
        "applies": "relabel",
    },
    "detection.min_score": {
        "kind": "float",
        "group": "Gán nhãn",
        "label": "Ngưỡng giữ box",
        "min": 0.05,
        "max": 0.9,
        "step": 0.05,
        "help": "",
        "detail": "\n".join(
            [
                "Box có độ tin cậy dưới ngưỡng này bị bỏ.",
                "• Ngưỡng thấp: ít sót vật, nhiều box sai phải xoá.",
                "• Ngưỡng cao: ít box sai, dễ sót vật.",
            ]
        ),
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
        "help": "",
        "detail": "\n".join(
            [
                "Tính lại độ tin cậy của mỗi box dựa trên các ảnh ngay trước và sau keyframe: vật xuất hiện đều ở các ảnh đó thì được tin hơn.",
                "• Tắt: giữ độ tin cậy của mô hình.",
                "• Trung bình 5 ảnh: lấy trung bình trên 5 ảnh quanh keyframe. Ít box sai hơn, dễ sót vật hơn.",
                "• Trung bình các lần thấy: chỉ tính những ảnh có thấy vật. Ít sót hơn, nhiều box sai hơn.",
                "Khuyên dùng: Tắt.",
            ]
        ),
        "applies": "relabel",
    },
    "qa.temporal.flow": {
        "kind": "bool",
        "group": "Gán nhãn",
        "label": "So khớp sweep bằng optical flow",
        "help": "",
        "detail": "\n".join(
            [
                "Khi kiểm tra một box có xuất hiện đều ở các ảnh lân cận không, dời box về đúng thời điểm keyframe bằng optical flow trước khi so.",
                "• Bật: cảnh báo nhấp nháy chính xác hơn, nhưng bỏ lọt nhiều lỗi hơn.",
                "• Tắt: so trực tiếp vị trí box.",
                "Khuyên dùng: Tắt.",
            ]
        ),
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
