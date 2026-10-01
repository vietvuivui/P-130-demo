"""Schema cho configs/autolabel.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ClassSpec(BaseModel):
    prompts: list[str]
    height_m: tuple[float, float]
    aspect_wh: tuple[float, float]
    color: str = "#64748b"


class Yolo26Cfg(BaseModel):
    # YOLO26 thường: tập lớp đóng COCO 80, chỉ giữ lớp trùng tên prompt (car, truck, bus, person, bicycle, motorcycle)
    weights: str = "weights/yolo26l.pt"
    imgsz: int = 1280


class YoloE26Cfg(BaseModel):
    weights: str = "weights/yoloe-26l-seg.pt"
    imgsz: int = 1280
    # Text encoder mã hoá prompt (MobileCLIP2-B, TorchScript) mà YOLOE-26 được train cùng
    text_encoder: str = "weights/mobileclip2_b.ts"


class YoloWorldCfg(BaseModel):
    weights: str = "yolov8l-worldv2.pt"
    imgsz: int = 1280


class YoloeCfg(BaseModel):
    # YOLOE-26: open-vocab trên nền YOLO26, text encoder MobileCLIP2 (tự tải từ GitHub của Ultralytics)
    weights: str = "yoloe-26l-seg.pt"
    imgsz: int = 1280
    # Chạy thêm ảnh lật ngang rồi gộp (augment=True của Ultralytics không có tác dụng với YOLOE): ~2x thời gian
    tta_flip: bool = False
    # Lưu mask segmentation sơ bộ của model -seg (đa giác đã đơn giản hoá), FR-04
    masks: bool = True


class GroundingDinoCfg(BaseModel):
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.2
    text_threshold: float = 0.2


class Florence2Cfg(BaseModel):
    model_id: str = "microsoft/Florence-2-base"
    default_score: float = 0.5


class DemoDetectorCfg(BaseModel):
    # Detector theo màu cho video demo (python -m src.demo), không cần GPU
    tolerance: int = 48
    min_area: int = 60
    confident_area: int = 3500


class Lidar3DCfg(BaseModel):
    """Box 3D của ensemble LiDAR chiếu xuống ảnh làm nguồn nhãn 2D (src/services/lidar2d.py)."""

    enabled: bool = True
    min_score: float = 0.05  # box 3D yếu hơn không chiếu
    # Box 2D trùng hình chiếu box 3D cùng lớp từ ngưỡng này: lấy box 3D. 0.4 chọn trên dev trong {0.5, 0.4, 0.3}: vật nhỏ
    # (cọc tiêu, người) hai nguồn lệch vài px nên 0.5 để sót nhiều cặp trùng; 0.3 ghép nhầm sang cọc bên cạnh.
    match_iou: float = 0.4
    camera_only_scale: float = 0.5  # hệ số điểm của box chỉ detector ảnh thấy


class DetectionCfg(BaseModel):
    detectors: list[str] = ["yoloe"]
    score_threshold: float = 0.1
    # Lọc sau khi gộp box, trước QA Agent (không đổi khoá cache: đổi ngưỡng không phải detect lại)
    min_score: float = 0.1
    min_score_per_class: dict[str, float] = Field(default_factory=dict)
    fusion_iou: float = 0.55
    lidar3d: Lidar3DCfg = Lidar3DCfg()
    yolo26: Yolo26Cfg = Yolo26Cfg()
    yoloe26: YoloE26Cfg = YoloE26Cfg()
    yolo_world: YoloWorldCfg = YoloWorldCfg()
    yoloe: YoloeCfg = YoloeCfg()
    grounding_dino: GroundingDinoCfg = GroundingDinoCfg()
    florence2: Florence2Cfg = Florence2Cfg()
    demo: DemoDetectorCfg = DemoDetectorCfg()

    def keep(self, label: str, score: float) -> bool:
        return score >= self.min_score_per_class.get(label, self.min_score)


class ConfidenceCfg(BaseModel):
    low_threshold: float = 0.35
    class_conflict_ratio: float = 0.6


class LidarCfg(BaseModel):
    min_points: int = 3
    min_box_height_px: float = 60
    min_depth_m: float = 1.0
    max_depth_m: float = 45.0
    size_tolerance: tuple[float, float] = (0.5, 1.5)


class TemporalCfg(BaseModel):
    match_iou: float = 0.3
    min_support: int = 2
    recover_min_score: float = 0.35
    # Dời box của sweep về thời điểm keyframe bằng optical flow trước khi so khớp (src/services/flow.py)
    flow: bool = False
    flow_scale: float = 0.5
    # Tính lại score keyframe theo các sweep (src/services/temporal_fusion.py): off | mean | linked
    rescore: Literal["off", "mean", "linked"] = "off"
    # Ngưỡng score cho box ở sweep khi làm bằng chứng cho keyframe (kiểu ByteTrack: ngưỡng cao để tạo box, ngưỡng
    # thấp để xác nhận vật đã thấy ở keyframe). None = như detection.min_score
    sweep_min_score: float | None = None


class GeometryCfg(BaseModel):
    max_area_ratio: float = 0.35
    border_margin_px: float = 4


class RiskWeights(BaseModel):
    detection: float = 0.35
    lidar: float = 0.30
    temporal: float = 0.20
    geometric: float = 0.15


class RiskLevels(BaseModel):
    medium: float = 0.30
    high: float = 0.60


class RiskCfg(BaseModel):
    weights: RiskWeights = RiskWeights()
    class_conflict_penalty: float = 0.30
    issue_floor: float = 0.30
    levels: RiskLevels = RiskLevels()


class QACfg(BaseModel):
    confidence: ConfidenceCfg = ConfidenceCfg()
    lidar: LidarCfg = LidarCfg()
    temporal: TemporalCfg = TemporalCfg()
    geometry: GeometryCfg = GeometryCfg()
    risk: RiskCfg = RiskCfg()


class PropagationWeights(BaseModel):
    agreement: float = 0.5
    continuity: float = 0.3
    lidar: float = 0.2


class PropagationCfg(BaseModel):
    match_iou: float = 0.3
    suppress_iou: float = 0.5
    smoothing: float = 0.7
    velocity_smoothing: float = 0.5
    max_coast_images: int = 6
    min_visible: float = 0.3
    min_box_px: float = 6
    weights: PropagationWeights = PropagationWeights()
    hop_decay: float = 0.99
    flag_below: float = 0.6
    stop_below: float = 0.15
    # Track không khớp detection ở chính keyframe đích (đang "trôi" theo vận tốc) có được ghi ra không
    emit_coasting: bool = False
    # Optical flow cho tracker lan truyền: off (dự đoán theo vận tốc) | missing (chỉ ảnh chưa có detection) | always
    # dam4sam: thay flow bằng DAM4SAM (SAM 2.1, cần GPU; src/services/dam4sam.py): mỗi vật được phân đoạn ở từng ảnh,
    # box dự đoán = hộp bao mask; detection vẫn dùng để ghép / dừng track như thường
    flow: Literal["off", "missing", "always", "dam4sam"] = "always"
    flow_scale: float = 0.5
    dam4sam_model: str = "sam21pp-L"
    # Ghép track với detection: single = một lượt với mọi box ≥ score_threshold (như trước) | byte = hai lượt kiểu
    # ByteTrack (Zhang et al. 2022): box score ≥ byte_high_score trước, box score thấp chỉ cho track còn thiếu (IoU chặt
    # hơn byte_low_iou), để box score thấp nằm gần không "cướp" track của vật có box rõ
    association: Literal["single", "byte", "botsort"] = "byte"
    byte_high_score: float = 0.3
    byte_low_iou: float = 0.5
    # BoT-SORT (Aharon et al. 2022, src/services/botsort.py) = ByteTrack + bù chuyển động camera (GMC, chỉ dùng khi
    # không có optical flow) + ngoại hình: chi phí = min(1 - IoU, khoảng cách ngoại hình), ngoại hình chỉ tính khi hai
    # box gần nhau (1 - IoU <= botsort_proximity) và đủ giống (<= botsort_appearance); đặc trưng track cập nhật EMA
    botsort_gmc: bool = True
    botsort_proximity: float = 0.5
    botsort_appearance: float = 0.3
    botsort_alpha: float = 0.9
    # OC-SORT (Cao et al., CVPR 2023) — chỉ dựa vào quan sát thật (detection đã khớp), không vào box dự đoán lúc bị che:
    # - oc_recover (OCR): sau các lượt ghép, track đang mất ghép thêm với detection còn thừa theo box quan sát cuối
    #   (và box quan sát cuối dời theo vận tốc quan sát); cho track sống tới oc_max_lost ảnh thay vì max_coast_images
    # - oc_reupdate (ORU): ghép lại sau khi mất thì lấy vận tốc theo đường nối quan sát cuối -> quan sát mới, box = detection
    # - oc_momentum (OCM): cộng oc_momentum x độ cùng hướng (cos) giữa hướng đi đã quan sát và hướng tới detection
    oc_recover: bool = False
    oc_reupdate: bool = False
    oc_momentum: float = 0.0
    oc_max_lost: int = 6
    oc_recover_iou: float = 0.5
    oc_delta: int = 3
    max_keyframes: int = 20
    class_differs_score: float = 0.5


class Propagation3DCfg(BaseModel):
    """Lan truyền box 3D đã duyệt sang keyframe sau (src/services/propagation3d.py)."""

    # velocity: dịch box theo vận tốc mô hình dự đoán (hệ toàn cục, đã bù chuyển động xe) | none: chỉ bù chuyển động xe
    motion: Literal["velocity", "none"] = "velocity"
    # Nhân ngưỡng khoảng cách khớp theo lớp của tracker CenterPoint
    dist_scale: float = 0.5
    # Không khớp quá bấy nhiêu keyframe liên tiếp thì dừng track (4: chọn khi thử OC-SORT, eval/results/ocsort.md)
    max_misses: int = 4
    max_keyframes: int = 20
    # OC-SORT như lan truyền 2D: OCR ghép lại track đang mất theo tâm quan sát cuối (dời theo vận tốc quan sát), sống tới
    # oc_max_lost keyframe; ORU lấy vận tốc theo quan sát cuối -> quan sát mới khi mô hình không cho vận tốc; OCM ưu tiên
    # detection cùng hướng đi đã quan sát
    oc_recover: bool = False
    oc_reupdate: bool = False
    oc_momentum: float = 0.0
    oc_max_lost: int = 4
    oc_recover_scale: float = 1.0  # ngưỡng khoảng cách của OCR = ngưỡng khớp x hệ số này


class VideoCfg(BaseModel):
    # Video mp4 tải lên: cắt frame ở track_fps để tracking, cứ track_fps/label_fps frame có một keyframe để gán nhãn
    track_fps: float = 10.0
    label_fps: float = 2.0
    max_seconds: float = 120.0
    max_upload_mb: int = 500
    # Check temporal (FLICKER, RECOVERED_BY_TRACK) chỉ dùng frame lân cận cách keyframe không quá ngần này (giây).
    # Ngưỡng QA temporal được chỉnh cho sweep 12Hz (±167 ms); video thưa (vd. chỉ có keyframe 2 Hz) thì bỏ check
    # thay vì báo FLICKER sai cho mọi vật đang chuyển động
    max_sweep_gap_s: float = 0.25


class QuickCheckCfg(BaseModel):
    max_file_mb: int = 20
    missing_min_score: float = 0.5
    missing_match_iou: float = 0.3
    disagree_min_score: float = 0.6


class AuditCfg(BaseModel):
    # Mẫu object duyệt theo lô (lỗi nhãn) và mẫu frame (vật bị sót). sample_size = 0: không bắt buộc audit loại đó
    object_sample_size: int = Field(default=50, ge=0)
    object_max_error_upper: float = 0.10
    frame_sample_size: int = Field(default=10, ge=0)
    frame_max_error_upper: float = 0.30
    confidence_z: float = 1.96


class QCCfg(BaseModel):
    gate_on_approve: bool = True
    min_box_px: float = 2
    duplicate_iou: float = 0.7
    cross_class_iou: float = 0.85
    cross_class_allowed: list[tuple[str, str]] = [("pedestrian", "bicycle"), ("pedestrian", "motorcycle")]
    quick_check: QuickCheckCfg = QuickCheckCfg()
    audit: AuditCfg = AuditCfg()


class Verify3DCfg(BaseModel):
    # Box 3D dưới ngưỡng điểm này không đưa vào duyệt (như ngưỡng 0.3 nhóm 3D dùng)
    min_score: float = 0.3
    # Ngưỡng điểm của box 2D dùng để kiểm chứng
    det_conf: float = 0.2
    # Prompt "đối thủ": vật dễ nhầm với các lớp (lan can cố định, cột, biển báo...). Box của chúng bị bỏ
    distractors: list[str] = Field(
        default_factory=lambda: [
            "fence", "guardrail", "pole", "traffic sign", "fire hydrant", "trash can", "bollard",
            "mailbox", "stroller", "wheelchair", "kick scooter",
        ]
    )  # fmt: skip
    # Số lần quét LiDAR liền trước gộp thêm (nếu có trên đĩa) khi đếm điểm trong box / tính mức che
    lidar_sweeps: int = 4
    # Số điểm tối đa gửi lên UI mỗi keyframe
    max_points_ui: int = 60000


class PrivacyCfg(BaseModel):
    """Làm mờ mặt người và biển số trước khi ảnh tới trình duyệt / file xuất (FR-03), src/services/privacy.py."""

    enabled: bool = True
    # Biển số: detector chuyên dụng (open-image-models, YOLOv9 ONNX, tự tải ~8 MB), chạy cả ảnh và 4 ô cắt để bắt
    # biển số nhỏ ở xa
    plate_model: str = "yolo-v9-t-640-license-plate-end2end"
    plate_conf: float = 0.25
    plate_tiles: bool = True
    # Mặt người: YOLOE (prompt "human face") + vùng đầu của mỗi người đủ lớn (đầu = 18% trên của box người):
    # ưu tiên không sót mặt hơn là làm mờ thừa
    face_prompts: list[str] = ["human face"]
    face_conf: float = 0.25
    person_conf: float = 0.3
    min_person_px: int = 40  # người thấp hơn thế: mặt quá nhỏ để nhận ra, không cần làm mờ
    pad: float = 0.2  # nới vùng làm mờ ra mỗi phía theo tỉ lệ kích thước
    imgsz: int = 1280


class AutoLabelConfig(BaseModel):
    camera: str = "CAM_FRONT"
    sweep_offsets: list[int] = [-2, -1, 1, 2]
    detection: DetectionCfg = DetectionCfg()
    classes: dict[str, ClassSpec] = Field(default_factory=dict)
    gt_category_map: dict[str, str] = Field(default_factory=dict)
    qa: QACfg = QACfg()
    qc: QCCfg = QCCfg()
    propagation: PropagationCfg = PropagationCfg()
    propagation3d: Propagation3DCfg = Propagation3DCfg()
    video: VideoCfg = VideoCfg()
    verify3d: Verify3DCfg = Verify3DCfg()
    privacy: PrivacyCfg = PrivacyCfg()

    def prompt_to_class(self) -> dict[str, str]:
        """Map mỗi prompt văn bản về lớp nội bộ."""
        return {p.lower(): name for name, spec in self.classes.items() for p in spec.prompts}


def load_autolabel_config(path: str | Path) -> AutoLabelConfig:
    with open(path, encoding="utf-8") as f:
        return AutoLabelConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_autolabel_config(path: str) -> AutoLabelConfig:
    return load_autolabel_config(path)
