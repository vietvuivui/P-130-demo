"""Loader gọn cho nuScenes (không cần nuscenes-devkit).

Cung cấp đúng những gì pipeline 2D cần: keyframe camera, sweep 12Hz lân cận,
point cloud LiDAR chiếu xuống ảnh, và GT 2D chiếu từ box 3D để đánh giá.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from src.services.geometry import apply_transform, clip_box, project_points, quaternion_to_matrix, transform_matrix

# visibility token "1" là mức 0-40%: object gần như bị che hết, không tính là GT bắt buộc
LOW_VISIBILITY_TOKENS = {"1"}


@dataclass
class ImageRef:
    sd_token: str
    path: str
    timestamp: int
    width: int
    height: int


@dataclass(frozen=True)
class TimelineImage:
    """Một ảnh trong chuỗi camera của scene (keyframe 2Hz hoặc sweep 12Hz)."""

    sd_token: str
    timestamp: int  # micro giây
    # Chỉ keyframe mới có sample_token: đó là những ảnh có FrameRecord để ghi nhãn
    sample_token: str | None = None
    path: str = ""

    @property
    def is_keyframe(self) -> bool:
        return self.sample_token is not None


@dataclass
class CameraFrame:
    sample_token: str
    scene: str
    index: int
    camera: str
    image: ImageRef
    intrinsic: np.ndarray
    # offset -> ảnh sweep; offset âm là quá khứ
    sweeps: dict[int, ImageRef] = field(default_factory=dict)
    lidar_sd_token: str | None = None

    @property
    def frame_id(self) -> str:
        return f"{self.scene}_{self.index:03d}"


class NuScenesMini:
    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini"):
        self.dataroot = Path(dataroot)
        self.table_dir = self.dataroot / version
        if not self.table_dir.is_dir():
            raise FileNotFoundError(f"Không thấy bảng nuScenes ở {self.table_dir}")

    def _load(self, name: str) -> dict[str, dict]:
        with open(self.table_dir / f"{name}.json", encoding="utf-8") as f:
            return {row["token"]: row for row in json.load(f)}

    @cached_property
    def scene(self) -> dict[str, dict]:
        return self._load("scene")

    @cached_property
    def sample(self) -> dict[str, dict]:
        return self._load("sample")

    @cached_property
    def sample_data(self) -> dict[str, dict]:
        return self._load("sample_data")

    @cached_property
    def calibrated_sensor(self) -> dict[str, dict]:
        return self._load("calibrated_sensor")

    @cached_property
    def ego_pose(self) -> dict[str, dict]:
        return self._load("ego_pose")

    @cached_property
    def sample_annotation(self) -> dict[str, dict]:
        return self._load("sample_annotation")

    @cached_property
    def category_of_instance(self) -> dict[str, str]:
        category = self._load("category")
        return {tok: category[row["category_token"]]["name"] for tok, row in self._load("instance").items()}

    @cached_property
    def _channel_of_calib(self) -> dict[str, str]:
        sensor = self._load("sensor")
        return {tok: sensor[row["sensor_token"]]["channel"] for tok, row in self.calibrated_sensor.items()}

    @cached_property
    def _keyframe_data(self) -> dict[str, dict[str, str]]:
        """sample_token -> {channel: sample_data token của keyframe}."""
        out: dict[str, dict[str, str]] = {}
        for tok, sd in self.sample_data.items():
            if sd["is_key_frame"]:
                channel = self._channel_of_calib[sd["calibrated_sensor_token"]]
                out.setdefault(sd["sample_token"], {})[channel] = tok
        return out

    @cached_property
    def _annotations_of_sample(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for ann in self.sample_annotation.values():
            out.setdefault(ann["sample_token"], []).append(ann)
        return out

    def keyframes(self, scene_names: list[str] | None = None) -> list[tuple[str, int, str]]:
        """Danh sách (tên scene, thứ tự trong scene, sample_token) theo thời gian."""
        out = []
        for sc in sorted(self.scene.values(), key=lambda s: s["name"]):
            if scene_names and sc["name"] not in scene_names:
                continue
            tok, idx = sc["first_sample_token"], 0
            while tok:
                out.append((sc["name"], idx, tok))
                tok = self.sample[tok]["next"]
                idx += 1
        return out

    def _image_ref(self, sd_token: str) -> ImageRef:
        sd = self.sample_data[sd_token]
        return ImageRef(sd_token, sd["filename"], sd["timestamp"], sd["width"], sd["height"])

    def camera_frame(
        self, sample_token: str, camera: str, sweep_offsets: list[int], scene: str, index: int
    ) -> CameraFrame:
        data = self._keyframe_data[sample_token]
        cam_tok = data[camera]
        cs = self.calibrated_sensor[self.sample_data[cam_tok]["calibrated_sensor_token"]]
        frame = CameraFrame(
            sample_token=sample_token,
            scene=scene,
            index=index,
            camera=camera,
            image=self._image_ref(cam_tok),
            intrinsic=np.asarray(cs["camera_intrinsic"], dtype=np.float64),
            lidar_sd_token=data.get("LIDAR_TOP"),
        )
        for offset in sweep_offsets:
            tok, key = cam_tok, "next" if offset > 0 else "prev"
            for _ in range(abs(offset)):
                tok = self.sample_data[tok][key] if tok else ""
            if tok:
                frame.sweeps[offset] = self._image_ref(tok)
        return frame

    def camera_timeline(self, scene_name: str, camera: str) -> list[TimelineImage]:
        """Mọi ảnh của một camera trong scene theo thời gian: keyframe (2Hz) xen kẽ sweep (12Hz).

        Đi theo chuỗi `next` của sample_data bắt đầu từ keyframe đầu tiên, dừng khi sang scene khác.
        """
        sc = next((s for s in self.scene.values() if s["name"] == scene_name), None)
        if sc is None:
            raise KeyError(f"Không có scene {scene_name}")
        tok = self._keyframe_data[sc["first_sample_token"]][camera]
        out: list[TimelineImage] = []
        while tok:
            sd = self.sample_data.get(tok)  # bảng đã lọc (scripts/pack_nuscenes_subset.py) có thể cắt chuỗi next
            if sd is None or self.sample[sd["sample_token"]]["scene_token"] != sc["token"]:
                break
            out.append(
                TimelineImage(
                    sd_token=tok,
                    timestamp=sd["timestamp"],
                    sample_token=sd["sample_token"] if sd["is_key_frame"] else None,
                    path=sd["filename"],
                )
            )
            tok = sd["next"]
        return out

    def _ego_from_sensor(self, sd_token: str) -> np.ndarray:
        cs = self.calibrated_sensor[self.sample_data[sd_token]["calibrated_sensor_token"]]
        return transform_matrix(cs["translation"], cs["rotation"])

    def _global_from_ego(self, sd_token: str) -> np.ndarray:
        pose = self.ego_pose[self.sample_data[sd_token]["ego_pose_token"]]
        return transform_matrix(pose["translation"], pose["rotation"])

    def cam_from_global(self, cam_sd_token: str) -> np.ndarray:
        return np.linalg.inv(self._ego_from_sensor(cam_sd_token)) @ np.linalg.inv(self._global_from_ego(cam_sd_token))

    def load_lidar(self, lidar_sd_token: str) -> np.ndarray:
        path = self.dataroot / self.sample_data[lidar_sd_token]["filename"]
        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)

    def lidar_in_image(self, frame: CameraFrame, min_depth: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Chiếu LiDAR keyframe xuống ảnh camera, có bù chuyển động ego giữa hai timestamp.

        Trả về (uv (N, 2), depth (N,)) chỉ gồm điểm rơi trong ảnh.
        """
        if not frame.lidar_sd_token:
            return np.zeros((0, 2)), np.zeros(0)
        pts = self.load_lidar(frame.lidar_sd_token)
        cam_from_lidar = (
            self.cam_from_global(frame.image.sd_token)
            @ self._global_from_ego(frame.lidar_sd_token)
            @ self._ego_from_sensor(frame.lidar_sd_token)
        )
        return points_to_image(pts, cam_from_lidar, frame.intrinsic, frame.image.width, frame.image.height, min_depth)

    def gt_boxes_2d(self, frame: CameraFrame, category_map: dict[str, str]) -> list[dict]:
        """GT 2D = hộp bao của 8 đỉnh box 3D chiếu xuống ảnh (giống export_2d_annotations_as_json của devkit).

        Object mức visibility 0-40% hoặc không có điểm LiDAR/radar được giữ lại với ignore=True,
        để khi đánh giá không phạt detector vì thấy chúng.
        """
        cam_from_global = self.cam_from_global(frame.image.sd_token)
        out = []
        for ann in self._annotations_of_sample.get(frame.sample_token, []):
            label = category_map.get(self.category_of_instance[ann["instance_token"]])
            if label is None:
                continue
            corners = box_corners(ann["translation"], ann["size"], ann["rotation"])
            corners_cam = apply_transform(cam_from_global, corners)
            in_front = corners_cam[:, 2] > 0.1
            if not in_front.any():
                continue
            uv = project_points(corners_cam[in_front], frame.intrinsic)
            bbox = clip_box(
                [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], frame.image.width, frame.image.height
            )
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                continue
            ignore = (
                ann["visibility_token"] in LOW_VISIBILITY_TOKENS or ann["num_lidar_pts"] + ann["num_radar_pts"] == 0
            )
            out.append(
                {
                    "bbox": [round(v, 1) for v in bbox],
                    "label": label,
                    "instance_token": ann["instance_token"],
                    "visibility": ann["visibility_token"],
                    "num_lidar_pts": ann["num_lidar_pts"],
                    "depth_m": round(
                        float(np.linalg.norm(apply_transform(cam_from_global, np.array([ann["translation"]]))[0])), 2
                    ),
                    "ignore": bool(ignore),
                }
            )
        return out


def box_corners(center, size_wlh, rotation) -> np.ndarray:
    """8 đỉnh box 3D theo quy ước nuScenes (size = [w, l, h]) -> (8, 3)."""
    w, length, h = size_wlh
    x = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return (quaternion_to_matrix(rotation) @ np.vstack([x, y, z])).T + np.asarray(center)


def points_to_image(
    points: np.ndarray,
    cam_from_points: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = apply_transform(cam_from_points, points)
    depth = pts_cam[:, 2]
    keep = depth > min_depth
    uv = project_points(pts_cam[keep], intrinsic)
    depth = depth[keep]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv[inside], depth[inside]
