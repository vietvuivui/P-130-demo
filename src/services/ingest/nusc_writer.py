"""Ghi một chuỗi frame (LiDAR + camera + tư thế xe) thành dataset định dạng nuScenes để dùng lại toàn bộ pipeline.

Quy ước (giống nuScenes):
- hệ ego: gốc dưới trục sau, x trước, y trái, z lên; tư thế ego_pose = ego -> toàn cục;
- file LiDAR lưu trong hệ LIDAR_TOP kiểu nuScenes (x sang phải, y về trước, z lên), float32 x y z cường độ(0-255) ring.
  Mô hình 3D học trên đúng hệ trục này, nên điểm của nguồn khác (KITTI: x trước, y trái) được xoay trước khi ghi;
- mỗi frame là một keyframe (sample); LiDAR của các frame trước làm sweep (theo chuỗi prev), như nuScenes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATIC = json.loads((Path(__file__).parent / "nuscenes_static.json").read_text(encoding="utf-8"))
NUSC_CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# hệ LiDAR kiểu nuScenes (x phải, y trước) từ hệ "x trước, y trái": x' = -y, y' = x
R_NUSC_FROM_XFWD = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])


def token() -> str:
    return uuid.uuid4().hex


def quat_from_matrix(m: np.ndarray) -> list[float]:
    """Ma trận quay 3x3 -> quaternion [w, x, y, z] (thuận tay phải, như pyquaternion)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    return [round(float(v), 8) for v in (q if q[0] >= 0 else -q)]


@dataclass
class CameraInput:
    image: Path  # ảnh gốc (jpg/png)
    intrinsic: np.ndarray  # 3x3
    cam_from_lidar: np.ndarray  # 4x4, lidar = hệ LiDAR gốc của nguồn


@dataclass
class FrameInput:
    timestamp_s: float
    points: np.ndarray  # (N, >=3) trong hệ LiDAR gốc của nguồn; cột 4 (nếu có) là cường độ
    ego_from_lidar: np.ndarray  # 4x4, hệ LiDAR gốc -> ego
    lidar_axes: str = "x_forward"  # hệ LiDAR gốc: "x_forward" (KITTI, đa số) hoặc "x_right" (nuScenes)
    cameras: dict[str, CameraInput] = field(default_factory=dict)
    global_from_ego: np.ndarray | None = None  # 4x4; None = đứng yên tại gốc
    name: str = ""


class NuScenesWriter:
    """Tích luỹ scene/frame rồi ghi bảng JSON + file cảm biến vào dataroot/<version>/."""

    def __init__(self, dataroot: Path, version: str = "v1.0-custom", location: str = "custom"):
        self.root = Path(dataroot)
        self.version = version
        self.location = location
        self.t = {k: [] for k in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "log",
                                  "map", "instance", "sample_annotation")}  # fmt: skip
        self.sensor_tokens: dict[str, str] = {}
        self.calib_cache: dict[tuple, str] = {}
        self.log_token = token()
        self.t["log"].append(
            dict(token=self.log_token, logfile="", vehicle="custom", date_captured="", location=location)
        )

    def _sensor(self, channel: str) -> str:
        if channel not in self.sensor_tokens:
            tok = token()
            self.sensor_tokens[channel] = tok
            self.t["sensor"].append(
                dict(token=tok, channel=channel, modality="lidar" if "LIDAR" in channel else "camera")
            )
        return self.sensor_tokens[channel]

    def _calib(self, channel: str, ego_from_sensor: np.ndarray, intrinsic: np.ndarray | None) -> str:
        key = (
            channel,
            tuple(np.round(ego_from_sensor, 6).ravel()),
            None if intrinsic is None else tuple(np.round(intrinsic, 4).ravel()),
        )
        if key not in self.calib_cache:
            tok = token()
            self.calib_cache[key] = tok
            self.t["calibrated_sensor"].append(dict(
                token=tok, sensor_token=self._sensor(channel),
                translation=[round(float(v), 6) for v in ego_from_sensor[:3, 3]],
                rotation=quat_from_matrix(ego_from_sensor[:3, :3]),
                camera_intrinsic=[] if intrinsic is None else np.round(intrinsic, 6).tolist(),
            ))  # fmt: skip
        return self.calib_cache[key]

    def add_scene(self, name: str, frames: list[FrameInput], description: str = "") -> str:
        from PIL import Image

        scene_tok = token()
        sample_toks = [token() for _ in frames]
        prev_sd: dict[str, str] = {}
        sd_by_channel: dict[str, list[dict]] = {}
        for i, (f, stok) in enumerate(zip(frames, sample_toks, strict=True)):
            ts = int(round(f.timestamp_s * 1e6))
            g = np.eye(4) if f.global_from_ego is None else np.asarray(f.global_from_ego, float)
            pose_tok = token()
            self.t["ego_pose"].append(dict(token=pose_tok, timestamp=ts, translation=[round(float(v), 6) for v in g[:3, 3]],
                                           rotation=quat_from_matrix(g[:3, :3])))  # fmt: skip
            self.t["sample"].append(dict(token=stok, timestamp=ts, scene_token=scene_tok,
                                         prev=sample_toks[i - 1] if i else "", next=sample_toks[i + 1] if i + 1 < len(frames) else ""))  # fmt: skip
            # LiDAR -> hệ nuScenes
            src = np.asarray(f.points, np.float32)
            r_nl = R_NUSC_FROM_XFWD if f.lidar_axes == "x_forward" else np.eye(3)
            xyz = src[:, :3] @ r_nl.T
            inten = src[:, 3] if src.shape[1] > 3 else np.zeros(len(src), np.float32)
            if len(inten) and inten.max() <= 1.0:
                inten = inten * 255.0
            pts = np.zeros((len(src), 5), np.float32)
            pts[:, :3], pts[:, 3] = xyz, inten
            nusc_to_src = np.eye(4)
            nusc_to_src[:3, :3] = r_nl.T  # điểm hệ nuScenes -> hệ LiDAR gốc
            ego_from_lidar_nusc = np.asarray(f.ego_from_lidar, float) @ nusc_to_src
            rel = f"samples/LIDAR_TOP/{name}__LIDAR_TOP__{ts}.pcd.bin"
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            pts.tofile(self.root / rel)
            entries = [("LIDAR_TOP", rel, self._calib("LIDAR_TOP", ego_from_lidar_nusc, None), 0, 0)]
            for ch, cam in f.cameras.items():
                img = Image.open(cam.image).convert("RGB")
                rel_c = f"samples/{ch}/{name}__{ch}__{ts}.jpg"
                (self.root / rel_c).parent.mkdir(parents=True, exist_ok=True)
                img.save(self.root / rel_c, quality=92)
                ego_from_cam = np.asarray(f.ego_from_lidar, float) @ np.linalg.inv(
                    np.asarray(cam.cam_from_lidar, float)
                )
                entries.append(
                    (ch, rel_c, self._calib(ch, ego_from_cam, np.asarray(cam.intrinsic, float)), img.width, img.height)
                )
            for ch, rel_f, calib, w, h in entries:
                sd = dict(token=token(), sample_token=stok, ego_pose_token=pose_tok, calibrated_sensor_token=calib,
                          timestamp=ts, fileformat="pcd" if ch == "LIDAR_TOP" else "jpg", is_key_frame=True,
                          height=h, width=w, filename=rel_f, prev=prev_sd.get(ch, ""), next="")  # fmt: skip
                if ch in prev_sd:
                    sd_by_channel[ch][-1]["next"] = sd["token"]
                sd_by_channel.setdefault(ch, []).append(sd)
                prev_sd[ch] = sd["token"]
        for sds in sd_by_channel.values():
            self.t["sample_data"].extend(sds)
        self.t["scene"].append(dict(token=scene_tok, log_token=self.log_token, nbr_samples=len(frames),
                                    first_sample_token=sample_toks[0], last_sample_token=sample_toks[-1],
                                    name=name, description=description))  # fmt: skip
        return scene_tok

    def write(self) -> Path:
        out = self.root / self.version
        out.mkdir(parents=True, exist_ok=True)
        tables = dict(self.t, **STATIC)
        tables["map"] = [dict(token=token(), log_tokens=[self.log_token], category="semantic_prior", filename="")]
        for name, rows in tables.items():
            (out / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
        return out


# ---------------------------------------------------------------- đọc point cloud
def read_points(path: Path) -> np.ndarray:
    """.bin (float32 x y z i [...], 4 hoặc 5 cột), .npy, .pcd (ascii / binary; không hỗ trợ binary_compressed)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path).astype(np.float32)
    if suf == ".pcd":
        return _read_pcd(path)
    raw = np.fromfile(path, np.float32)
    for cols in (4, 5, 3):  # KITTI 4 cột, nuScenes 5 cột
        if raw.size % cols == 0:
            return raw.reshape(-1, cols)
    raise ValueError(f"{path.name}: không đoán được số cột (cần float32 x y z [cường độ])")


def _read_pcd(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header, fields, sizes, types, counts, n, fmt = {}, [], [], [], [], 0, "ascii"
    pos = 0
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end].decode("ascii", "ignore").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            fmt = vals[0].lower()
            break
    fields = header["FIELDS"]
    sizes = [int(v) for v in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    n = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    if fmt == "binary_compressed":
        raise ValueError(f"{path.name}: PCD binary_compressed chưa hỗ trợ, lưu lại dạng binary hoặc .bin")
    names = []
    for f, c in zip(fields, counts, strict=True):
        names += [f] if c == 1 else [f"{f}_{i}" for i in range(c)]
    if fmt == "ascii":
        arr = np.loadtxt(data[pos:].decode("ascii", "ignore").splitlines(), dtype=np.float64, ndmin=2)
        cols = {nm: arr[:, i] for i, nm in enumerate(names)}
    else:
        kind = {"F": "f", "I": "i", "U": "u"}
        dt = np.dtype([(nm, f"<{kind[t]}{s}") for f, s, t, c in zip(fields, sizes, types, counts, strict=True)
                       for nm in ([f] if c == 1 else [f"{f}_{i}" for i in range(c)])])  # fmt: skip
        rec = np.frombuffer(data[pos : pos + n * dt.itemsize], dtype=dt, count=n)
        cols = {nm: rec[nm].astype(np.float64) for nm in names}
    inten = next(
        (cols[k] for k in ("intensity", "i", "reflectance", "remission") if k in cols), np.zeros(len(cols["x"]))
    )
    out = np.stack([cols["x"], cols["y"], cols["z"], inten], 1).astype(np.float32)
    return out[np.isfinite(out).all(1)]
