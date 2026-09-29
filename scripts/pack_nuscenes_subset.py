"""Chọn / đóng gói vài scene nuScenes (+ workspace đã auto-label) thành một file zip nhỏ.

Dùng để test nhanh trên một lượng nhỏ dữ liệu, hoặc gửi cho thành viên chưa tải đủ dataset. Đóng gói kèm kết quả
auto-label (sau khi đã chạy `python -m src.cli run --scenes ...`):

    python scripts/pack_nuscenes_subset.py --dataroot ../v1.0-trainval --version v1.0-trainval \\
        --scenes scene-0031 scene-0065 --out ../autolabel_subset.zip

Lấy mẫu ngẫu nhiên một lượng nhỏ để test (chỉ chọn trong các scene có ảnh trên máy, hợp với bản trainval mới tải
vài phần blob), mỗi scene cắt một đoạn ngẫu nhiên 10 keyframe:

    python scripts/pack_nuscenes_subset.py --dataroot ../v1.0-trainval --version v1.0-trainval \\
        --random 2 --seed 20260927 --window 10 --with-lidar --out ../nusc_random.zip

Nội dung zip (thư mục gốc `autolabel_subset/`):
    nuscenes/<version>/*.json          bảng nuScenes đã lọc theo scene (loader của repo đọc được như bản đầy đủ)
    nuscenes/samples|sweeps/<CAMERA>/  ảnh camera của các scene đó: keyframe 2Hz + sweep 12Hz (cho timeline, lan truyền)
    nuscenes/samples/LIDAR_TOP/        chỉ khi --with-lidar: point cloud keyframe, để chạy lại `run` trên máy
    nuscenes/samples/CAM_*/            chỉ khi --all-cameras: ảnh keyframe của đủ 6 camera (phần 3D)
    nuscenes/sweeps/LIDAR_TOP/         chỉ khi --lidar-sweeps N: N lần quét LiDAR liền trước mỗi keyframe
    workspace/                         frame đã auto-label, LiDAR đã chiếu, GT 2D, cache detection, correction log
    eval/                              báo cáo trong eval/results nếu có
    README.txt                         cách dùng

Trên máy: giải nén vào thư mục repo, thêm vào `.env`
    NUSCENES_DATAROOT=./autolabel_subset/nuscenes
    NUSCENES_VERSION=<version>
    WORKSPACE_DIR=./autolabel_subset/workspace
rồi `uvicorn src.main:app` (chỉ cần `pip install -r requirements.txt`, không cần GPU).

Phần 3D: chỉ chọn trong tập val của nuScenes (mô hình 3D có sẵn đều đã học trên tập train), kèm đủ 6 camera và LiDAR:

    python scripts/pack_nuscenes_subset.py --dataroot ../v1.0-trainval --version v1.0-trainval \\
        --random 3 --split val --seed 20260928 --with-lidar --all-cameras --out ../nusc3d.zip
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path

ROOT_NAME = "autolabel_subset"
# Bảng nhỏ, giữ nguyên (để devkit chính thức cũng đọc được)
WHOLE_TABLES = ("attribute", "category", "sensor", "visibility", "log", "map", "calibrated_sensor")
# Bảng rất lớn ở bản trainval (sample_data 1.3 GB, ego_pose 0.6 GB, sample_annotation 0.6 GB): đọc kiểu stream,
# giữ lại đúng các dòng cần, để máy 8–16 GB RAM vẫn chạy được
BIG_TABLES = ("sample_data", "ego_pose", "sample_annotation")
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")


def split_scenes(split: str) -> set[str]:
    """Tên scene của một split chính thức nuScenes (train / val / mini_val ...), lấy từ nuscenes-devkit.

    Chỉ đọc file splits.py của devkit (import cả gói devkit kéo theo sklearn, matplotlib, pyquaternion...).
    """
    import importlib.util

    spec = importlib.util.find_spec("nuscenes")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Cần nuscenes-devkit cho --split: pip install --no-deps nuscenes-devkit")
    src = (Path(next(iter(spec.submodule_search_locations))) / "utils" / "splits.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(src.replace("from nuscenes import NuScenes", "NuScenes = None"), ns)  # noqa: S102
    create_splits_scenes = ns["create_splits_scenes"]
    splits = create_splits_scenes()
    if split not in splits:
        raise SystemExit(f"Split {split!r} không có; chọn một trong {', '.join(sorted(splits))}")
    return set(splits[split])


def _iter_json_array(path: Path, chunk: int = 1 << 22) -> Iterator[dict]:
    """Đọc lần lượt từng phần tử của file JSON dạng mảng object mà không nạp cả file vào RAM."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf, pos = f.read(chunk).lstrip().removeprefix("["), 0
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = more, 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)  # object bị cắt ở cuối đoạn đã đọc: đọc thêm rồi thử lại
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end


@cache
def _load_cached(table_dir: str, name: str) -> list[dict] | None:
    path = Path(table_dir) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load(table_dir: Path, name: str) -> list[dict] | None:
    """Bảng nhỏ: đọc một lần cho mỗi lần chạy. Không sửa trực tiếp bản ghi trả về: copy trước."""
    assert name not in BIG_TABLES, name
    return _load_cached(str(table_dir), name)


def _iter(table_dir: Path, name: str) -> Iterator[dict]:
    """Bảng lớn: duyệt từng dòng (mỗi dòng là dict mới, sửa thoải mái). Không có file thì rỗng."""
    path = table_dir / f"{name}.json"
    if path.exists():
        yield from _iter_json_array(path)


def _channels(table_dir: Path) -> dict[str, str]:
    channel_of_sensor = {r["token"]: r["channel"] for r in _load(table_dir, "sensor")}
    return {r["token"]: channel_of_sensor[r["sensor_token"]] for r in _load(table_dir, "calibrated_sensor")}


def pick_random_scenes(
    dataroot: Path, version: str, n: int, seed: int, camera: str = "CAM_FRONT", allowed: set[str] | None = None
) -> list[str]:
    """Chọn ngẫu nhiên n scene trong số scene có ảnh keyframe đầu và cuối trên máy (bản tải thiếu blob vẫn chọn được).

    allowed: chỉ chọn trong các scene này (ví dụ tập val).
    """
    table_dir = dataroot / version
    scenes = _load(table_dir, "scene")
    if scenes is None:
        raise SystemExit(f"Không thấy {table_dir / 'scene.json'} — kiểm tra --dataroot và --version")
    channel_of_cs = _channels(table_dir)
    key_image = {
        sd["sample_token"]: sd["filename"]
        for sd in _iter(table_dir, "sample_data")
        if sd["is_key_frame"] and channel_of_cs[sd["calibrated_sensor_token"]] == camera
    }

    def on_disk(sample_token: str) -> bool:
        return sample_token in key_image and (dataroot / key_image[sample_token]).is_file()

    available = sorted(
        s["name"]
        for s in scenes
        if (allowed is None or s["name"] in allowed)
        and on_disk(s["first_sample_token"])
        and on_disk(s.get("last_sample_token") or s["first_sample_token"])
    )
    if not available:
        raise SystemExit(f"Không scene nào có ảnh {camera} trong {dataroot} (đã giải nén blob chưa?)")
    return sorted(random.Random(seed).sample(available, min(n, len(available))))


def _cut_chain(rows: list[dict], keys: tuple[str, ...]) -> None:
    """Token trỏ tới bản ghi đã bị lọc bỏ thì đổi thành "" (như đầu/cuối chuỗi)."""
    kept = {r["token"] for r in rows}
    for r in rows:
        for k in keys:
            if r.get(k) and r[k] not in kept:
                r[k] = ""


def filter_tables(
    table_dir: Path,
    scene_names: list[str],
    camera: str = "CAM_FRONT",
    window: int | None = None,
    seed: int = 0,
    all_cameras: bool = False,
    lidar_sweeps: int = 0,
) -> tuple[dict[str, list[dict]], list[str], list[str]]:
    """Lọc bảng theo scene. Trả về (bảng đã lọc, file ảnh camera, file LiDAR).

    window: chỉ giữ một đoạn ngẫu nhiên `window` keyframe liên tiếp của mỗi scene (scene rút gọn, loader vẫn đọc được).
    all_cameras: thêm ảnh keyframe của 5 camera còn lại (camera chính vẫn lấy cả sweep).
    lidar_sweeps: thêm N lần quét LiDAR liền trước mỗi keyframe (mô hình 3D gộp nhiều lần quét).
    """
    all_scenes = _load(table_dir, "scene")
    if all_scenes is None:
        raise SystemExit(f"Không thấy {table_dir / 'scene.json'} — kiểm tra --dataroot và --version")
    scenes = [dict(s) for s in all_scenes if s["name"] in scene_names]
    missing = sorted(set(scene_names) - {s["name"] for s in scenes})
    if missing:
        have = ", ".join(sorted(s["name"] for s in all_scenes)[:12])
        raise SystemExit(f"Không có scene {', '.join(missing)} trong {table_dir.name} (có: {have} …)")
    scene_tokens = {s["token"] for s in scenes}

    samples = [dict(s) for s in _load(table_dir, "sample") if s["scene_token"] in scene_tokens]
    if window:
        rng = random.Random(seed)
        by_token = {s["token"]: s for s in samples}
        kept = []
        for sc in sorted(scenes, key=lambda s: s["name"]):
            chain, tok = [], sc["first_sample_token"]
            while tok and tok in by_token:
                chain.append(by_token[tok])
                tok = by_token[tok]["next"]
            if len(chain) > window:
                start = rng.randrange(len(chain) - window + 1)
                chain = chain[start : start + window]
            sc.update(
                first_sample_token=chain[0]["token"], last_sample_token=chain[-1]["token"], nbr_samples=len(chain)
            )
            kept += chain
        samples = kept
        _cut_chain(samples, ("prev", "next"))
    sample_tokens = {s["token"] for s in samples}
    channel_of_cs = _channels(table_dir)

    sample_data, cam_files, lidar_files = [], [], []
    lidar_all: dict[str, dict] = {}
    for sd in _iter(table_dir, "sample_data"):
        if sd["sample_token"] not in sample_tokens:
            continue
        channel = channel_of_cs[sd["calibrated_sensor_token"]]
        if channel == camera:
            sample_data.append(sd)
            cam_files.append(sd["filename"])
        elif all_cameras and channel in CAMERAS and sd["is_key_frame"]:
            sample_data.append(sd)
            cam_files.append(sd["filename"])
        elif channel == "LIDAR_TOP":
            lidar_all[sd["token"]] = sd
    # LiDAR keyframe dùng cho QA (chiếu điểm lên ảnh); sweep chỉ lấy khi cần gộp nhiều lần quét
    keep_lidar = set()
    for tok, sd in lidar_all.items():
        if not sd["is_key_frame"]:
            continue
        keep_lidar.add(tok)
        prev = sd["prev"]
        for _ in range(lidar_sweeps):
            if not prev or prev not in lidar_all:
                break
            keep_lidar.add(prev)
            prev = lidar_all[prev]["prev"]
    for tok in keep_lidar:
        sample_data.append(lidar_all[tok])
        lidar_files.append(lidar_all[tok]["filename"])
    lidar_files.sort()
    _cut_chain(sample_data, ("prev", "next"))

    ego_tokens = {sd["ego_pose_token"] for sd in sample_data}
    ego_pose = [e for e in _iter(table_dir, "ego_pose") if e["token"] in ego_tokens]

    annotations = [a for a in _iter(table_dir, "sample_annotation") if a["sample_token"] in sample_tokens]
    _cut_chain(annotations, ("prev", "next"))
    instance_tokens = {a["instance_token"] for a in annotations}
    instances = [i for i in (_load(table_dir, "instance") or []) if i["token"] in instance_tokens]

    tables = {
        "scene": scenes,
        "sample": samples,
        "sample_data": sample_data,
        "ego_pose": ego_pose,
        "sample_annotation": annotations,
        "instance": instances,
    }
    for name in WHOLE_TABLES:
        rows = _load(table_dir, name)
        if rows is not None:
            tables[name] = rows
    return tables, cam_files, lidar_files


def workspace_files(workspace: Path, scene_names: list[str], sd_tokens: set[str]) -> list[Path]:
    """File workspace thuộc các scene đã chọn: frame/aux theo tên scene, cache detection theo sample_data token."""
    if not workspace.is_dir():
        return []
    prefixes = tuple(f"{s}_" for s in scene_names)
    out = []
    for p in sorted(workspace.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(workspace)
        top = rel.parts[0]
        if top in {"exports", "videos"}:
            continue  # dataset đã xuất và video mp4 tải lên không thuộc bộ nuScenes
        if top == "cache":
            if p.stem in sd_tokens:
                out.append(p)
        elif len(rel.parts) == 2 and p.suffix == ".json":  # frames/, lidar/, gt/, ...
            if p.stem.startswith(prefixes):
                out.append(p)
        elif len(rel.parts) == 1:  # corrections.jsonl
            out.append(p)
    return out


def readme(version: str, scene_names: list[str], camera: str, with_lidar: bool, all_cameras: bool = False) -> str:
    extra = (" · có LiDAR" if with_lidar else "") + (" · đủ 6 camera (keyframe)" if all_cameras else "")
    return f"""AutoLabel — gói nuScenes rút gọn
Scene: {", ".join(scene_names)} · camera {camera} · version {version}{extra}

1. Giải nén vào thư mục gốc của repo (cạnh src/), được thư mục {ROOT_NAME}/
2. Thêm vào file .env của repo:
   NUSCENES_DATAROOT=./{ROOT_NAME}/nuscenes
   NUSCENES_VERSION={version}
   WORKSPACE_DIR=./{ROOT_NAME}/workspace
3. pip install -r requirements.txt   (không cần GPU)
4. uvicorn src.main:app --port 8000  -> http://localhost:8000, chọn 🎞 Video để duyệt theo scene và lan truyền nhãn
{"" if with_lidar else "Gói không có file LiDAR gốc: xem/duyệt/lan truyền được, nhưng chạy lại `python -m src.cli run` thì cần dataset đầy đủ." + chr(10)}"""


def pack(
    dataroot: Path,
    version: str,
    scene_names: list[str],
    out: Path,
    workspace: Path | None = None,
    camera: str = "CAM_FRONT",
    with_lidar: bool = False,
    eval_dir: Path | None = None,
    window: int | None = None,
    seed: int = 0,
    all_cameras: bool = False,
    lidar_sweeps: int = 0,
) -> dict:
    tables, cam_files, lidar_files = filter_tables(
        dataroot / version, scene_names, camera, window, seed, all_cameras, lidar_sweeps
    )
    with_lidar = with_lidar or lidar_sweeps > 0
    files = cam_files + (lidar_files if with_lidar else [])
    missing = [f for f in files if not (dataroot / f).is_file()]
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} file trong dataroot, ví dụ {missing[0]}")
    sd_tokens = {sd["token"] for sd in tables["sample_data"]}
    ws_files = workspace_files(workspace, scene_names, sd_tokens) if workspace else []
    ev_files = sorted(p for p in eval_dir.glob("*") if p.is_file()) if eval_dir and eval_dir.is_dir() else []

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, rows in tables.items():
            zf.writestr(f"{ROOT_NAME}/nuscenes/{version}/{name}.json", json.dumps(rows))
        for f in files:
            # ảnh jpg / point cloud đã nén sẵn: lưu thẳng cho nhanh
            zf.write(dataroot / f, f"{ROOT_NAME}/nuscenes/{f}", compress_type=zipfile.ZIP_STORED)
        for p in ws_files:
            zf.write(p, f"{ROOT_NAME}/workspace/{p.relative_to(workspace).as_posix()}")
        for p in ev_files:
            zf.write(p, f"{ROOT_NAME}/eval/{p.name}")
        zf.writestr(f"{ROOT_NAME}/README.txt", readme(version, scene_names, camera, with_lidar, all_cameras))
    n_frames = sum(1 for p in ws_files if p.parent.name == "frames")
    return {
        "zip": out,
        "size_mb": round(out.stat().st_size / 1e6, 1),
        "keyframes": sum(1 for s in tables["sample"]),
        "images": len(cam_files),
        "lidar": len(lidar_files) if with_lidar else 0,
        "workspace_frames": n_frames,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path, help="Thư mục chứa <version>/, samples/, sweeps/")
    ap.add_argument("--version", default="v1.0-mini")
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--scenes", nargs="+", help="Ví dụ scene-0061 scene-0103")
    pick.add_argument("--random", type=int, metavar="N", help="Chọn ngẫu nhiên N scene có ảnh trên máy")
    ap.add_argument("--seed", type=int, default=0, help="Seed cho --random / --window (ghi lại để lặp lại được)")
    ap.add_argument("--window", type=int, help="Chỉ lấy một đoạn ngẫu nhiên N keyframe liên tiếp mỗi scene")
    ap.add_argument("--out", type=Path, default=Path(f"{ROOT_NAME}.zip"))
    ap.add_argument("--workspace", type=Path, default=Path("data/workspace"), help="Workspace đã auto-label")
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--with-lidar", action="store_true", help="Kèm point cloud keyframe (to hơn ~40 MB/scene)")
    ap.add_argument("--eval-dir", type=Path, default=Path("eval/results"))
    ap.add_argument("--split", help="Chỉ chọn --random trong split nuScenes này (val cho phần 3D)")
    ap.add_argument("--all-cameras", action="store_true", help="Kèm ảnh keyframe của đủ 6 camera (phần 3D)")
    ap.add_argument("--lidar-sweeps", type=int, default=0, help="Kèm N lần quét LiDAR trước mỗi keyframe")
    args = ap.parse_args(argv)
    allowed = split_scenes(args.split) if args.split else None
    scenes = args.scenes or pick_random_scenes(
        args.dataroot, args.version, args.random, args.seed, args.camera, allowed
    )
    if args.random:
        print(f"Chọn ngẫu nhiên (seed {args.seed}): {', '.join(scenes)}")
    info = pack(
        args.dataroot,
        args.version,
        scenes,
        args.out,
        args.workspace,
        args.camera,
        args.with_lidar,
        args.eval_dir,
        args.window,
        args.seed,
        args.all_cameras,
        args.lidar_sweeps,
    )
    print(
        f"Đã ghi {info['zip']} ({info['size_mb']} MB): {info['keyframes']} keyframe, {info['images']} ảnh "
        f"{args.camera}, {info['lidar']} file LiDAR, {info['workspace_frames']} frame đã auto-label"
    )
    if info["workspace_frames"] == 0:
        print(
            "Lưu ý: workspace chưa có frame của các scene này — chạy `python -m src.cli run --scenes …` trước",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
