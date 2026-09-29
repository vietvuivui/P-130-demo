"""Đổi nuImages (box 2D gán tay, cùng miền xe / thành phố với nuScenes) sang dataset YOLO để fine-tune YOLOE.

Chạy ở máy huấn luyện, từ thư mục gốc của repo:

    python tools2d/nuimages_to_yolo.py --dataroot D:/nuimages --out D:/nuimages_yolo

    --dataroot          thư mục có v1.0-train/, v1.0-val/ (bảng json) và samples/ (ảnh keyframe)
    --out               dataset YOLO: images/{train,val}, labels/{train,val}, data.yaml
    chống rò rỉ         mặc định bỏ mọi ảnh nuImages chụp cùng xe, cùng ngày với một scene val của nuScenes (danh sách
                        đi kèm: tools2d/nuscenes_val_logs.json), để mô hình không thấy đúng con đường / đúng xe của tập
                        dùng để chấm cuối. Không cần có nuScenes trên máy huấn luyện.
    --exclude-nuscenes  đọc lại danh sách log val từ bảng nuScenes có trên máy (thay cho file đi kèm)
    --no-exclude        không loại (chỉ để so sánh)
    --link              hardlink | symlink | copy (mặc định hardlink: không tốn thêm ổ, cần cùng ổ đĩa)
    --limit N           chỉ lấy N ảnh mỗi tập (thử nhanh)

Lớp: 10 lớp của configs/autolabel.yaml (gt_category_map, giống nuScenes detection). Tên lớp trong data.yaml viết
bằng chữ thường có dấu cách ("traffic cone") vì YOLOE lấy text embedding từ tên lớp làm điểm khởi đầu cho đầu phân lớp.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Scene val chính thức của nuScenes (splits.py của devkit) — chỉ cần khi không có bảng scene.json để đọc
MIN_BOX_PX = 4  # box nhỏ hơn 4 px (sau khi đổi cỡ ảnh vẫn là nhiễu) bị bỏ


def class_names(config_path: Path) -> tuple[list[str], dict[str, str]]:
    from src.models.qa_config import load_autolabel_config

    cfg = load_autolabel_config(config_path)
    classes = [c for c in cfg.classes if not c.startswith("__")]
    return classes, dict(cfg.gt_category_map)


def load_table(root: Path, version: str, name: str) -> list[dict]:
    with open(root / version / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def nuscenes_val_logs(nusc_root: Path) -> set[tuple[str, str]]:
    """(xe, ngày) của các scene val nuScenes có trên máy (đọc bảng của v1.0-trainval hoặc v1.0-mini)."""
    from nuscenes_val import VAL_SCENES  # noqa: PLC0415 — file cạnh script, tách cho dễ đọc

    out = set()
    for version in ("v1.0-trainval", "v1.0-mini"):
        if not (nusc_root / version / "scene.json").exists():
            continue
        logs = {lg["token"]: lg for lg in load_table(nusc_root, version, "log")}
        for sc in load_table(nusc_root, version, "scene"):
            if sc["name"] in VAL_SCENES:
                lg = logs[sc["log_token"]]
                out.add((lg["vehicle"], lg["date_captured"]))
    if not out:
        raise SystemExit(f"Không đọc được scene val nào trong {nusc_root} (cần v1.0-trainval/ hoặc v1.0-mini/)")
    return out


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:  # khác ổ đĩa / hệ file không hỗ trợ -> chép
            pass
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def convert_split(root: Path, version: str, split: str, out: Path, classes: list[str], cat_map: dict[str, str],
                  exclude: set[tuple[str, str]], mode: str, limit: int | None, seed: int = 0) -> dict:  # fmt: skip
    cats = {c["token"]: c["name"] for c in load_table(root, version, "category")}
    logs = {lg["token"]: lg for lg in load_table(root, version, "log")}
    samples = {s["token"]: s for s in load_table(root, version, "sample")}
    sds = {sd["token"]: sd for sd in load_table(root, version, "sample_data") if sd["is_key_frame"]}
    anns = defaultdict(list)
    for a in load_table(root, version, "object_ann"):
        if a["sample_data_token"] in sds:
            anns[a["sample_data_token"]].append(a)

    tokens = sorted(sds)
    dropped_leak = 0
    if exclude:
        keep = []
        for t in tokens:
            lg = logs[samples[sds[t]["sample_token"]]["log_token"]]
            if (lg["vehicle"], lg["date_captured"]) in exclude:
                dropped_leak += 1
            else:
                keep.append(t)
        tokens = keep
    if limit:
        random.Random(seed).shuffle(tokens)
        tokens = sorted(tokens[:limit])

    idx = {c: i for i, c in enumerate(classes)}
    n_box = Counter()
    missing = 0
    for t in tokens:
        sd = sds[t]
        src = root / sd["filename"]
        if not src.exists():
            missing += 1
            continue
        w, h = sd["width"], sd["height"]
        lines = []
        for a in anns.get(t, []):
            label = cat_map.get(cats[a["category_token"]])
            if label is None:
                continue
            x1, y1, x2, y2 = a["bbox"]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                continue
            n_box[label] += 1
            lines.append(
                f"{idx[label]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        stem = Path(sd["filename"]).stem
        place(src, out / "images" / split / f"{stem}{src.suffix}", mode)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return {"images": len(tokens) - missing, "missing_files": missing, "dropped_same_log_as_nuscenes_val": dropped_leak,
            "boxes": dict(sorted(n_box.items()))}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "autolabel.yaml")
    ap.add_argument("--exclude-nuscenes", type=Path, default=None)
    ap.add_argument("--no-exclude", action="store_true")
    ap.add_argument("--link", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    classes, cat_map = class_names(args.config)
    if args.no_exclude:
        exclude = set()
    elif args.exclude_nuscenes:
        exclude = nuscenes_val_logs(args.exclude_nuscenes)
    else:
        bundled = Path(__file__).with_name("nuscenes_val_logs.json")
        exclude = {tuple(x) for x in json.loads(bundled.read_text(encoding="utf-8"))["logs"]}
    if exclude:
        print(f"Loại ảnh cùng xe + cùng ngày với {len(exclude)} log val của nuScenes")
    stats = {}
    for version, split in (("v1.0-train", "train"), ("v1.0-val", "val")):
        if not (args.dataroot / version).is_dir():
            raise SystemExit(f"Thiếu {args.dataroot / version} (giải nén nuimages-v1.0-all-metadata.tgz)")
        stats[split] = convert_split(
            args.dataroot, version, split, args.out, classes, cat_map, exclude, args.link, args.limit
        )
        print(split, json.dumps(stats[split], ensure_ascii=False))
    names = "\n".join(f"  {i}: {c.replace('_', ' ')}" for i, c in enumerate(classes))
    (args.out / "data.yaml").write_text(
        f"# Sinh bởi tools2d/nuimages_to_yolo.py\npath: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Xong: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
