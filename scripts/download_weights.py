"""Tải weights cho detector mặc định (YOLOE-26-L) vào weights/ và kiểm SHA256.

    python scripts/download_weights.py            # tải file còn thiếu / sai hash
    python scripts/download_weights.py --all      # thêm yolo26l.pt (detector tuỳ chọn `yolo26`)
    python scripts/download_weights.py --check    # chỉ kiểm file đã có

Nguồn (thử lần lượt, file tải về phải khớp một trong các hash đã biết, không khớp thì xoá):
- yoloe-26l-seg.pt  YOLOE-26-L open-vocab, bản phát hành ultralytics v8.4.0. Dự phòng: model.pt của
                    https://huggingface.co/openvision/yoloe26-l-seg — là cùng model đã fuse Conv+BN rồi lưu lại
                    (hash khác; đã kiểm: pickle chỉ tham chiếu ultralytics/torch, 77/79 box trùng bản chính thức,
                    lệch score ≤ 0.004 trên 5 ảnh nuScenes).
- mobileclip2_b.ts  text encoder mà YOLOE-26 dùng để mã hoá prompt.
- yolo26l.pt        (--all) https://github.com/ultralytics/yolo26 — bản phát hành ultralytics v8.4.0; Hugging Face
                    Ultralytics/YOLO26 có đúng file đó (cùng hash).

File .pt là pickle (nạp là chạy code), nên chỉ nhận file có hash nằm trong danh sách dưới đây.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
GH = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

# tên file -> [(url, sha256)], thử theo thứ tự
FILES = {
    "yoloe-26l-seg.pt": [
        (f"{GH}/yoloe-26l-seg.pt", "a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9"),
        (
            "https://huggingface.co/openvision/yoloe26-l-seg/resolve/main/model.pt",
            "a9413bf3f15772c223a03bbedd71b79af5822f830e80aa1b51b1a469e65927b1",
        ),
    ],
    "mobileclip2_b.ts": [
        (f"{GH}/mobileclip2_b.ts", "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"),
    ],
}
# Chỉ cho detector tuỳ chọn `yolo26` (COCO); tải khi có --all
OPTIONAL = {
    "yolo26l.pt": [
        (f"{GH}/yolo26l.pt", "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),
        (
            "https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26l.pt",
            "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
        ),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "autolabel2d-download-weights"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm file đã có, không tải")
    parser.add_argument("--all", action="store_true", help="Tải thêm yolo26l.pt cho detector tuỳ chọn `yolo26`")
    args = parser.parse_args()
    WEIGHTS_DIR.mkdir(exist_ok=True)

    failed = []
    for name, sources in (FILES | OPTIONAL if args.all else FILES).items():
        dest = WEIGHTS_DIR / name
        known = {digest for _, digest in sources}
        if dest.exists() and sha256(dest) in known:
            print(f"OK       {name}")
            continue
        if args.check:
            print(f"THIẾU    {name}" if not dest.exists() else f"SAI HASH {name}")
            failed.append(name)
            continue
        for url, digest in sources:
            print(f"Tải {name} <- {url}")
            try:
                download(url, dest)
            except OSError as e:
                print(f"  lỗi: {e}")
                continue
            if sha256(dest) == digest:
                print(f"OK       {name}")
                break
            print("  SHA256 không khớp, xoá file")
            dest.unlink()
        else:
            failed.append(name)

    if failed:
        print(f"\nChưa có: {', '.join(failed)}. Tải tay từ URL trong script vào {WEIGHTS_DIR} rồi chạy --check")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
