"""Tải SAM 2.1 dạng ONNX (encoder + decoder) cho công cụ "✨ Chọn vật" vào weights/sam2-onnx/.

    python scripts/download_sam_onnx.py               # tiny (120 MB): CPU ~2 s / ảnh lần đầu, bấm tiếp ~0.1 s
    python scripts/download_sam_onnx.py --model large  # chính xác nhất, nên có GPU (pip install onnxruntime-gpu)

Mô hình do tác giả AnyLabeling xuất và công bố ở huggingface.co/vietanhdev/segment-anything-2.1-onnx-models
(SAM 2.1 của Meta, Apache-2.0). Server tự dùng bản lớn nhất có trong thư mục.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/resolve/main"
MODELS = {"tiny": "sam2.1_hiera_tiny_20260221", "small": "sam2.1_hiera_small_20260221",
          "base_plus": "sam2.1_hiera_base_plus_20260221", "large": "sam2.1_hiera_large_20260221"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="tiny")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "weights" / "sam2-onnx"))
    args = ap.parse_args()
    out = Path(args.out) / args.model
    if list(out.glob("*.encoder.onnx")):
        print(f"Đã có {args.model} ở {out}")
        return
    url = f"{BASE}/{MODELS[args.model]}.zip"
    print(f"Tải {url} …", flush=True)
    with urllib.request.urlopen(url) as r:  # noqa: S310 - URL cố định
        total = int(r.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        while chunk := r.read(1 << 20):
            buf.write(chunk)
            if total:
                sys.stdout.write(f"\r  {buf.tell() / 1e6:.0f} / {total / 1e6:.0f} MB")
                sys.stdout.flush()
    print()
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as z:
        for name in z.namelist():
            if name.endswith((".onnx", ".yaml")):
                (out / Path(name).name).write_bytes(z.read(name))
    print(f"Xong: {sorted(p.name for p in out.iterdir())} -> {out}")


if __name__ == "__main__":
    main()
