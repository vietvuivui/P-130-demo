"""So sánh trước / sau khi dùng optical flow cho QA temporal, tính lại score theo sweep (idea 2) và lan truyền (idea 1).

Chạy trên một workspace đã `python -m src.cli run` (cache detection của keyframe + sweep t-2..t+2 đã có), nên mọi cấu hình
dùng đúng cùng một detection — chỉ khác phần xử lý sau detector:

    python tools2d/eval_temporal.py --dataroot <nuscenes> --workspace <ws đã run> --out eval/results/temporal/dev

Cấu hình 2D (mỗi cái auto-label lại toàn bộ keyframe vào một workspace riêng, chung cache detection):
    base        : như trước (so khớp sweep trực tiếp, score detector)
    flow        : so khớp sweep sau khi dời bằng optical flow (FLICKER / RECOVERED chính xác hơn), score detector
    flow+mean   : + score tính lại = trung bình keyframe và các sweep (sweep không thấy góp 0)
    flow+linked : + score tính lại = trung bình các lần thấy (kiểu Seq-NMS)
    mean        : score tính lại nhưng không flow (để xem flow có cần không)
Lan truyền (thí nghiệm keyframe hoàn hảo, src/services/sequence.py; bắt đầu ở keyframe 0, 5, 10… của mỗi scene, đi tối
đa 10 keyframe = 5 s): propagation.flow = off | missing | always.

Chọn cấu hình bằng scene dev (mAP50, rồi F1), báo kết quả chính trên held-out — không chỉnh gì theo held-out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.temporal_eval import DEFAULT_VARIANTS, VARIANTS, evaluate, print_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataroot", required=True, type=Path)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument(
        "--workspace", required=True, type=Path, help="workspace đã chạy `src.cli run` (có cache detection)"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--scenes", nargs="*", help="chỉ các scene này (mặc định: mọi scene của workspace)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "autolabel.yaml"))
    args = ap.parse_args()

    from src.models.qa_config import load_autolabel_config
    from src.services.nuscenes_data import NuScenesMini

    config = load_autolabel_config(args.config)
    data = NuScenesMini(args.dataroot, args.version)
    print_summary(evaluate(data, args.workspace, args.out, config, args.variants, args.scenes))


if __name__ == "__main__":
    main()
