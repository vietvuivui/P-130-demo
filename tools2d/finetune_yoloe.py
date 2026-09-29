"""Fine-tune YOLOE-26L trên dataset YOLO sinh bởi nuimages_to_yolo.py (máy có GPU, vd. RTX 3090 24 GB).

    python tools2d/finetune_yoloe.py baseline --data D:/nuimages_yolo/data.yaml       # zero-shot, để so sánh
    python tools2d/finetune_yoloe.py lp   --data D:/nuimages_yolo/data.yaml           # linear probe (~vài giờ)
    python tools2d/finetune_yoloe.py full --data D:/nuimages_yolo/data.yaml           # toàn bộ mạng (lâu hơn nhiều)

    lp    chỉ học lớp cuối của nhánh phân lớp (khởi tạo từ text embedding tên lớp), giữ nguyên backbone/neck/nhánh box:
          nhanh, ít VRAM, gần như không thể overfit — nên chạy trước.
    full  học toàn bộ mạng với lr nhỏ, AdamW, weight decay, mosaic tắt ở 10 epoch cuối, dừng sớm theo mAP của nuImages
          val (patience). Chỉ dùng khi lp đã tốt hơn baseline và còn thời gian GPU.

Chống overfit: chọn checkpoint theo nuImages val (Ultralytics tự làm: best.pt); con số cuối cùng phải đo trên các
scene held-out của nuScenes bằng tools2d/eval2d.py — dữ liệu mà quá trình huấn luyện và chọn checkpoint chưa thấy.

Kết quả: runs/yoloe_ft/<tên>/weights/best.pt. Đem file này về máy chạy web, đặt vào weights/ và sửa
configs/autolabel.yaml: detection.yoloe.weights: weights/yoloe-26l-nuimages.pt (xem tools2d/README.md).
Trọng số đã fine-tune là mô hình tập lớp đóng (10 lớp nuScenes): detector tự nhận ra và bỏ qua prompt chữ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_WEIGHTS = "yoloe-26l-seg.pt"  # trọng số gốc (open-vocab, có nhánh mask); fine-tune bỏ mask, chỉ học box
DET_YAML = "yoloe-26l.yaml"  # cùng kiến trúc, không có nhánh mask (nuImages có mask nhưng pipeline chỉ dùng box)


def resume_trainer():
    """YOLOEPETrainer dựng lại mô hình từ yaml + trọng số rồi gộp text embedding vào đầu phân lớp lần nữa: với checkpoint
    đã fine-tune (đầu phân lớp đã gộp và đã học) việc đó làm hỏng mô hình (mAP về 0 khi chạy tiếp). Khi resume, dùng
    thẳng mô hình trong checkpoint."""
    from copy import deepcopy

    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    class Trainer(YOLOEPETrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if self.args.resume and weights is not None and not isinstance(weights, (str, Path)):
                model = deepcopy(weights).float()
                for p in model.parameters():
                    p.requires_grad_(True)  # đóng băng lại theo args.freeze ở _setup_train
                if (
                    getattr(model, "criterion", 1) is None
                ):  # checkpoint lưu criterion=None; YOLOEModel chỉ tạo lại khi thiếu
                    del model.criterion
                model.train()
                return model
            return super().get_model(cfg, weights, verbose)

    return Trainer


def build(weights: str):
    from ultralytics import YOLOE

    model = YOLOE(DET_YAML)
    model.load(weights)
    return model


def linear_probe_freeze(model) -> list[str]:
    """Đóng băng mọi thứ trừ lớp cuối (x.2) của cv3 / one2one_cv3 — đúng cách Ultralytics hướng dẫn cho YOLOE."""
    head = len(model.model.model) - 1
    freeze = [str(i) for i in range(head)]
    for name, _ in model.model.model[-1].named_children():
        if "cv3" not in name:
            freeze.append(f"{head}.{name}")
    for branch in ("cv3", "one2one_cv3"):
        if hasattr(model.model.model[-1], branch):
            for i in range(3):
                freeze += [f"{head}.{branch}.{i}.0", f"{head}.{branch}.{i}.1"]
    return freeze


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["baseline", "lp", "full"])
    ap.add_argument("--data", default=None, help="data.yaml của nuimages_to_yolo.py")
    ap.add_argument(
        "--resume",
        default=None,
        help="chạy tiếp từ runs/yoloe_ft/<tên>/weights/last.pt (máy tắt / cập nhật Windows giữa chừng)",
    )
    ap.add_argument("--weights", default=BASE_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=1280, help="1280 giống lúc suy luận; hết VRAM thì 960 + batch nhỏ")
    ap.add_argument("--epochs", type=int, default=None, help="mặc định lp 10, full 30")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fraction", type=float, default=1.0, help="dùng một phần tập train (thử nhanh)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default=str(ROOT / "runs" / "yoloe_ft"))
    args = ap.parse_args()

    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    if args.resume:  # mọi tham số (dữ liệu, epoch, lr, freeze...) lấy lại từ checkpoint
        from ultralytics import YOLOE

        model = YOLOE(args.resume)
        model.train(resume=True, trainer=resume_trainer())
        print(f"Xong. Checkpoint tốt nhất theo nuImages val: {model.trainer.best}")
        return
    if not args.data:
        ap.error("cần --data (hoặc --resume)")
    name = args.name or f"{args.stage}-{args.imgsz}"
    if args.stage == "baseline":
        # zero-shot: trọng số gốc + tên lớp làm prompt, chấm trên nuImages val bằng đúng validator của lúc train
        from ultralytics.utils import YAML

        names = list(YAML.load(args.data)["names"].values())
        model = build(args.weights)  # bản không mask: validator box của dataset detect
        model.set_classes(names, model.get_text_pe(names))
        m = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, task="detect",
                      project=args.project, name=name, workers=args.workers)  # fmt: skip
        print(json.dumps({"stage": "baseline", "mAP50": m.box.map50, "mAP50-95": m.box.map}, indent=1))
        return

    model = build(args.weights)
    common = dict(data=args.data, imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
                  optimizer="AdamW", warmup_bias_lr=0.0, momentum=0.9, weight_decay=0.025, fraction=args.fraction,
                  trainer=YOLOEPETrainer, project=args.project, name=name, exist_ok=False, plots=True)  # fmt: skip
    if args.stage == "lp":
        epochs = args.epochs or 10
        model.train(epochs=epochs, close_mosaic=min(5, epochs), lr0=1e-3, patience=epochs,
                    freeze=linear_probe_freeze(model), **common)  # fmt: skip
    else:
        epochs = args.epochs or 30
        # lr nhỏ hơn mặc định 10 lần: khởi đầu đã tốt, học lại mạnh sẽ quên tri thức open-vocab và dễ overfit
        model.train(epochs=epochs, close_mosaic=min(10, epochs), lr0=2e-4, lrf=0.1, cos_lr=True, patience=8,
                    **common)  # fmt: skip
    best = Path(model.trainer.best)
    print(f"Xong. Checkpoint tốt nhất theo nuImages val: {best}")
    print("Bước tiếp: python tools2d/eval2d.py --weights", best, "(chấm trên scene held-out của nuScenes)")


if __name__ == "__main__":
    main()
