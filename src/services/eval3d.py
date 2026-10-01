"""Đánh giá bước kiểm chứng 3D so với nhãn gốc (nhãn gốc chỉ dùng ở đây, không dùng khi kết luận)."""

from __future__ import annotations

import math
from collections import Counter

from src.services.store import WorkspaceStore
from src.services.verify3d import CLASS_RANGE, V_OK, VERDICT_LEVEL

MATCH_DIST = 2.0  # chuẩn nuScenes: cùng lớp, tâm cách nhau < 2 m trên mặt phẳng ngang
REFS = ("dung", "sai lop", "bao nham")


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reference(objects, gts) -> tuple[list[str | None], set[int]]:
    """Nhãn tham chiếu của từng box: đúng (cùng lớp, < 2 m) / sai lớp (có vật khác lớp < 2 m) / báo nhầm; kèm tập GT đã khớp."""
    ref: list[str | None] = [None] * len(objects)
    taken = set()
    for i in sorted(range(len(objects)), key=lambda i: -objects[i].score):
        o = objects[i]
        cands = [
            (_dist(o.box.center, g["center"]), k) for k, g in enumerate(gts) if k not in taken and g["label"] == o.label
        ]
        if cands and min(cands)[0] < MATCH_DIST:
            taken.add(min(cands)[1])
            ref[i] = "dung"
    for i, o in enumerate(objects):
        if ref[i] is None:
            near = [_dist(o.box.center, g["center"]) for g in gts]
            ref[i] = "sai lop" if near and min(near) < MATCH_DIST else "bao nham"
    return ref, taken


def evaluate_verification(store: WorkspaceStore, model: str) -> dict:
    table = {v: Counter() for v in VERDICT_LEVEL}
    missed = Counter()
    n_gt = 0
    for f in store.list_frames3d(model):
        gts = [g for g in (store.load_aux("gt3d", f.frame_id) or []) if g["num_pts"] > 0]
        gts = [g for g in gts if math.hypot(*g["center"][:2]) <= CLASS_RANGE.get(g["label"], 50)]
        n_gt += len(gts)
        objs = [o for o in f.objects if o.verify is not None]
        refs, taken = reference(objs, gts)
        for o, r in zip(objs, refs, strict=True):
            table[o.verify.verdict][r] += 1
        for k, g in enumerate(gts):
            if k not in taken:
                missed[g["label"]] += 1
    n = sum(sum(c.values()) for c in table.values())
    errors = sum(c["sai lop"] + c["bao nham"] for c in table.values())
    auto = table[V_OK]
    auto_n = sum(auto.values())
    flagged = n - auto_n
    flagged_err = errors - auto["sai lop"] - auto["bao nham"]
    return {
        "model": model,
        "n_boxes": n,
        "n_gt": n_gt,
        "table": {v: {r: table[v][r] for r in REFS} for v in VERDICT_LEVEL},
        "error_rate_all": round(errors / n, 4) if n else None,
        "auto_share": round(auto_n / n, 4) if n else 0.0,
        "auto_precision": round(auto["dung"] / auto_n, 4) if auto_n else 0.0,
        "error_recall": round(flagged_err / errors, 4) if errors else 0.0,
        "flag_precision": round(flagged_err / flagged, 4) if flagged else None,
        "gt_missed": sum(missed.values()),
        "gt_missed_by_class": dict(missed),
    }
