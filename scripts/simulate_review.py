"""Mô phỏng người duyệt "chuẩn" (quyết định theo GT nuScenes) để đo công duyệt, so sánh có / không lan truyền.

Cả hai chế độ dùng cùng một cách duyệt mỗi frame:
  - box bị gắn cờ (medium/high): người xem và quyết định đúng theo GT (Keep / đổi lớp / xoá)
  - box low: duyệt theo lô, không xem (lỗi trong nhóm này lọt qua)
  - vật GT không có box đúng: người vẽ thêm
Khác nhau: chế độ "lan truyền" approve xong frame nào thì lan truyền sang các frame sau (như UI khi bật Tự lan truyền).

Chạy khi server đang mở trên workspace đã `run` (mỗi lần chạy trên một bản sao workspace, vì script approve thật):

    python scripts/simulate_review.py http://127.0.0.1:8000 prop eval/results/review_sim/prop@0.3.json
"""

import json
import sys
import time
import urllib.request

BASE, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
API = BASE + "/api/v1"


def call(path, body=None):
    req = urllib.request.Request(
        API + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    return json.loads(urllib.request.urlopen(req).read())


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def box(o):
    return o["review"].get("final_bbox") or o["bbox"]


def lab(o):
    return o["review"].get("final_label") or o["label"]


def best_gt(b, gt):
    best, bi = 0.0, None
    for i, g in enumerate(gt):
        v = iou(b, g["bbox"])
        if v > best:
            best, bi = v, i
    return (bi, best) if best >= 0.5 else (None, best)


def review(fid, stats):
    fr = call(f"/frames/{fid}")
    gt = call(f"/frames/{fid}/gt")
    t0 = time.time()
    for o in fr["objects"]:
        if o["review"]["status"] != "pending" or (o.get("qa") or {}).get("level", "low") == "low":
            continue
        stats["flagged"] += 1
        gi, _ = best_gt(o["bbox"], gt)
        if gi is None:
            act = {"action": "DELETE"}
        elif gt[gi]["ignore"] or gt[gi]["label"] == o["label"]:
            act = {"action": "KEEP"}
        else:
            act = {"action": "CHANGE_CLASS", "label": gt[gi]["label"]}
        stats[act["action"]] += 1
        call(f"/frames/{fid}/actions", {**act, "object_id": o["object_id"]})
    fr = call(f"/frames/{fid}/approve-low-risk", {})
    kept = [o for o in fr["objects"] if o["review"]["status"] == "approved"]
    for g in gt:
        if g["ignore"]:
            continue
        if not any(lab(o) == g["label"] and iou(box(o), g["bbox"]) >= 0.5 for o in kept):
            call(f"/frames/{fid}/actions", {"action": "ADD_BOX", "label": g["label"], "bbox": g["bbox"]})
            stats["ADD_BOX"] += 1
    call(f"/frames/{fid}/approve", {"review_time_s": round(time.time() - t0, 2)})


def final_quality(frames):
    tp = fp = fn = 0
    for f in frames:
        fr = call(f"/frames/{f['frame_id']}")
        gt = call(f"/frames/{f['frame_id']}/gt")
        kept = [o for o in fr["objects"] if o["review"]["status"] == "approved"]
        used = set()
        for o in kept:
            gi, _ = best_gt(box(o), gt)
            if gi is not None and gt[gi]["ignore"]:
                continue
            if gi is not None and gi not in used and gt[gi]["label"] == lab(o):
                tp += 1
                used.add(gi)
            else:
                fp += 1
        fn += sum(1 for i, g in enumerate(gt) if not g["ignore"] and i not in used)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3)}


result = {}
for scene in ("scene-0031", "scene-0065"):
    stats = {"flagged": 0, "KEEP": 0, "DELETE": 0, "CHANGE_CLASS": 0, "ADD_BOX": 0}
    per_frame = []
    frames = call(f"/videos/{scene}")["frames"]
    for f in frames:
        before = dict(stats)
        cur = call(f"/frames/{f['frame_id']}")
        review(f["frame_id"], stats)
        per_frame.append(
            {
                "frame": f["frame_id"],
                "propagated_from": cur.get("propagated_from"),
                **{k: stats[k] - before[k] for k in ("flagged", "DELETE", "CHANGE_CLASS", "ADD_BOX")},
            }
        )
        if MODE == "prop":
            r = call(f"/frames/{f['frame_id']}/propagate", {})
            per_frame[-1]["propagated_to"] = len(r.get("frames_updated") or [])
            per_frame[-1]["stop_reason"] = r.get("stop_reason")
    result[scene] = {"stats": stats, "per_frame": per_frame, "quality": final_quality(frames)}
    print(scene, stats, result[scene]["quality"], flush=True)

json.dump(result, open(OUT, "w"), indent=1, ensure_ascii=False)
