import numpy as np

from src.agents.nodes.confidence import check_confidence
from src.agents.nodes.issues import check_geometry
from src.agents.nodes.lidar import check_lidar
from src.agents.nodes.risk import risk_level, score_risk
from src.agents.nodes.temporal import check_temporal, match_greedy
from src.models.schemas import Detection, LabelObject, QAIssue, QAResult

FY = 1266.4


def obj(oid, bbox, label="car", score=0.9, alternatives=None):
    return LabelObject(object_id=oid, bbox=bbox, label=label, score=score, alternatives=alternatives or {})


def points_in(bbox, depth, n=10):
    xs, ys = np.meshgrid(np.linspace(bbox[0] + 2, bbox[2] - 2, n), np.linspace(bbox[1] + 2, bbox[3] - 2, n))
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return uv, np.full(len(uv), float(depth))


def codes(result):
    return [i.code for i in result["issues"]]


# ---- 3.1 confidence ----


def test_low_confidence(config):
    r = check_confidence(obj("1", [0, 0, 10, 10], score=0.27), config)
    assert codes(r) == ["LOW_CONFIDENCE"]
    assert r["term"] == 1 - 0.27


def test_class_conflict_adds_penalty(config):
    r = check_confidence(obj("1", [0, 0, 10, 10], label="truck", score=0.6, alternatives={"car": 0.5}), config)
    assert "CLASS_CONFLICT" in codes(r)
    assert r["term"] == min(1.0, 0.4 + config.qa.risk.class_conflict_penalty)


def test_weak_alternative_is_not_conflict(config):
    r = check_confidence(obj("1", [0, 0, 10, 10], score=0.8, alternatives={"truck": 0.2}), config)
    assert codes(r) == []


# ---- 3.2 lidar ----


def test_no_lidar_support_on_tall_box(config):
    r = check_lidar([100, 300, 300, 500], "car", np.zeros((0, 2)), np.zeros(0), FY, 900, config)
    assert codes(r) == ["NO_LIDAR_SUPPORT"]
    assert r["term"] == 1.0


def test_small_far_box_without_points_is_flagged(config):
    # Box nhỏ 0 điểm: gắn cờ và lấy no_points_term (04/10/2026: 96% box loại này không khớp nhãn gốc)
    r = check_lidar([100, 300, 120, 320], "car", np.zeros((0, 2)), np.zeros(0), FY, 900, config)
    assert codes(r) == ["NO_LIDAR_SUPPORT"]
    assert r["term"] == config.qa.lidar.no_points_term
    # Tắt cờ bằng config (cách làm cũ): không issue, vẫn cộng rủi ro
    cfg = config.model_copy(deep=True)
    cfg.qa.lidar.zero_points_issue, cfg.qa.lidar.no_points_term = False, 0.2
    r = check_lidar([100, 300, 120, 320], "car", np.zeros((0, 2)), np.zeros(0), FY, 900, cfg)
    assert codes(r) == [] and r["term"] == 0.2


def test_small_far_box_with_few_points_is_not_flagged(config):
    box = [100, 300, 120, 320]
    uv = np.array([[110.0, 310.0], [112.0, 312.0]])
    r = check_lidar(box, "car", uv, np.full(2, 40.0), FY, 900, config)
    assert codes(r) == []
    assert r["term"] == config.qa.lidar.few_points_term


def test_sparse_points_add_some_risk(config):
    box = [500, 400, 700, 500]  # xe cao 100px ở 20 m: kích thước hợp lý, nhưng chỉ 4 điểm
    uv = np.array([[590.0, 440.0], [600.0, 450.0], [610.0, 460.0], [605.0, 445.0]])
    r = check_lidar(box, "car", uv, np.full(4, 20.0), FY, 900, config)
    assert codes(r) == []
    assert r["term"] == config.qa.lidar.sparse_term


def test_plausible_car_height(config):
    box = [500, 400, 700, 500]
    uv, depth = points_in(box, 20.0)
    r = check_lidar(box, "car", uv, depth, FY, 900, config)
    assert codes(r) == []
    assert r["depth_m"] == 20.0
    assert abs(r["est_height_m"] - 100 * 20 / FY) < 0.01


def test_size_depth_mismatch(config):
    box = [500, 200, 600, 500]  # người cao 300px ở 20 m -> ~4.7 m
    uv, depth = points_in(box, 20.0)
    r = check_lidar(box, "pedestrian", uv, depth, FY, 900, config)
    assert codes(r) == ["SIZE_DEPTH_MISMATCH"]


def test_depth_ignores_background_points(config):
    box = [500, 400, 700, 500]
    uv_obj, d_obj = points_in(box, 15.0)
    uv_bg, d_bg = points_in(box, 60.0, n=5)
    r = check_lidar(box, "car", np.vstack([uv_obj, uv_bg]), np.concatenate([d_obj, d_bg]), FY, 900, config)
    assert r["depth_m"] == 15.0


def test_lidar_unavailable(config):
    r = check_lidar([0, 0, 100, 100], "car", None, None, FY, 900, config)
    assert r == {"available": False, "issues": [], "term": 0.0}


# ---- 3.3 temporal ----


def sweep(ts, *dets):
    return {"timestamp": ts, "detections": [Detection(bbox=b, label=lb, score=s) for b, lb, s in dets]}


def test_match_greedy_one_to_one():
    pairs = match_greedy([[0, 0, 10, 10], [0, 0, 11, 11]], [[0, 0, 10, 10]], 0.3)
    assert pairs == {0: 0}


def test_flicker_and_support(config):
    a, b = [100, 100, 200, 200], [500, 500, 600, 600]
    shifted = [104, 100, 204, 200]
    sweeps = {o: sweep(1000 + o * 80, (shifted, "car", 0.8)) for o in (-2, -1, 1, 2)}
    results, recovered = check_temporal([obj("1", a), obj("2", b)], sweeps, 1000, config)
    assert results["1"]["support"] == 4 and codes(results["1"]) == []
    assert results["1"]["track"]["-1"] == shifted
    assert codes(results["2"]) == ["FLICKER"]
    assert results["2"]["term"] == 1.0
    assert recovered == []


def test_recovered_by_track_interpolates(config):
    sweeps = {
        -1: sweep(900, ([800, 300, 900, 400], "car", 0.8)),
        1: sweep(1050, ([810, 300, 910, 400], "car", 0.7)),
    }
    # Mặc định (recover_min_score 1.0): box sweep của riêng detector không tạo đề xuất
    assert check_temporal([], sweeps, 1000, config)[1] == []
    cfg = config.model_copy(deep=True)
    cfg.qa.temporal.recover_min_score = 0.35
    results, recovered = check_temporal([], sweeps, 1000, cfg)
    assert len(recovered) == 1
    r = recovered[0]
    assert r.source == "track" and r.label == "car" and r.score == 0.7
    assert abs(r.bbox[0] - (800 + 10 * 100 / 150)) < 0.1
    assert codes(results[r.object_id]) == ["RECOVERED_BY_TRACK"]


def test_no_recovery_when_keyframe_has_box(config):
    sweeps = {
        -1: sweep(900, ([800, 300, 900, 400], "car", 0.8)),
        1: sweep(1050, ([810, 300, 910, 400], "car", 0.7)),
    }
    _, recovered = check_temporal([obj("1", [805, 300, 905, 400])], sweeps, 1000, config)
    assert recovered == []


def test_no_recovery_for_low_score_track(config):
    sweeps = {
        -1: sweep(900, ([800, 300, 900, 400], "car", 0.2)),
        1: sweep(1050, ([810, 300, 910, 400], "car", 0.7)),
    }
    _, recovered = check_temporal([], sweeps, 1000, config)
    assert recovered == []


# ---- geometric ----


def test_box_too_large(config):
    r = check_geometry(obj("1", [0, 100, 1200, 800]), 1600, 900, config)
    assert "BOX_TOO_LARGE" in codes(r)


def test_aspect_ratio_abnormal_but_not_at_border(config):
    wide_person = obj("1", [500, 400, 800, 500], label="pedestrian")
    assert codes(check_geometry(wide_person, 1600, 900, config)) == ["ASPECT_RATIO_ABNORMAL"]
    truncated = obj("2", [0, 400, 300, 500], label="pedestrian")
    assert codes(check_geometry(truncated, 1600, 900, config)) == []


# ---- 3.5 risk ----


def test_risk_formula_and_levels(config):
    cfg = config.qa.risk
    w = cfg.weights
    qa = QAResult(risk=0, level="low", terms={"detection": 0.61, "lidar": 1.0, "temporal": 1.0, "geometric": 0.0})
    scored = score_risk(qa, cfg)
    expected = (w.detection * 0.61 + w.lidar + w.temporal) / (w.detection + w.lidar + w.temporal + w.geometric)
    assert scored.risk == round(expected, 3)
    assert scored.level == "high"


def test_issue_floor_keeps_flagged_objects_out_of_low(config):
    qa = QAResult(
        risk=0,
        level="low",
        issues=[QAIssue(code="FLICKER", group="temporal", message="")],
        terms={"detection": 0.05},
    )
    scored = score_risk(qa, config.qa.risk)
    assert scored.risk == config.qa.risk.issue_floor
    assert scored.level == "medium"


def test_risk_level_thresholds(config):
    cfg = config.qa.risk
    assert risk_level(0.0, cfg) == "low"
    assert risk_level(cfg.levels.medium, cfg) == "medium"
    assert risk_level(cfg.levels.high, cfg) == "high"

def test_weak_keyframe_detection_confirmed_by_both_sides_is_recovered(config):
    # Xe đạp: keyframe 0.20 (< ngưỡng giữ 0.30), sweep t-1 0.26 / t+1 0.23 (cũng dưới ngưỡng giữ, >= 0.2)
    weak = [Detection(bbox=[2064, 417, 2112, 489], label="bicycle", score=0.2)]
    sweeps = {-1: sweep(900), 1: sweep(1050)}
    sweeps[-1]["weak"] = [Detection(bbox=[2052, 413, 2100, 476], label="bicycle", score=0.26)]
    sweeps[1]["weak"] = [Detection(bbox=[2072, 422, 2128, 486], label="bicycle", score=0.23)]
    assert check_temporal([], sweeps, 1000, config, weak)[1] == []  # mặc định tắt (recover_weak_min_score: null)
    config = config.model_copy(deep=True)
    config.qa.temporal.recover_weak_min_score = 0.2
    results, recovered = check_temporal([], sweeps, 1000, config, weak)
    assert len(recovered) == 1
    r = recovered[0]
    assert r.source == "track" and r.label == "bicycle" and r.bbox == [2064, 417, 2112, 489] and r.score == 0.2
    assert codes(results[r.object_id]) == ["RECOVERED_BY_TRACK"]
    assert "thấy mờ" in results[r.object_id]["issues"][0].message
    # Chỉ một bên có bằng chứng -> không đề xuất
    sweeps[1]["weak"] = []
    assert check_temporal([], sweeps, 1000, config, weak)[1] == []
    # Tắt bằng config
    cfg = config.model_copy(deep=True)
    cfg.qa.temporal.recover_weak_min_score = None
    sweeps[1]["weak"] = [Detection(bbox=[2072, 422, 2128, 486], label="bicycle", score=0.23)]
    assert check_temporal([], sweeps, 1000, cfg, weak)[1] == []


def test_recovered_box_snaps_to_weak_keyframe_detection(config):
    # Xe ở xa: detector thấy 0.58 nhưng score bị hạ còn 0.29 khi gộp với 3D -> bị bỏ; sweep t-1 / t+1 đều thấy rõ
    sweeps = {
        -1: sweep(900, ([715, 447.5, 742.5, 468.8], "car", 0.69)),
        1: sweep(1050, ([708.8, 451.9, 736.2, 474.4], "car", 0.62)),
    }
    weak = [Detection(bbox=[710, 447, 738, 471], label="car", score=0.29, det_score=0.58)]
    config = config.model_copy(deep=True)
    config.qa.temporal.recover_min_score, config.qa.temporal.recover_weak_min_score = 0.35, 0.2
    results, recovered = check_temporal([], sweeps, 1000, config, weak)
    assert len(recovered) == 1
    r = recovered[0]
    assert r.bbox == [710, 447, 738, 471] and r.score == 0.29 and r.det_score == 0.58
    msg = results[r.object_id]["issues"][0].message
    assert "0.58" in msg and "gộp với box 3D" in msg

