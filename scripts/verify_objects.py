"""Kiem chung tung box 3D (LiDAR) bang camera.

Voi moi box 3D:
  1. Chieu len ca 6 camera.
  2. Cham diem "luong thong tin" cua tung camera: phan nhin thay (khong bi cat mep), muc che khuat,
     do ro (kich thuoc, do net, do sang). Chon camera co diem cao nhat.
  3. Tren camera do: YOLO co thay vat khong, cung lop khong, box co khop khong; kem so diem lidar
     nam trong box 3D.
  4. Ket luan: DUNG / DUNG VAT, BOX LECH / SAI LOP / NGHI BAO NHAM / CAMERA KHONG XAC NHAN /
     CHUA DU THONG TIN.
Nhan goc nuScenes KHONG duoc dung de ket luan, chi in kem (cot "tham chieu") de doi chieu.

Chay:
    python verify_objects.py                                   # demo 1 keyframe, du doan mo phong
    python verify_objects.py --scene scene-0103 --keyframe 15
    python verify_objects.py --all                             # ca 81 keyframe cua mini_val
    python verify_objects.py --results path/to/results_nusc.json   # ket qua PointPillars that
    python verify_objects.py --yoloe yoloe-26s-seg.pt          # YOLOE open-vocabulary, du 10 lop nuScenes
Can: pip install ultralytics (kem torch); YOLOE can them CLIP (ultralytics tu cai lan dau).

Ket qua (out/verify/):
    <scene>_kf<k>_overview.png  - 6 camera, moi box 3D ve o camera duoc chon, mau theo ket luan
    <scene>_kf<k>_cards.png     - anh cat tung vat o camera duoc chon, kem ket luan
    ket_luan_<scene>_kf<k>.csv  - moi vat mot dong (--all: ket_luan_mini_val.csv)
"""
import argparse
import collections
import csv
import functools
import json
import os
import os.path as osp
import sys
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path  # noqa: E402

# Doi duong dan bang bien moi truong khi dung o noi khac (vd. Colab: NUSC_DATAROOT=/content/nuscenes).
DATAROOT = os.environ.get('NUSC_DATAROOT', r'D:\AI_Action_Project')
VERSION = 'v1.0-mini'
OUT_DIR = os.environ.get('VERIFY_OUT_DIR', osp.join(DATAROOT, 'out', 'verify'))

if osp.isdir(osp.join(DATAROOT, 'nuscenes-devkit', 'python-sdk')):     # ban devkit trong thu muc du an
    sys.path.insert(0, osp.join(DATAROOT, 'nuscenes-devkit', 'python-sdk'))
from pyquaternion import Quaternion  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.eval.detection.config import config_factory  # noqa: E402
from nuscenes.eval.detection.utils import category_to_detection_name  # noqa: E402
from nuscenes.utils.data_classes import Box, LidarPointCloud  # noqa: E402
from nuscenes.utils.geometry_utils import points_in_box, transform_matrix, view_points  # noqa: E402
from nuscenes.utils.splits import create_splits_scenes  # noqa: E402

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']       # thu tu nay = luoi 2 x 3 khi ve
CLASS_RANGE = config_factory('detection_cvpr_2019').class_range   # pham vi cham diem theo lop (m)

# --- Chieu box va chon camera ---
NEAR = 0.1              # mat phang gan truoc camera (m)
MIN_AREA = 100          # box 2D nho hon (px^2) thi khong xet camera do
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
SIZE_REF = 60.0         # canh box (px) tu muc nay tro len coi la du lon
SHARP_REF = 60.0        # phuong sai Laplacian tu muc nay tro len coi la du net
LIGHT_REF = 60.0        # do sang trung binh (0-255) tu muc nay tro len coi la du sang
CLEAR_OCC = 0.3         # "anh ro" = che < 30% ...
CLEAR_INFO = 0.5        # ... va diem thong tin >= 0.5; chi khi do moi dam ket luan camera KHONG thay vat

# --- Che khuat ---
MASK_SCALE = 1 / 8      # to mat na o 1/8 do phan giai cho nhanh
OCC_MARGIN = 1.0        # diem lidar gan camera hon mat truoc cua box qua 1 m moi tinh la vat che
MIN_LIDAR_PTS = 5       # it diem hon trong vung box thi khong dung cach lidar
DUP_DIST = 1.5          # hai box 3D cach nhau < 1.5 m coi la trung, khong tinh la che nhau

# --- Bo phat hien 2D ---
NUS_CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian',
               'motorcycle', 'bicycle', 'traffic_cone', 'barrier']           # 10 lop detection cua nuScenes

# YOLO thuong (80 lop COCO): chi 6 lop khop voi nuScenes.
COCO_TO_NUS = {'person': 'pedestrian', 'bicycle': 'bicycle', 'car': 'car',
               'motorcycle': 'motorcycle', 'bus': 'bus', 'truck': 'truck'}

# YOLOE (open-vocabulary): moi lop nuScenes la mot nhom prompt, chon theo dinh nghia lop cua nuScenes.
NUS_PROMPTS = {
    'car':                  ['car', 'sedan', 'suv', 'van', 'minivan'],             # nuScenes: car gom SUV, van
    'truck':                ['truck', 'pickup truck', 'box truck', 'lorry', 'semi truck'],   # ban tai la truck
    'bus':                  ['bus', 'city bus'],
    'trailer':              ['trailer', 'semi-trailer', 'cargo trailer'],
    'construction_vehicle': ['construction vehicle', 'excavator', 'bulldozer', 'crane', 'forklift'],
    'pedestrian':           ['pedestrian', 'person', 'child', 'construction worker', 'police officer'],
    'motorcycle':           ['motorcycle', 'motor scooter', 'moped', 'motorcyclist'],  # box bao ca nguoi lai
    'bicycle':              ['bicycle', 'cyclist'],                                     # box bao ca nguoi lai
    'traffic_cone':         ['traffic cone'],
    'barrier':              ['traffic barrier', 'concrete barrier', 'road barrier'],
}
# Prompt "doi thu": vat de bi nham voi cac lop tren (lan can co dinh khong phai barrier cua nuScenes),
# hoac vat nuScenes khong tinh (xe day, xe lan, scooter dung). Box cua chung bi bo, khong doi ve lop nao.
# Khong dung 'ambulance' / 'police car': thu thay chung hut mat xe tai / xe van that.
DISTRACTOR_PROMPTS = ['fence', 'guardrail', 'pole', 'traffic sign', 'fire hydrant', 'trash can', 'bollard',
                      'mailbox', 'stroller', 'wheelchair', 'kick scooter']

# Cau hinh bo phat hien dang dung (mac dinh: YOLO COCO). setup_open_vocab() doi sang YOLOE.
#   label_map: ten lop cua mo hinh -> lop nuScenes (None = bo)
#   checkable: lop nuScenes camera kiem duoc lop; matchable: lop nuScenes dem di so khop
#   class_conf: nguong diem tin cay rieng cho tung lop nuScenes
DETECTOR = {'kind': 'YOLO (COCO)', 'label_map': dict(COCO_TO_NUS),
            'checkable': set(COCO_TO_NUS.values()),
            'matchable': set(COCO_TO_NUS.values()) | {'trailer', 'construction_vehicle'},   # YOLO goi la truck
            'class_conf': {}}


def setup_open_vocab(model, prompts=None, distractors=None, class_conf=None):
    """Dat lop cho YOLOE bang prompt (du 10 lop nuScenes) va chuyen bo kiem chung sang che do nay."""
    prompts = NUS_PROMPTS if prompts is None else prompts
    distractors = DISTRACTOR_PROMPTS if distractors is None else distractors
    names = [p for ps in prompts.values() for p in ps] + list(distractors)
    assert len(set(names)) == len(names), 'Co prompt bi khai bao hai lan'
    model.set_classes(names)
    label_map = {p: c for c, ps in prompts.items() for p in ps}
    label_map.update({p: None for p in distractors})
    DETECTOR.update(kind='YOLOE (open-vocabulary)', label_map=label_map, checkable=set(prompts),
                    matchable=set(prompts), class_conf=dict(class_conf or {}))
    return names


RIDER_IOA = 0.3         # >= 30% dien tich nguoi nam trong box xe dap / xe may: coi la nguoi lai
IOU_THR = 0.4           # khop kieu IoU
IOA_THR = 0.7           # khop mot phan: >= 70% box YOLO nam trong box 3D ...
COVER_THR = 0.4         # ... va phu >= 40% phan nhin thay duoc cua box 3D
NEAR_Z = 8.0            # vat gan hon 8 m: box chieu phinh to hon vat that nhieu
DEPTH_MARGIN = 1.5      # sai so cho phep khi so do sau lidar (m)
CLASS_HEIGHT = {'car': 1.6, 'truck': 2.8, 'bus': 3.2, 'trailer': 3.5, 'construction_vehicle': 3.0,
                'pedestrian': 1.7, 'bicycle': 1.3, 'motorcycle': 1.3, 'traffic_cone': 0.7, 'barrier': 1.0}
RIDER_HEIGHT = 1.7
SHIFT_THR = 0.35        # tam lech ngang / day lech doc > 35% kich thuoc box YOLO: box lech
MIN_PTS_IN_BOX = 3      # so diem lidar toi thieu trong box 3D de coi la "co vat"
GT_DIST = 2.0           # doi chieu voi nhan goc: tam cach < 2 m (chuan nuScenes)

V_OK = 'DUNG'
V_SHIFT = 'DUNG VAT, BOX LECH'
V_CLASS = 'SAI LOP'
V_FP = 'NGHI BAO NHAM'
V_MISS = 'CAMERA KHONG XAC NHAN'
V_UNSURE = 'CHUA DU THONG TIN'
VERDICTS = [V_OK, V_SHIFT, V_CLASS, V_FP, V_MISS, V_UNSURE]
V_COLOR = {V_OK: 'limegreen', V_SHIFT: 'gold', V_CLASS: 'orange', V_FP: 'red',
           V_MISS: 'magenta', V_UNSURE: 'silver'}


# =============================================================================================
# Du lieu
# =============================================================================================
def ego_xy(nusc, sample):
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    return np.array(nusc.get('ego_pose', sd['ego_pose_token'])['translation'][:2])


def bike_racks(nusc, sample):
    return [nusc.get_box(t) for t in sample['anns']
            if nusc.get('sample_annotation', t)['category_name'] == 'static_object.bicycle_rack']


def keep_box(name, center, xy, racks):
    """Giong bo cham diem nuScenes: trong pham vi cua lop, bo xe dap / xe may trong bai do xe dap."""
    if np.linalg.norm(np.asarray(center[:2]) - xy) > CLASS_RANGE[name]:
        return False
    return not (name in ('bicycle', 'motorcycle') and
                any(points_in_box(r, np.asarray(center, float).reshape(3, 1))[0] for r in racks))


def load_gt(nusc, sample):
    """Nhan goc thuoc 10 lop detection (chi de doi chieu, khong dung de ket luan)."""
    out = []
    for tok in sample['anns']:
        ann = nusc.get('sample_annotation', tok)
        name = category_to_detection_name(ann['category_name'])
        if name is None:
            continue
        b = nusc.get_box(tok)
        b.name, b.num_pts = name, ann['num_lidar_pts'] + ann['num_radar_pts']
        out.append(b)
    return out


SWAP = {'car': 'truck', 'truck': 'car', 'pedestrian': 'bicycle', 'bicycle': 'motorcycle',
        'motorcycle': 'bicycle', 'bus': 'truck', 'traffic_cone': 'barrier', 'barrier': 'traffic_cone',
        'trailer': 'truck', 'construction_vehicle': 'truck'}


def simulate_predictions(nusc, sample, rng):
    """Du doan mo phong tu nhan goc, co gai loi de kiem tra: lech vi tri (moi box), 10% bo sot,
    8% sai lop, 5% lech lon (1.5-3 m), them 2 box bao nham moi keyframe."""
    xy, out = ego_xy(nusc, sample), []
    for b in load_gt(nusc, sample):
        if b.num_pts == 0 or rng.random() < 0.10:
            continue
        p = b.copy()
        p.name = SWAP[b.name] if rng.random() < 0.08 else b.name
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(1.5, 3.0) if rng.random() < 0.05 else abs(rng.normal(0, 0.25))
        p.translate(np.array([np.cos(ang), np.sin(ang), 0.0]) * dist)
        p.score = float(rng.uniform(0.3, 0.9))
        out.append(p)
    z = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])['translation'][2]
    for _ in range(2):
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(8, 35)
        out.append(Box([xy[0] + r * np.cos(ang), xy[1] + r * np.sin(ang), z + 0.8], [1.9, 4.5, 1.6],
                       Quaternion(axis=[0, 0, 1], angle=float(rng.uniform(0, 2 * np.pi))),
                       name='car', score=float(rng.uniform(0.3, 0.6))))
    return out


def load_predictions(nusc, samples, results_path, score_thr, seed):
    """{sample token: [Box he toan cuc]}, da loc theo diem tin cay, pham vi cua lop, bai do xe dap."""
    rng = np.random.default_rng(seed)
    if results_path:
        with open(results_path) as f:
            results = json.load(f)['results']
    out = {}
    for s in samples:
        if results_path:
            boxes = [Box(d['translation'], d['size'], Quaternion(d['rotation']),
                         name=d['detection_name'], score=d['detection_score'])
                     for d in results[s['token']] if d['detection_score'] >= score_thr]
        else:
            boxes = simulate_predictions(nusc, s, rng)
        xy, racks = ego_xy(nusc, s), bike_racks(nusc, s)
        out[s['token']] = [b for b in boxes if keep_box(b.name, b.center, xy, racks)]
    return out


def lidar_points(nusc, sample, nsweeps):
    """Diem lidar cua keyframe gop voi cac lan quet ngay truoc (da bu chuyen dong cua xe minh),
    trong he toan cuc (3, N). Mot lan quet rat thua o 20-30 m: nguoi di bo chi con 1-2 diem."""
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)
    pc.remove_close(1.0)
    tf = (transform_matrix(pose['translation'], Quaternion(pose['rotation']))
          @ transform_matrix(cs['translation'], Quaternion(cs['rotation'])))
    return tf[:3, :3] @ pc.points[:3] + tf[:3, 3:4]


def count_points_in_box(box, pts):
    """So diem lidar trong box 3D, bo lop mong sat day box (diem mat duong): 20 cm, hoac 15% chieu
    cao voi vat thap nhu coc giao thong."""
    local = box.orientation.inverse.rotation_matrix @ (pts - box.center.reshape(3, 1))
    w, l, h = box.wlh
    inside = ((np.abs(local[0]) <= l / 2) & (np.abs(local[1]) <= w / 2) &
              (local[2] <= h / 2) & (local[2] >= -h / 2 + min(0.2, 0.15 * h)))
    return int(inside.sum())


# =============================================================================================
# Chieu box, che khuat, do ro
# =============================================================================================
class Cam:
    """Thong so mot anh camera: cs, ego_pose luc chup, K, kich thuoc, ma tran global -> camera."""

    def __init__(self, nusc, cam_token):
        sd = nusc.get('sample_data', cam_token)
        self.cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        self.pose = nusc.get('ego_pose', sd['ego_pose_token'])
        self.K = np.array(self.cs['camera_intrinsic'])
        self.w, self.h = sd['width'], sd['height']
        self.path = nusc.get_sample_data_path(cam_token)
        self.cam_from_global = (
            transform_matrix(self.cs['translation'], Quaternion(self.cs['rotation']), inverse=True)
            @ transform_matrix(self.pose['translation'], Quaternion(self.pose['rotation']), inverse=True))

    def box_to_cam(self, box):
        b = box.copy()
        b.translate(-np.array(self.pose['translation']))
        b.rotate(Quaternion(self.pose['rotation']).inverse)
        b.translate(-np.array(self.cs['translation']))
        b.rotate(Quaternion(self.cs['rotation']).inverse)
        return b

    def points_to_image(self, pts_global):
        """(u, v, do sau) cua cac diem nam trong anh."""
        p = self.cam_from_global[:3, :3] @ pts_global + self.cam_from_global[:3, 3:4]
        p = p[:, p[2] > NEAR]
        uv = view_points(p, self.K, normalize=True)[:2]
        ok = (uv[0] >= 0) & (uv[0] < self.w) & (uv[1] >= 0) & (uv[1] < self.h)
        return uv[0, ok], uv[1, ok], p[2, ok]


def project_box(box_cam, cam):
    """Box trong he camera -> thong tin 2D, hoac None neu khong lot vao anh. Cat box theo mat phang
    gan truoc khi chieu, vi goc nam sau camera chieu ra toa do sai."""
    k = box_cam.corners()
    front = k[2] > NEAR
    if not front.any():
        return None
    pts = [k[:, i] for i in range(8) if front[i]]
    for i, j in EDGES:
        if front[i] != front[j]:
            t = (NEAR - k[2, i]) / (k[2, j] - k[2, i])
            pts.append(k[:, i] + t * (k[:, j] - k[:, i]))
    pts = np.array(pts).T
    uv = view_points(pts, cam.K, normalize=True)[:2]
    x1, y1 = uv.min(axis=1)
    x2, y2 = uv.max(axis=1)
    cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, cam.w), min(y2, cam.h)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    area = (cx2 - cx1) * (cy2 - cy1)
    return {'bbox': [float(cx1), float(cy1), float(cx2), float(cy2)],
            'vis': float(area / max((x2 - x1) * (y2 - y1), 1e-9)),   # phan nam trong anh
            'area': float(area), 'depth': float(box_cam.center[2]),
            'zmin': float(pts[2].min()), 'zmax': float(pts[2].max()),
            'hull': cv2.convexHull(uv.T.astype(np.float32)).reshape(-1, 2)}


def _mask(hull, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.clip(hull * MASK_SCALE, -5000, 5000).round().astype(np.int32), 1)
    return m.astype(bool)


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def add_occlusion(views, boxes, lidar_uvz, cam):
    """views: {chi so box: view} tren mot camera. Ghi 'occ' (muc che khuat, 0..1):
    - vung box co >= MIN_LIDAR_PTS diem lidar: ty le diem gan camera hon mat truoc cua box qua 1 m
      (do truc tiep vat chan tren duong nhin);
    - it diem hon: phan box bi box 3D khac phu len, voi box che phai gan camera hon it nhat 1 m
      (vat dung ngang hang canh nhau khong tinh la che).
    Chi dung du doan va lidar, khong dung nhan goc."""
    shape = (int(np.ceil(cam.h * MASK_SCALE)), int(np.ceil(cam.w * MASK_SCALE)))
    masks = {i: _mask(v['hull'], shape) for i, v in views.items()}
    u, vv, z = lidar_uvz
    for i, v in views.items():
        x1, y1, x2, y2 = v['bbox']
        sel = (u >= x1) & (u <= x2) & (vv >= y1) & (vv <= y2)
        inside = Path(v['hull']).contains_points(np.c_[u[sel], vv[sel]]) if sel.any() else []
        zin = z[sel][inside]
        if len(zin) >= MIN_LIDAR_PTS:
            v.update(occ=float((zin < v['zmin'] - OCC_MARGIN).mean()), occ_src='lidar')
            continue
        closer = np.zeros(shape, bool)
        for j, v2 in views.items():
            if (j != i and v2['depth'] < v['depth'] - OCC_MARGIN and _overlap(v['bbox'], v2['bbox'])
                    and np.linalg.norm(boxes[j].center - boxes[i].center) >= DUP_DIST):
                closer |= masks[j]
        n_own = masks[i].sum()
        v.update(occ=float((masks[i] & closer).sum() / n_own) if n_own else 0.0, occ_src='box')


def add_clarity(views, gray):
    """Do ro = kich thuoc x do net x do sang cua vung box tren anh xam."""
    for v in views.values():
        x1, y1, x2, y2 = (int(round(c)) for c in v['bbox'])
        crop = gray[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)]
        size = min(1.0, np.sqrt(v['area']) / SIZE_REF)
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        light = float(crop.mean()) if crop.size else 0.0
        v.update(size=size, sharp=sharp, light=light,
                 clarity=size * max(0.3, min(1.0, sharp / SHARP_REF)) * max(0.3, min(1.0, light / LIGHT_REF)))
        v['info'] = v['vis'] * (1.0 - v['occ']) * v['clarity']


# =============================================================================================
# YOLO va so khop
# =============================================================================================
def _area(b):
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _ioa(small, big):
    return _inter(small, big) / max(_area(small), 1e-9)


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def merge_riders(dets):
    """nuScenes: box xe dap / xe may bao ca nguoi lai. Gop box 'person' dang ngoi tren xe vao box xe."""
    drop = set()
    for i, p in enumerate(dets):
        if p['name'] != 'pedestrian':
            continue
        cx = (p['bbox'][0] + p['bbox'][2]) / 2
        cands = [(_ioa(p['bbox'], c['bbox']), c) for c in dets
                 if c['name'] in ('bicycle', 'motorcycle') and not c['rider'] and c['bbox'][0] <= cx <= c['bbox'][2]]
        ioa, cyc = max(cands, key=lambda t: t[0], default=(0.0, None))
        if cyc is not None and ioa >= RIDER_IOA:
            a, b = cyc['bbox'], p['bbox']
            cyc.update(bbox=[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                       rider=True, conf=max(cyc['conf'], p['conf']))
            drop.add(i)
    return [d for i, d in enumerate(dets) if i not in drop]


def run_yolo(model, cams, imgsz, conf):
    """{cam: [box 2D]} cho 6 anh cua mot keyframe, lop da doi ve nuScenes theo DETECTOR.
    agnostic_nms: prompt gan nghia (car / suv) hoac prompt doi thu khong tao nhieu box cho cung vat."""
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    label_map, class_conf = DETECTOR['label_map'], DETECTOR['class_conf']
    keep = [i for i, n in names.items() if n in label_map]
    res = model.predict([cams[c].path for c in CAMERAS], imgsz=imgsz, conf=min([conf, *class_conf.values()]),
                        classes=keep, agnostic_nms=True, verbose=False)
    out = {}
    for c, r in zip(CAMERAS, res):
        dets = []
        for xyxy, cf, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            label = names[int(k)]
            name = label_map[label]
            if name is not None and cf >= class_conf.get(name, conf):
                dets.append({'bbox': xyxy, 'conf': cf, 'name': name, 'label': label, 'rider': False})
        out[c] = merge_riders(dets)
    return out


def yolo_depth(d, lidar_uvz, cam):
    """(do sau, sai so cho phep) cua box YOLO: phan vi 30% do sau lidar trong box; khong du diem thi
    uoc luong tu chieu cao box (xe mau den thuong khong co diem lidar)."""
    u, v, z = lidar_uvz
    x1, y1, x2, y2 = d['bbox']
    sel = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    if sel.sum() >= 3:
        return float(np.percentile(z[sel], 30)), DEPTH_MARGIN
    if y1 <= 1 or y2 >= cam.h - 1:
        return np.nan, np.nan
    zz = cam.K[1, 1] * (RIDER_HEIGHT if d['rider'] else CLASS_HEIGHT[d['name']]) / max(y2 - y1, 1.0)
    return zz, 0.3 * zz


def pair_scores(views, names, dets, depths):
    """Diem ghep (N, M) giua box 3D chieu xuong va box YOLO; > 0 la ghep duoc.
    IoU >= IOU_THR: diem = IoU. Khop mot phan (box 3D bi che / cat mep / o gan, box YOLO nam gon trong
    box 3D va phu du phan nhin thay, do sau khop): diem < 0.05. Cap khac lop phai co do sau khop."""
    iou = iou_matrix([v['bbox'] for v in views], [d['bbox'] for d in dets])
    score = np.where(iou >= IOU_THR, iou, 0.0)
    for i, v in enumerate(views):
        partial = v['occ'] >= 0.3 or v['vis'] < 0.95 or v['zmin'] < NEAR_Z
        visible_area = _area(v['bbox']) * max(1.0 - v['occ'], 0.2)
        for j, d in enumerate(dets):
            z, margin = depths[j]
            depth_ok = not np.isnan(z) and v['zmin'] - margin <= z <= v['zmax'] + margin
            if score[i, j] > 0:
                if names[i] != d['name'] and not np.isnan(z) and not depth_ok:
                    score[i, j] = 0.0
                continue
            if (partial and depth_ok and _ioa(d['bbox'], v['bbox']) >= IOA_THR
                    and _inter(d['bbox'], v['bbox']) / visible_area >= COVER_THR):
                score[i, j] = 0.01 + 0.01 * _ioa(d['bbox'], v['bbox'])
    return score, iou


def match_camera(views, boxes, dets, depths):
    """Hungarian tren mot camera: {chi so box: (chi so box YOLO, IoU, kieu khop)}."""
    from scipy.optimize import linear_sum_assignment
    idx = [i for i in views if boxes[i].name in DETECTOR['matchable']]
    if not idx or not dets:
        return {}
    score, iou = pair_scores([views[i] for i in idx], [boxes[i].name for i in idx], dets, depths)
    same = np.array([[boxes[i].name == d['name'] for d in dets] for i in idx])
    cost = np.where(score > 0, -(score + 0.05 * same), 1e6)
    out = {}
    for a, j in zip(*linear_sum_assignment(cost)):
        if score[a, j] > 0:
            out[idx[a]] = (int(j), float(iou[a, j]), 'IoU' if iou[a, j] >= IOU_THR else 'mot phan')
    return out


# =============================================================================================
# Ket luan
# =============================================================================================
def box_shift(v, d):
    """Do lech giua box 3D chieu xuong va box YOLO: (tam ngang, day) theo kich thuoc box YOLO."""
    (a1, b1, a2, b2), (c1, d1, c2, d2) = v['bbox'], d['bbox']
    w, h = max(c2 - c1, 1.0), max(d2 - d1, 1.0)
    return abs((a1 + a2) / 2 - (c1 + c2) / 2) / w, abs(b2 - d2) / h


def reference(boxes, gts):
    """Doi chieu voi nhan goc (chi de tham chieu): khop cung lop tam < 2 m (tham lam theo diem),
    neu khong thi nhan goc gan nhat bat ky lop."""
    ref, taken = [None] * len(boxes), set()
    for i in np.argsort([-b.score for b in boxes], kind='stable'):
        d = [(np.linalg.norm(boxes[i].center[:2] - g.center[:2]), k) for k, g in enumerate(gts)
             if k not in taken and g.name == boxes[i].name]
        if d and min(d)[0] < GT_DIST:
            taken.add(min(d)[1])
            ref[i] = 'dung (co vat that, cung lop)'
    for i, b in enumerate(boxes):
        if ref[i] is None:
            d = [(np.linalg.norm(b.center[:2] - g.center[:2]), g.name) for g in gts]
            ref[i] = (f'sai lop (nhan goc: {min(d)[1]})' if d and min(d)[0] < GT_DIST
                      else 'khong co vat that (bao nham)')
    return ref


def conclude(box, best_cam, v, match, d, mv, n_pts):
    """(ket luan, nhan xet) cho mot box 3D. v: view o camera duoc chon; match = (camera, (j, iou, kieu))
    neu YOLO khop; d, mv: box YOLO va view cua box 3D o camera khop."""
    pts_txt = f'{n_pts} diem lidar trong box'
    if best_cam is None:
        return V_UNSURE, f'Khong lot ro vao camera nao (chi lo o mep / qua nho). {pts_txt}.'
    view_txt = f'{best_cam}, nhin thay {v["vis"]:.0%}, che {v["occ"]:.0%}, do ro {v["clarity"]:.2f}'
    if match is not None:
        cam, (j, iou, kind) = match
        where = '' if cam == best_cam else f' (xac nhan o {cam})'
        if box.name not in DETECTOR['checkable']:
            return V_OK, (f'Camera thay vat{where} (YOLO goi la {d["name"]}); lop {box.name} YOLO khong co '
                          f'nen khong kiem duoc lop. {pts_txt}.')
        label = d.get('label', d['name'])                  # prompt / lop goc cua mo hinh 2D
        said = d['name'] if label.replace(' ', '_') == d['name'] else f"{d['name']} ('{label}')"
        if d['name'] != box.name:
            return V_CLASS, (f'Camera thay dung vi tri{where} nhung goi la {said} ({d["conf"]:.2f}), '
                             f'3D goi la {box.name}. {pts_txt}.')
        if kind == 'mot phan':
            return V_OK, f'Camera xac nhan{where}, vat bi che / cat mep nen chi khop mot phan. {pts_txt}.'
        dx, dy = box_shift(mv, d)
        if iou < 0.5 and (dx > SHIFT_THR or dy > SHIFT_THR):
            return V_SHIFT, (f'Camera thay dung vat, dung lop{where}, nhung box 3D lech '
                             f'(IoU {iou:.2f}, tam lech {dx:.0%}, day lech {dy:.0%} kich thuoc). {pts_txt}.')
        return V_OK, f'Camera xac nhan dung vat, dung lop{where} (IoU {iou:.2f}). {pts_txt}.'
    if box.name not in DETECTOR['checkable']:
        # Camera khong kiem duoc lop nay, nen khong du co so de ket luan bao nham.
        return V_UNSURE, f'YOLO khong co lop {box.name} nen camera khong kiem duoc. {pts_txt}.'
    if v['occ'] >= CLEAR_OCC or v['info'] < CLEAR_INFO:
        return V_UNSURE, f'Anh tot nhat van kem ({view_txt}); YOLO khong thay. {pts_txt}.'
    if n_pts < MIN_PTS_IN_BOX:
        return V_FP, f'Anh ro ({view_txt}) nhung YOLO khong thay gi, va {pts_txt}: nhieu kha nang bao nham.'
    return V_MISS, (f'Anh ro ({view_txt}) nhung YOLO khong thay; lidar van co {n_pts} diem. '
                    'Co the YOLO bo sot, hoac box 3D sai vi tri / kich thuoc.')


# =============================================================================================
# Xu ly mot keyframe
# =============================================================================================
def verify_sample(nusc, sample, boxes, model, args):
    cams = {c: Cam(nusc, sample['data'][c]) for c in CAMERAS}
    pts = lidar_points(nusc, sample, args.sweeps)
    lidar_uvz = {c: cams[c].points_to_image(pts) for c in CAMERAS}
    dets = run_yolo(model, cams, args.imgsz, args.conf)
    xy = ego_xy(nusc, sample)

    # 1) Chieu moi box len 6 camera, tinh che khuat va do ro.
    views = {c: {} for c in CAMERAS}
    for c, cam in cams.items():
        for i, b in enumerate(boxes):
            v = project_box(cam.box_to_cam(b), cam)
            if v is not None and v['area'] >= MIN_AREA:
                views[c][i] = v
        add_occlusion(views[c], boxes, lidar_uvz[c], cam)
        add_clarity(views[c], cv2.imread(cam.path, cv2.IMREAD_GRAYSCALE))

    # 2) So khop voi YOLO tren tung camera.
    matches = collections.defaultdict(dict)      # box -> {cam: (j, iou, kieu)}
    for c, cam in cams.items():
        depths = [yolo_depth(d, lidar_uvz[c], cam) for d in dets[c]]
        for i, m in match_camera(views[c], boxes, dets[c], depths).items():
            matches[i][c] = m

    # 3) Chon camera nhieu thong tin nhat, ket luan.
    refs = reference(boxes, [g for g in load_gt(nusc, sample)])
    rows = []
    for i, b in enumerate(boxes):
        ranked = sorted(((views[c][i]['info'], c) for c in CAMERAS if i in views[c]), reverse=True)
        best_cam = ranked[0][1] if ranked else None
        v = views[best_cam][i] if best_cam else None
        match = None
        if best_cam in matches[i]:
            match = (best_cam, matches[i][best_cam])
        elif matches[i]:
            c = max(matches[i], key=lambda c: views[c][i]['info'])
            match = (c, matches[i][c])
        n_pts = count_points_in_box(b, pts)
        d = dets[match[0]][match[1][0]] if match else None
        mv = views[match[0]][i] if match else None
        verdict, comment = conclude(b, best_cam, v, match, d, mv, n_pts)
        rows.append({
            'id': i, 'lop_3d': b.name, 'diem_3d': round(b.score, 2),
            'khoang_cach': round(float(np.linalg.norm(b.center[:2] - xy)), 1),
            'camera_chon': best_cam or '-',
            'cac_camera': ' > '.join(f'{c}:{s:.2f}' for s, c in ranked) or '-',
            'nhin_thay': round(v['vis'], 2) if v else None, 'che': round(v['occ'], 2) if v else None,
            'do_ro': round(v['clarity'], 2) if v else None, 'diem_thong_tin': round(v['info'], 2) if v else None,
            'lop_yolo': d['name'] if d else None, 'nhan_yolo': d.get('label') if d else None,
            'conf_yolo': round(d['conf'], 2) if d else None,
            'iou': round(match[1][1], 2) if match else None, 'kieu_khop': match[1][2] if match else None,
            'camera_khop': match[0] if match else None, 'diem_lidar': n_pts,
            'ket_luan': verdict, 'nhan_xet': comment, 'tham_chieu_nhan_goc': refs[i],
            'bbox_3d': [round(x) for x in v['bbox']] if v else None,
            'bbox_yolo': [round(x) for x in d['bbox']] if d else None,
        })
    return rows, cams, dets


# =============================================================================================
# Hinh ve
# =============================================================================================
def _rect(ax, b, color, ls='-', lw=2.0, dx=0, dy=0):
    ax.add_patch(patches.Rectangle((b[0] - dx, b[1] - dy), b[2] - b[0], b[3] - b[1], fill=False,
                                   edgecolor=color, linestyle=ls, linewidth=lw))


def draw_overview(rows, cams, title, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(24, 10.5))
    for ax, c in zip(axes.ravel(), CAMERAS):
        ax.imshow(plt.imread(cams[c].path))
        sub = [r for r in rows if r['camera_chon'] == c]
        for r in sub:
            color = V_COLOR[r['ket_luan']]
            _rect(ax, r['bbox_3d'], color)
            if r['bbox_yolo'] and r['camera_khop'] == c:
                _rect(ax, r['bbox_yolo'], 'white', '--', 1.2)
            ax.text(r['bbox_3d'][0], r['bbox_3d'][1] - 4, f"#{r['id']} {r['lop_3d']}", fontsize=7.5,
                    bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))
        ax.set_title(f'{c}: {len(sub)} vat chon camera nay', fontsize=11)
        ax.set_xlim(0, cams[c].w)
        ax.set_ylim(cams[c].h, 0)
        ax.axis('off')
    handles = [Line2D([], [], color=V_COLOR[k], lw=3, label=k) for k in VERDICTS]
    handles.append(Line2D([], [], color='gray', lw=1.2, ls='--', label='box YOLO khop'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=11, frameon=False)
    fig.suptitle(title + '  |  moi vat chi ve o camera nhieu thong tin nhat, net lien = box 3D chieu xuong',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=3.0)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def draw_cards(rows, cams, title, out_png, ncol=6):
    rows = [r for r in rows if r['bbox_3d']]
    if not rows:
        return
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.9 * nrow), squeeze=False)
    imgs = {}
    for ax in axes.ravel():
        ax.axis('off')
    for ax, r in zip(axes.ravel(), rows):
        c = r['camera_chon']
        img = imgs.setdefault(c, plt.imread(cams[c].path))
        bs = [r['bbox_3d']] + ([r['bbox_yolo']] if r['bbox_yolo'] and r['camera_khop'] == c else [])
        x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs)
        x2, y2 = max(b[2] for b in bs), max(b[3] for b in bs)
        pad = 0.3 * max(x2 - x1, y2 - y1) + 25
        cx1, cy1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
        cx2, cy2 = int(min(x2 + pad, cams[c].w)), int(min(y2 + pad, cams[c].h))
        ax.imshow(img[cy1:cy2, cx1:cx2])
        _rect(ax, r['bbox_3d'], V_COLOR[r['ket_luan']], dx=cx1, dy=cy1)
        if len(bs) == 2:
            _rect(ax, r['bbox_yolo'], 'white', '--', 1.3, dx=cx1, dy=cy1)
        label = r['nhan_yolo'] if r['nhan_yolo'] and r['nhan_yolo'].replace(' ', '_') != r['lop_yolo'] else ''
        yolo = (f"2D {r['lop_yolo']}{f' ({label})' if label else ''} {r['conf_yolo']:.2f}" if r['lop_yolo']
                else '2D: khong thay')
        ax.set_title(f"#{r['id']} 3D {r['lop_3d']} {r['diem_3d']:.2f} | {r['khoang_cach']:.0f} m\n"
                     f"{c[4:]} che {r['che']:.0%} ro {r['do_ro']:.2f} | lidar {r['diem_lidar']}\n"
                     f"{yolo}\n{r['ket_luan']}", fontsize=8.5, color='black',
                     bbox=dict(facecolor=V_COLOR[r['ket_luan']], alpha=0.35, pad=2, edgecolor='none'))
    fig.suptitle(title + '  |  net lien = box 3D chieu xuong, net dut trang = box YOLO khop', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=75)
    plt.close(fig)


# =============================================================================================
def get_sample(nusc, scene_name, index):
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    assert index < scene['nbr_samples'], f"{scene_name} chi co {scene['nbr_samples']} keyframe"
    token = scene['first_sample_token']
    for _ in range(index):
        token = nusc.get('sample', token)['next']
    return nusc.get('sample', token)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scene', default='scene-0916')
    ap.add_argument('--keyframe', type=int, default=20)
    ap.add_argument('--all', action='store_true', help='chay ca 81 keyframe cua mini_val (khong ve hinh)')
    ap.add_argument('--results', default=None, help='results_nusc.json cua PointPillars; bo trong = mo phong')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--yolo', default='yolo26s.pt', help='YOLO thuong (80 lop COCO)')
    ap.add_argument('--yoloe', default=None,
                    help='dung YOLOE open-vocabulary thay cho --yolo, vd. yoloe-26s-seg.pt (du 10 lop nuScenes)')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--sweeps', type=int, default=5, help='so lan quet lidar gop lai (keyframe + truoc do)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO, YOLOE
    if args.yoloe:
        model = YOLOE(args.yoloe)
        setup_open_vocab(model)
    else:
        model = YOLO(args.yolo)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    if args.all:
        val = create_splits_scenes()['mini_val']
        samples = [s for s in nusc.sample if nusc.get('scene', s['scene_token'])['name'] in val]
    else:
        samples = [get_sample(nusc, args.scene, args.keyframe)]
    preds = load_predictions(nusc, samples, args.results, args.score_thr, args.seed)
    source = args.results or 'MO PHONG (nhan goc + loi gai san)'
    print(f'Du doan 3D: {source} | {len(samples)} keyframe | {DETECTOR["kind"]}: {args.yoloe or args.yolo}, '
          f'imgsz {args.imgsz}')

    t0 = time.time()
    rows = verify_samples(nusc, samples, preds, model, args, draw=not args.all)
    if not args.all:
        print_objects(rows)
    print_summary(rows)
    save_csv(rows, osp.join(OUT_DIR, 'ket_luan_mini_val.csv' if args.all else
                            f'ket_luan_{args.scene}_kf{args.keyframe:02d}.csv'))
    print(f'({time.time() - t0:.0f}s)')


def verify_samples(nusc, samples, preds, model, args, draw=False, out_dir=None):
    """Kiem chung moi box 3D cua cac keyframe. preds: {sample token: [Box he toan cuc]};
    args can co imgsz, conf, sweeps. draw=True: luu hinh tong quan + anh cat cua tung keyframe."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    all_rows, t0 = [], time.time()
    for n, s in enumerate(samples):
        scene_name = nusc.get('scene', s['scene_token'])['name']
        kf = _scene_tokens(nusc, s['scene_token']).index(s['token'])
        rows, cams, _ = verify_sample(nusc, s, preds[s['token']], model, args)
        for r in rows:
            r.update(scene=scene_name, keyframe=kf, sample_token=s['token'])
        all_rows += rows
        if draw:
            title = f'{scene_name} - keyframe {kf}'
            stem = osp.join(out_dir, f'{scene_name}_kf{kf:02d}')
            draw_overview(rows, cams, title, stem + '_overview.png')
            draw_cards(rows, cams, title, stem + '_cards.png')
            print(f'Da luu {stem}_overview.png va {stem}_cards.png')
        elif len(samples) > 1 and n % 10 == 0:
            print(f'  {n + 1}/{len(samples)} keyframe ({time.time() - t0:.0f}s)', flush=True)
    return all_rows


CSV_COLS = ['scene', 'keyframe', 'id', 'lop_3d', 'diem_3d', 'khoang_cach', 'camera_chon', 'cac_camera',
            'nhin_thay', 'che', 'do_ro', 'diem_thong_tin', 'lop_yolo', 'nhan_yolo', 'conf_yolo', 'iou', 'kieu_khop',
            'camera_khop', 'diem_lidar', 'ket_luan', 'nhan_xet', 'tham_chieu_nhan_goc']


def save_csv(rows, path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'Da luu {path} ({len(rows)} vat)')


@functools.lru_cache(maxsize=None)
def _scene_tokens(nusc, scene_token):
    t, out = nusc.get('scene', scene_token)['first_sample_token'], []
    while t:
        out.append(t)
        t = nusc.get('sample', t)['next']
    return tuple(out)


def print_objects(rows):
    print(f"\n{'#':>3s} {'lop 3D':13s}{'diem':>5s}{'m':>5s}  {'camera chon':16s}{'che':>5s}{'ro':>5s}"
          f"{'thong tin':>10s}  {'YOLO':12s}{'IoU':>5s}{'lidar':>6s}  ket luan")
    print('-' * 118)
    for r in rows:
        yolo = f"{r['lop_yolo']} {r['conf_yolo']:.2f}" if r['lop_yolo'] else '-'
        iou = f"{r['iou']:.2f}" if r['iou'] is not None else '-'
        fmt = (lambda x: f'{x:5.2f}' if x is not None else '    -')
        print(f"{r['id']:3d} {r['lop_3d']:13s}{r['diem_3d']:5.2f}{r['khoang_cach']:5.0f}  {r['camera_chon']:16s}"
              f"{fmt(r['che'])}{fmt(r['do_ro'])}{fmt(r['diem_thong_tin']):>10s}  {yolo:12s}{iou:>5s}"
              f"{r['diem_lidar']:6d}  {r['ket_luan']}")
    print('\nNhan xet:')
    for r in rows:
        print(f"  #{r['id']:<3d} [{r['ket_luan']}] {r['nhan_xet']}  <tham chieu: {r['tham_chieu_nhan_goc']}>")


def print_summary(rows):
    n = len(rows)
    print(f'\n===== TONG KET: {n} box 3D =====')
    cnt = collections.Counter(r['ket_luan'] for r in rows)
    for k in VERDICTS:
        print(f'  {k:24s}{cnt[k]:6d}{cnt[k] / max(n, 1):8.0%}')
    print('\n===== DOI CHIEU KET LUAN VOI NHAN GOC (nhan goc KHONG duoc dung de ket luan) =====')
    refs = ['dung (co vat that, cung lop)', 'sai lop', 'khong co vat that (bao nham)']
    print(f"{'ket luan':24s}" + ''.join(f'{x[:22]:>24s}' for x in refs))
    for k in VERDICTS:
        sub = [r for r in rows if r['ket_luan'] == k]
        if sub:
            print(f'{k:24s}' + ''.join(
                f"{sum(r['tham_chieu_nhan_goc'].startswith(x[:7]) for r in sub):24d}" for x in refs))


if __name__ == '__main__':
    main()
