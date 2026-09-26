# Architecture Diagram — AutoLabel 2D

## Luồng tổng thể (6 bước)

```mermaid
graph LR
    subgraph IN[1. Input nuScenes]
        K[CAM_FRONT keyframe]
        SW[Camera sweeps t-2..t+2]
        L[LIDAR_TOP]
        C[Calibration + ego pose]
    end
    subgraph DET[2. 2D Detection open-vocab]
        YW[YOLO-World]
        GD[Grounding DINO]
        F2[Florence-2]
        FU[Fusion + cache]
    end
    subgraph QA[3. QA Agent - LangGraph]
        Q1[3.1 Confidence]
        Q2[3.2 LiDAR support]
        Q3[3.3 Temporal]
        Q4[3.4 Issue generation]
        Q5[3.5 Risk scoring]
    end
    K --> YW & GD & F2
    SW --> YW
    YW & GD & F2 --> FU
    FU --> Q1 & Q2 & Q3
    L & C --> Q2
    Q1 & Q2 & Q3 --> Q4 --> Q5
    Q5 --> UI[4. Review by exception UI]
    UI --> LOG[5. Correction log]
    UI --> DS[6. Auto-labeled dataset]
```

## QA Agent (src/agents/graph.py)

```mermaid
graph LR
    START((START)) --> A[confidence_check]
    START --> B[lidar_check]
    START --> T[temporal_check]
    A --> I[issue_generation]
    B --> I
    T --> I
    I --> R[risk_scoring]
    R --> END((END))
```

Ba check chạy song song, `issue_generation` chờ đủ cả ba (`add_edge([...], ...)`).
Agent deterministic, không gọi LLM.

## Thành phần

| Thành phần | File | Vai trò |
|---|---|---|
| Loader nuScenes | `src/services/nuscenes_data.py` | Keyframe, sweep 12Hz, chiếu LiDAR→ảnh có bù ego-motion, GT 2D chiếu từ 3D |
| Detector | `src/services/detectors/` | YOLO-World / Grounding DINO / Florence-2, fuse kiểu WBF, cache theo `sample_data` token |
| QA Agent | `src/agents/` | 3.1–3.5, sinh issue code + risk |
| Pipeline | `src/services/pipeline.py`, `src/cli.py` | Chạy batch, ghi frame JSON vào workspace |
| Review | `src/services/review.py` | Keep / Delete / Change class / Edit box / Add box / Batch approve, M4, flag precision/recall |
| Store | `src/services/store.py` | JSON file + `corrections.jsonl` |
| Export | `src/services/exporter.py` | COCO + JSONL, chỉ frame đã approve |
| Đánh giá | `src/services/evaluation.py` | mAP@0.5/0.7, flag recall/precision so với GT |
| API | `src/api/routes.py` | REST cho UI |
| UI | `src/web/` | HTML/JS thuần, FastAPI phục vụ ở `/ui/` |
