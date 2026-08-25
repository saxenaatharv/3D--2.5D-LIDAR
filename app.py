"""
app.py
-------
FastAPI service for the Adaptive 2.5D Lidar Mapping frontend.

Endpoints:
    POST   /api/frames               upload a .bin frame (multipart/form-data)
    GET    /api/frames               list uploaded frames
    DELETE /api/frames/{frame_id}    drop a frame from memory
    GET    /api/backend/status       which classification backend is active
    POST   /api/runs                 start a run over a set of frame ids + config
    GET    /api/runs/{run_id}/status poll progress (status, current/total, log tail)
    GET    /api/runs/{run_id}/results   per-frame metrics + cell data for rendering
    GET    /api/runs/{run_id}/log.csv   the run as a CSV, same columns as run_log.csv

Frame and run data live in memory only (dicts) -- fine for a hackathon
demo / single-user session, not for production. Swap FRAMES/RUNS for a
real store (Redis, a DB, disk-backed cache) before deploying multi-user.
"""

import io
import time
import uuid

import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from grid_engine import VariableResolutionGrid
from metrics import memory_saving_report
from backends import get_backend

app = FastAPI(title="Adaptive 2.5D Lidar Mapping API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRAMES: dict[str, np.ndarray] = {}
RUNS: dict[str, dict] = {}

DEFAULT_RINGS = [
    {"name": "near", "r_min": 0.0, "r_max": 10.0, "cell_size": 0.05},
    {"name": "mid", "r_min": 10.0, "r_max": 30.0, "cell_size": 0.20},
    {"name": "far", "r_min": 30.0, "r_max": 60.0, "cell_size": 0.35},
    {"name": "edge", "r_min": 60.0, "r_max": 100.0, "cell_size": 0.50},
]


def _parse_frame_bytes(filename: str, raw: bytes) -> np.ndarray:
    """
    Mirrors run_inference.load_points(): accepts KITTI-style .bin (float32,
    N x 4: x,y,z,intensity) or .npy (N x 3 or N x 4). Always returns N x 4
    float32 so every frame in FRAMES has a consistent shape (missing
    intensity is zero-filled, same as run_inference.py / dataset_utils.py).
    """
    name = (filename or "").lower()

    if name.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not parse .npy file: {exc}")
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError(
                f".npy point cloud must be N x 3 or N x 4, got shape {arr.shape}"
            )
        if arr.shape[1] == 3:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
        return arr.astype(np.float32)

    # default: KITTI-style .bin (float32, N x 4)
    if len(raw) % 16 != 0:  # 4 float32 values x 4 bytes
        raise ValueError(
            "File size is not a multiple of 16 bytes -- expected KITTI-style "
            "float32 N x 4 (x,y,z,intensity), or upload a .npy file instead"
        )
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).astype(np.float32)


# ---------------------------------------------------------------- frames --
@app.post("/api/frames")
async def upload_frame(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        arr = _parse_frame_bytes(file.filename, raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    frame_id = str(uuid.uuid4())[:8]
    FRAMES[frame_id] = arr
    return {"frame_id": frame_id, "name": file.filename, "n_points": int(arr.shape[0])}


@app.get("/api/frames")
def list_frames():
    return [{"frame_id": k, "n_points": int(v.shape[0])} for k, v in FRAMES.items()]


@app.delete("/api/frames/{frame_id}")
def delete_frame(frame_id: str):
    FRAMES.pop(frame_id, None)
    return {"ok": True}


# --------------------------------------------------------------- backend --
@app.get("/api/backend/status")
def backend_status(model: str = "RandLANet", device: str = "cpu", checkpoint_path: str | None = None):
    backend = get_backend(model, device, checkpoint_path)
    return backend.status()


# ------------------------------------------------------------------ runs --
class RunConfig(BaseModel):
    frame_ids: list[str]
    rings: list[dict] = DEFAULT_RINGS
    z_clip: list[float] = [-3.0, 3.0]
    ego_origin: list[float] = [0.0, 0.0]
    model: str = "RandLANet"
    device: str = "cpu"
    checkpoint_path: str | None = None


@app.post("/api/runs")
def create_run(cfg: RunConfig, background_tasks: BackgroundTasks):
    missing = [fid for fid in cfg.frame_ids if fid not in FRAMES]
    if missing:
        raise HTTPException(400, f"Unknown frame ids: {missing}")
    if not cfg.frame_ids:
        raise HTTPException(400, "frame_ids is empty")

    run_id = str(uuid.uuid4())[:8]
    RUNS[run_id] = {
        "status": "queued", "total": len(cfg.frame_ids), "current": 0,
        "log": [], "results": [],
    }
    background_tasks.add_task(_execute_run, run_id, cfg)
    return {"run_id": run_id}


def _execute_run(run_id: str, cfg: RunConfig):
    run = RUNS[run_id]
    run["status"] = "running"
    backend = get_backend(cfg.model, cfg.device, cfg.checkpoint_path)
    run["log"].append(f"backend: {backend.name} ({'pretrained' if getattr(backend,'name','')=='open3d-ml' else 'placeholder'})")

    for i, fid in enumerate(cfg.frame_ids):
        t0 = time.time()
        pts = FRAMES[fid]
        xyz = pts[:, :3]

        labels, source = backend.predict(xyz)

        grid = VariableResolutionGrid(rings_cfg=cfg.rings, ego_origin=tuple(cfg.ego_origin), z_clip=tuple(cfg.z_clip))
        grid.insert_points(xyz, labels)
        mem = memory_saving_report(grid)
        fps = 1.0 / max(time.time() - t0, 1e-6)

        cells = [
            {"ring": ring_id, "ix": ix, "iy": iy, "cls": cell.dominant_class, "z_mean": round(cell.z_mean, 3), "count": cell.count}
            for (ring_id, ix, iy), cell in grid.cells.items()
        ]

        run["results"].append({
            "frame_id": fid,
            "fps": round(fps, 2),
            "active_cells": mem["sparse_cells_used"],
            "equivalent_uniform_cells": mem["equivalent_uniform_cells"],
            "memory_saving_percent": mem["memory_saving_percent"],
            "mean_iou": "n/a",
            "source": source,
            "n_points": int(xyz.shape[0]),
            "cells": cells,
        })
        run["current"] = i + 1
        run["log"].append(
            f"frame {i+1}/{len(cfg.frame_ids)}  fps={fps:.1f}  "
            f"cells={mem['sparse_cells_used']}  saving={mem['memory_saving_percent']}%  source={source}"
        )

    run["status"] = "done"
    run["log"].append("done")


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    return {"status": run["status"], "current": run["current"], "total": run["total"], "log": run["log"][-80:]}


@app.get("/api/runs/{run_id}/results")
def run_results(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    return run["results"]


@app.get("/api/runs/{run_id}/log.csv")
def run_log_csv(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    lines = ["frame,fps,active_cells,memory_saving_percent,mean_iou"]
    for i, r in enumerate(run["results"]):
        lines.append(f"{i},{r['fps']},{r['active_cells']},{r['memory_saving_percent']},{r['mean_iou']}")
    return PlainTextResponse("\n".join(lines), media_type="text/csv")


@app.get("/api/health")
def health():
    return {"ok": True}
