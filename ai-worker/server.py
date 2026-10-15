#!/usr/bin/env python3
"""Small local job server for Blueprint3D's AI Studio.

It accepts an image, runs the MLX Hunyuan3D shape pipeline in a background
thread, and exposes the status contract consumed by example/js/example.js.
The server binds to localhost only; it is intended for the local development
app, not as a public upload service.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHAPE_ROOT = ROOT / "hunyuan3d-mlx" / "python" / "shape"
PYTHON = SHAPE_ROOT / ".venv" / "bin" / "python"
WEIGHTS = SHAPE_ROOT / "weights" / "Hunyuan3D-2mini" / "hunyuan3d-dit-v2-mini"
JOBS_ROOT = ROOT / "ai-worker" / "jobs"
ASSET_ROOT = ROOT / "example" / "models" / "ai"
KAGGLE_KERNEL = os.environ.get("KAGGLE_KERNEL", "sparshnagpal/hunyuan3d-gpu-smoke-test")
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "sparshnagpal/furniture-image")
BACKEND = os.environ.get("AI_BACKEND", "kaggle").lower()

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def safe_stem(filename: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(filename).stem).strip("-").lower()
    return stem[:64] or "generated-asset"


def update(job_id: str, **values) -> None:
    with LOCK:
        JOBS[job_id].update(values)


def run_asset_job(job_id: str, image_path: Path, asset_stem: str) -> None:
    output = JOBS_ROOT / job_id / f"{asset_stem}.glb"
    command = [
        str(PYTHON), "-m", "hy3dmlx.pipeline", str(image_path),
        "--out", str(output), "--weights", str(WEIGHTS),
        "--steps", os.environ.get("HY3D_STEPS", "8"),
        "--octree", os.environ.get("HY3D_OCTREE", "192"),
        "--dtype", "float16", "--quantize", "4", "--octree-decode",
    ]
    update(job_id, status="running", stage="Generate shape", stageIndex=2,
           progress=34, message="Loading Hunyuan3D-2mini on Apple Silicon")
    try:
        process = subprocess.Popen(
            command, cwd=str(SHAPE_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if "[dino]" in line:
                update(job_id, stage="Generate shape", stageIndex=2, progress=42, message=line)
            elif "[denoise]" in line:
                update(job_id, stage="Generate shape", stageIndex=2, progress=58, message=line)
            elif "[vae]" in line:
                update(job_id, stage="Generate texture", stageIndex=3, progress=78, message="Decoding mesh geometry")
            elif "[mesh]" in line:
                update(job_id, stage="Save GLB asset", stageIndex=4, progress=92, message=line)
        return_code = process.wait()
        if return_code != 0 or not output.exists():
            raise RuntimeError(f"Hunyuan3D exited with code {return_code}")
        ASSET_ROOT.mkdir(parents=True, exist_ok=True)
        final_path = ASSET_ROOT / f"{asset_stem}.glb"
        final_path.write_bytes(output.read_bytes())
        update(job_id, status="complete", stage="Save GLB asset", stageIndex=5,
               progress=100, message="Generated GLB is ready.",
               output={"name": asset_stem, "url": f"/models/ai/{asset_stem}.glb"})
    except Exception as exc:  # noqa: BLE001 - report worker failure to the UI
        update(job_id, status="error", stage="Failed", progress=100,
               message=str(exc)[:500])


def run_kaggle_job(job_id: str, image_path: Path, asset_stem: str) -> None:
    """Version the private input dataset, run the GPU kernel, and fetch its GLB."""
    job_dir = JOBS_ROOT / job_id
    dataset_dir = job_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata = ROOT / "kaggle" / "furniture-image" / "dataset-metadata.json"
    (dataset_dir / "dataset-metadata.json").write_bytes(metadata.read_bytes())
    # The checked-in kernel accepts furniture.png as a fallback input and emits
    # nesting-tables.glb; the worker renames the result to the uploaded asset.
    (dataset_dir / "furniture.png").write_bytes(image_path.read_bytes())
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)
    try:
        update(job_id, status="running", stage="Upload image", stageIndex=0,
               progress=12, message="Uploading the image to the private Kaggle dataset")
        subprocess.run(["kaggle", "datasets", "version", "-p", str(dataset_dir),
                        "-m", f"Blueprint3D job {job_id}", "--dir-mode", "zip"],
                       cwd=str(ROOT), check=True, capture_output=True, text=True)
        update(job_id, stage="Generate shape", stageIndex=2, progress=28,
               message="Starting the Kaggle NVIDIA kernel")
        subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle" / "hunyuan3d-gpu")],
                       cwd=str(ROOT), check=True, capture_output=True, text=True)
        deadline = time.time() + int(os.environ.get("KAGGLE_TIMEOUT", "3600"))
        while time.time() < deadline:
            status = subprocess.run(["kaggle", "kernels", "status", KAGGLE_KERNEL],
                                    cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
            upper = status.upper()
            if "COMPLETE" in upper:
                break
            if "ERROR" in upper or "CANCEL" in upper or "FAILED" in upper:
                raise RuntimeError(status or "Kaggle kernel failed")
            update(job_id, stage="Generate shape", stageIndex=2, progress=55,
                   message="Kaggle kernel is running on an NVIDIA GPU")
            time.sleep(15)
        else:
            raise TimeoutError("Kaggle kernel timed out")
        update(job_id, stage="Generate texture", stageIndex=3, progress=82,
               message="Downloading the generated GLB")
        subprocess.run(["kaggle", "kernels", "output", KAGGLE_KERNEL,
                        "-p", str(output_dir), "--force"], cwd=str(ROOT), check=True,
                       capture_output=True, text=True)
        candidates = list(output_dir.glob("*.glb"))
        if not candidates:
            raise FileNotFoundError("Kaggle completed without a GLB output")
        final_path = ASSET_ROOT / f"{asset_stem}.glb"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(candidates[0].read_bytes())
        update(job_id, status="complete", stage="Save GLB asset", stageIndex=5,
               progress=100, message="Kaggle generated GLB is ready.",
               output={"name": asset_stem, "url": f"/models/ai/{asset_stem}.glb"})
    except Exception as exc:  # noqa: BLE001
        update(job_id, status="error", stage="Failed", progress=100,
               message=str(exc)[:500])


class Handler(BaseHTTPRequestHandler):
    server_version = "Blueprint3D-AI/0.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8000")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"ok": True, "backend": BACKEND,
                                 "modelReady": BACKEND == "kaggle" or (WEIGHTS.joinpath("config.yaml").exists() and WEIGHTS.joinpath("model.fp16.safetensors").exists()),
                                 "kaggleKernel": KAGGLE_KERNEL})
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9-]+)", self.path)
        if match:
            with LOCK:
                job = JOBS.get(match.group(1))
            if job:
                self.send_json(200, job)
            else:
                self.send_json(404, {"status": "error", "message": "Unknown job"})
            return
        self.send_json(404, {"status": "error", "message": "Not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/jobs":
            self.send_json(404, {"status": "error", "message": "Not found"})
            return
        if not PYTHON.exists():
            if BACKEND == "mlx":
                self.send_json(503, {"status": "error", "message": f"MLX environment missing: {PYTHON}"})
                return
            return
        if BACKEND == "mlx" and (not WEIGHTS.joinpath("config.yaml").exists() or not WEIGHTS.joinpath("model.fp16.safetensors").exists()):
            self.send_json(503, {"status": "error", "message": "Hunyuan3D-2mini weights are not downloaded yet."})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(
            (f"Content-Type: {self.headers.get('Content-Type', '')}\r\n"
             "MIME-Version: 1.0\r\n\r\n").encode() + body
        )
        fields = {}
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            name_match = re.search(r'name="([^"]+)"', disposition)
            if not name_match:
                continue
            name = name_match.group(1)
            fields[name] = {"filename": part.get_filename(), "data": part.get_payload(decode=True) or b"", "value": part.get_content()}
        upload = fields.get("image")
        workflow = fields.get("workflow", {}).get("value", "asset")
        if not upload or not upload.get("filename"):
            self.send_json(400, {"status": "error", "message": "Missing image"})
            return
        if workflow != "asset":
            self.send_json(501, {"status": "error", "message": "Only Image → 3D asset is connected to the MLX worker yet."})
            return
        job_id = str(uuid.uuid4())
        asset_stem = safe_stem(upload["filename"])
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        image_path = job_dir / f"input{Path(upload['filename']).suffix.lower() or '.png'}"
        with image_path.open("wb") as target:
            target.write(upload["data"])
        with LOCK:
            JOBS[job_id] = {"id": job_id, "workflow": workflow, "status": "queued", "stage": "Upload image", "stageIndex": 0, "progress": 8, "message": "Image received."}
        target = run_kaggle_job if BACKEND == "kaggle" else run_asset_job
        threading.Thread(target=target, args=(job_id, image_path, asset_stem), daemon=True).start()
        self.send_json(202, {"id": job_id})

    def log_message(self, format, *args):  # noqa: A002
        print(f"[ai-worker] {self.address_string()} - {format % args}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("AI_WORKER_PORT", "8001"))
    print(f"Blueprint3D AI worker on http://127.0.0.1:{port}", flush=True)
    print(f"Model weights: {WEIGHTS}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
