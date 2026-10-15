# Local AI worker

Start the worker from the repository root:

```sh
python3 ai-worker/server.py
```

It listens on `http://127.0.0.1:8001`. The Blueprint3D example is already configured to post Image → 3D asset jobs to this address. The worker requires the MLX environment and Hunyuan3D-2mini files under `hunyuan3d-mlx/python/shape/weights/`; download them from that directory with:

```sh
./.venv/bin/python scripts/dl_modelscope.py
```

The first download is several gigabytes and is resumable. Check readiness with:

```sh
curl http://127.0.0.1:8001/health
```

The worker currently generates a shape GLB. Texture painting is intentionally a separate follow-up stage because the native MLX shape port does not include the CUDA PBR paint pipeline. Generated files are copied into `example/models/ai/<image-name>.glb` after a successful job.
