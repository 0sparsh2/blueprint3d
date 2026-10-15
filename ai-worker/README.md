# Local AI worker

Start the worker from the repository root:

```sh
python3 ai-worker/server.py
```

It listens on `http://127.0.0.1:8001`. The Blueprint3D example is already configured to post Image → 3D asset jobs to this address. By default it uses the authenticated Kaggle CLI, uploads the image to the private `sparshnagpal/furniture-image` dataset, starts the `sparshnagpal/hunyuan3d-gpu-smoke-test` NVIDIA kernel, and downloads the generated GLB. It does not require local model weights.

To use the local MLX backend instead, set `AI_BACKEND=mlx`. That backend requires the MLX environment and Hunyuan3D-2mini files under `hunyuan3d-mlx/python/shape/weights/`; download them from that directory with:

```sh
./.venv/bin/python scripts/dl_modelscope.py
```

The first download is several gigabytes and is resumable. Check readiness with:

```sh
curl http://127.0.0.1:8001/health
```

The worker currently generates a shape GLB. Texture painting is intentionally a separate follow-up stage in the Kaggle kernel. Generated files are copied into `example/models/ai/<image-name>.glb` after a successful job.
