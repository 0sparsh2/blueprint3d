# Kaggle Hunyuan3D GPU smoke test

This kernel runs the CUDA/PyTorch Hunyuan3D-2mini shape pipeline on Kaggle. It does not use the local Mac storage for model weights. Kaggle downloads the model into the ephemeral kernel environment and writes the textured result using the input item name (for example, `nesting-tables.glb`).

Before pushing:

1. Create a private Kaggle Dataset named `furniture-image` containing `furniture.png`.
2. Add that dataset to `dataset_sources` in `kernel-metadata.json`, using the exact `username/dataset-slug`.
3. The configured account is `sparshnagpal`; change it if you use another Kaggle account.
4. Configure credentials locally with the current OAuth flow, then run:

```sh
kaggle auth login
kaggle kernels push -p kaggle/hunyuan3d-gpu --timeout 3600 --accelerator NvidiaTeslaT4
kaggle kernels logs sparshnagpal/hunyuan3d-gpu-smoke-test --follow
kaggle kernels output sparshnagpal/hunyuan3d-gpu-smoke-test -p kaggle/output
```

See Kaggle's current [authentication reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md). For non-interactive runs, set `KAGGLE_API_TOKEN` or save an access token at `~/.kaggle/access_token`.

The model card documents the same `Hunyuan3DDiTFlowMatchingPipeline` family and the mini checkpoint. A Kaggle T4 may be insufficient for textured generation; this smoke test intentionally generates shape-only GLB first.
