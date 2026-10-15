# Free local AI options for this Blueprint3D project

Checked October 14, 2026. Target hardware: Apple M4, 16 GB unified memory. Recommendations are based on documentation and published examples; no inference benchmark has been run here.

## Recommended starting stack

- Furniture geometry: Python MLX Hunyuan3D-2mini with 4-bit quantization and octree decoding. Compare against Hunyuan3D-2.0-turbo once the baseline works.
- Textured furniture experiment: TRELLIS.2 community MLX port, 512 preview, one generation at a time.
- Floorplan understanding: Qwen3.5 4B through local Ollama, with Qwen3-VL 4B as a comparison. Geometry should come from image processing/detection and deterministic graph validation.
- Local HTTP service: a Python backend queues generation, persists GLBs, and exposes status/results to Blueprint3D. There is no provider fee; downloaded models run locally.

## Furniture: Python diffusion/flow-based generation

### Hugging Face comparison

For this 16 GB M4, the practical shortlist is:

| Model | HF checkpoint | Output | Local fit | Recommendation |
| --- | --- | --- | --- | --- |
| TripoSR | about 1.68 GB | mesh with vertex colors or baked texture | smallest, MIT, CUDA-oriented official path | Fastest baseline to get an end-to-end asset working |
| Hunyuan3D-2mini | about 3.82 GB for the mini DiT plus VAE | mesh; MLX port exports GLB | best native Apple route; Tencent community license | Our main quality/compatibility candidate |
| Stable Fast 3D | 1B parameters, gated | UV-unwrapped textured mesh with material parameters | MPS is experimental and official README recommends 32 GB unified memory for Macs | Better asset format and likely quality, but risky on this machine |

[TripoSR](https://huggingface.co/stabilityai/TripoSR) is the best smaller model. Its repository is about 1.68 GB and the model is MIT licensed. The official implementation documents about 6 GB VRAM on its default path, but it is PyTorch/CUDA-oriented and does not have the native MLX route we already installed. A CoreML/ONNX/community port may be a useful second experiment, but we should not assume parity with the official output.

[Stable Fast 3D](https://huggingface.co/stabilityai/stable-fast-3d) is the better *asset-shaped* candidate: it emits a UV-unwrapped textured mesh and material parameters, and its model card says it is designed for sub-second reconstruction. It is gated, carries the Stability AI Community License, and the official Mac path says MPS is experimental and recommends CPU under 32 GB unified memory. That makes it a candidate for a later quality comparison rather than the first install on this laptop.

The Hugging Face `Hunyuan3D-2mini` card describes a 0.6B shape generator, but the mini safetensor file is still roughly 3.82 GB and the VAE is additional. Its smaller model is preferable to full Hunyuan 2.0 for our native MLX work; the `mini-fast` and `mini-turbo` directories are also listed on the Hub, but their current safetensor size is not smaller than the mini checkpoint. [Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini)

The MLX shape checkout has now been added locally at `hunyuan3d-mlx/python/shape`. Its
Python 3.12/MLX environment installs successfully and the weight-free layer, sampler and
octree tests pass (12 tests). The first mini VAE download was started from ModelScope and is
resumable; the host was too slow for this run, so no generated mesh has been claimed yet.
Once the two mini checkpoints finish downloading, the documented command is:

```sh
cd hunyuan3d-mlx/python/shape
uv run python -m hy3dmlx.pipeline path/to/isolated-furniture.png \
  --out ../../../generated-chair.glb \
  --weights weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini \
  --steps 30 --octree 256 --dtype float16 --quantize 4
```

The input should be a single centered object on a transparent or plain background. The output
is a GLB mesh; Blueprint3D still needs a modern GLTF loader/Item adapter before it can place
this file interactively.

[Hunyuan Python MLX shape implementation](https://github.com/ZimengXiong/Hunyuan3D-MLX/tree/main/python/shape) supports mini, 2.0, turbo and 2.1 variants. Maintainer measurements report roughly 2.7 GB for quantized mini with octree decoding and about 4 GB for turbo in an optimized configuration. These are shape-only measurements, not guaranteed peaks or timings on our M4.

The Python package exposes `Hunyuan3DShapePipeline.from_pretrained()`, `generate()`, and mesh export. It expects a foreground RGBA image. Use Python 3.12 and uv in an isolated environment under `python/shape`; follow the repository's checkpoint/config download instructions. Evaluate chair legs and thin parts at low resolution before increasing detail.

[The same project's shape/paint documentation](https://github.com/ZimengXiong/Hunyuan3D-MLX) reports much higher memory for complete texture generation. Start with plain materials and add texturing separately. Port code and pretrained weights have different licenses; retain upstream model terms.

[TRELLIS.2 MLX port](https://github.com/lyonsno/trellis2mlx) produces textured GLB locally. It reports a completed shoe on a 16 GB M2 Pro in about 21 minutes, while a complex higher-resolution run exceeded 60 GiB. Start with 512/no-cascade, eight steps and small textures; fitting one case does not establish a universal memory bound. DINOv3 weights require gated access. This is a community port rather than Microsoft's official Mac support.

[TripoSR](https://github.com/VAST-AI-Research/TripoSR) is an MIT-licensed feedforward reconstruction baseline rather than diffusion. Its roughly 6 GB default GPU footprint is attractive, but compiled dependencies and the CUDA-oriented path require additional Mac verification. Prefer the documented native MLX route for our first experiment.

## Floor-plan image to an editable home

[Qwen3.5 4B on Ollama](https://ollama.com/library/qwen3.5:4b) is a vision model with an approximately 4 GB download. Download size is not runtime memory. A moderate context and sequential execution make it a sensible 16 GB candidate; compare extraction accuracy on our plans rather than assuming it is best from general benchmarks.

After installing a current Ollama version:

```sh
ollama pull qwen3.5:4b
ollama serve
```

If Ollama's app already runs its server, do not start a second server. Python can submit local images and request a JSON schema using Ollama's [vision](https://docs.ollama.com/capabilities/vision) and [structured output](https://docs.ollama.com/capabilities/structured-outputs) APIs.

Suggested processing: wall detection -> VLM labels/dimension interpretation -> merge/split junctions -> calibrate centimeters -> validate closed rooms -> Blueprint3D corner/wall JSON. OpenCV is a reasonable deterministic first baseline for clean line drawings. Evaluate a floorplan-specific segmentation model on noisy plans; CubiCasa5K remains a research baseline with an old upstream stack. Do not treat schema-valid VLM coordinates as geometrically correct.

## Free hosted alternatives

[Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) offers existing free demos with queues and daily quotas; current free-account quota is five GPU minutes/day. Some Spaces expose callable Gradio endpoints, but each endpoint and artifact-download policy must be checked. This is useful for a model bake-off, not an unlimited backend.

[Meshy developer documentation](https://docs.meshy.ai/en/api/developer-platform) says free accounts can use its image-to-3D Playground through a testing key, but cannot create their own API keys. Free web credits do not provide a general free integration API.

## Blueprint3D integration remains necessary

Generated GLB output cannot be passed to the old THREE.JSONLoader. For a lasting implementation, modernize rendering and adapt multi-mesh/PBR assets to Item placement, dimensions and persistence. A temporary static geometry conversion can demonstrate insertion with reduced material fidelity.

Proposed experiment: one mini-model asset, one 512 textured TRELLIS asset, and one Qwen floorplan extraction; record peak memory, elapsed time, mesh quality, and correction effort. Unload the VLM before generating 3D assets on this 16 GB machine. No new model packages or weights were installed during this research.
