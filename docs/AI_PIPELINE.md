# AI pipeline roadmap

The **AI Studio** panel in `example/index.html` is the browser entry point for three workflows:

- **Image → 3D asset**: remove the background, run a shape model such as Hunyuan3D on a local GPU or Kaggle GPU, create/preserve materials, and save a named GLB under `example/models/ai/`.
- **Image → floorplan**: detect walls, doors, windows, and room boundaries, estimate a scale from a known dimension, then convert the result into Blueprint3D corners and walls. The generated plan must stay editable so a person can correct ambiguous openings or dimensions.
- **Image → room design**: classify flooring, paint, lighting, and furniture, retrieve the closest items from the local asset catalog, and place them into the scene. A match should carry a confidence score and a source image so the user can replace it.

## Worker contract

Set `window.BLUEPRINT3D_AI_ENDPOINT` before loading `example/js/example.js` to connect the panel to a worker. The browser sends a multipart `POST` with fields `image` and `workflow`:

```json
{ "id": "job-123" }
```

It polls `GET /api/jobs/job-123`. A response can include:

```json
{
  "status": "running",
  "stage": "Generate texture",
  "stageIndex": 3,
  "progress": 64,
  "message": "Painting the material map"
}
```

The terminal response is `status: "complete"` or `status: "error"`. A complete asset job should also return an output URL or serialized asset metadata, for example `{ "output": { "name": "blue-sofa", "url": "/models/ai/blue-sofa.glb" } }`.

The current browser fallback intentionally simulates stages when no endpoint is configured. It demonstrates the UX without claiming that a model ran.

## Local and Kaggle execution

For local execution, a small Python service can call `rembg` (or another matting model), Hunyuan3D-MLX on Apple Silicon, or a CUDA Hunyuan3D install, then write the GLB into the app's asset directory. The existing `kaggle/hunyuan3d-gpu/run.py` is a batch worker and can be adapted to the same job contract when a GPU notebook is available.

Keep generated files named after the asset (`nesting-tables.glb`, `blue-sofa.glb`) and keep the thumbnail and metadata beside them. Do not commit Kaggle credentials, notebook outputs, model caches, or temporary job files.

## Accuracy boundaries

An image does not contain reliable real-world scale or hidden geometry. Floorplan reconstruction should ask for at least one known measurement or expose a scale control. A single room photograph cannot prove the exact layout behind the camera. Design matching should therefore present estimated matches and let the user approve them before inserting assets.

Computer-use or web browsing can help find candidate products and reference images, but it should remain an optional retrieval step. The deterministic parts of the pipeline should use the local catalog, image embeddings, and explicit scene metadata so the result remains reproducible and safe to review.
