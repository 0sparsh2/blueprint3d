# Blueprint3D: local setup and AI integration

Research date: October 14, 2026. This checkout is upstream commit cac8b62 (July 10, 2016).

Update: free local Apple Silicon routes are documented in [FREE-LOCAL-MODELS.md](FREE-LOCAL-MODELS.md). Community MLX ports change the earlier deployment recommendation; a NVIDIA worker is not the only available route.


## Local setup

Node/npm and Python 3 are needed. Run from the repository root:

```sh
npm install
npm run build
npm start
```

Open http://127.0.0.1:8000. Stop the server with Ctrl-C. `.npmrc` enables legacy peer resolution because grunt-typescript 0.8 expects Grunt 0.4 while the project specifies Grunt 1. The upstream npm build command pointed to a nonexistent JavaScript entry; it now uses the working TypeScript Grunt build. The lockfile records dependencies.

Verified: Grunt compilation and dependency copying; browser rendering of the default textured room; floor-plan editor and furniture catalog. These checks do not verify any AI inference. No model service, image upload feature, or provider credentials are configured. The legacy dependency audit reports 43 vulnerabilities; modernizing dependencies is required before turning this prototype into a deployed product.

## What the existing code supports

- `src/model/floorplan.ts`: `loadFloorplan()` accepts a corner dictionary and wall edges. `saveFloorplan()` exports the same graph. Rooms are derived from wall cycles, rather than imported room labels.
- `src/model/model.ts`: `loadSerialized()` consumes `{floorplan, items}`. It is declared private in TypeScript but the existing JavaScript example calls it. Make this API public during integration rather than copying that mismatch.
- `src/model/scene.ts`: `addItem()` uses THREE.JSONLoader, THREE.Geometry and MeshFaceMaterial with Three.js r69. It expects the old Three.js geometry JSON, not GLB/glTF or an arbitrary JSON object.
- `src/items/item.ts`: furniture is a single Mesh, centered in its bounding box, with collision/placement behavior based on dimensions. A modern glTF may contain multiple meshes, transforms and PBR materials; those need a deliberate Item adapter.
- Coordinate units are centimeters. Floorplan `(x,y)` maps to scene `(x,z)`, with scene `y` vertical. The sidebar displays item dimensions in inches.
- Item types: 1 floor furniture, 2 wall item, 3 in-wall item, 7 in-wall floor item, 8 on-floor item, 9 wall/floor item. Doors/windows are specialized placed items, not native opening records in the floorplan JSON.

## Furniture photograph -> generated asset -> interior

Recommended first experiment: Meshy hosted image-to-3D API, with inference behind a local backend. This avoids a CUDA installation on this ARM Mac. Alternative for owning inference: TRELLIS.2 on a separate Linux NVIDIA GPU worker. Hunyuan3D 2.1 is another candidate worth evaluating on the same furniture reference set.

| Candidate | Documented capability | Deployment consideration |
| --- | --- | --- |
| [Meshy image-to-3D](https://docs.meshy.ai/en/api/image-to-3d) | Async image task with downloadable textured GLB | Paid provider API, backend secret, polling, local artifact persistence |
| [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-conditioned generation and GLB export with PBR materials | Official setup tested on Linux; NVIDIA >=24 GB; model/code MIT with separately licensed dependencies |
| [Tencent Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Separate shape and PBR texture stages | Documents 10 GB shape, 21 GB texture, 29 GB combined VRAM; custom community license; evaluate deployment terms |

The recommended architecture is an engineering proposal, not a measured quality comparison. Model performance and latency need testing with actual furniture. TRELLIS.2 reports H100 timings, which should not be treated as end-to-end latency on this machine.

Pipeline:

1. Upload one isolated furniture image; crop/remove background. Multiple views improve coverage if the selected provider supports them. A room photo first requires per-object segmentation/crops.
2. Backend starts generation and returns a job ID. UI shows queued/generating/processing/ready/failed states. Keep provider keys on the server.
3. Download the completed asset into our own storage before temporary provider links expire. Retain the GLB as the canonical artifact, along with reference, model/version, generation parameters and dimensions.
4. Normalize orientation to Y-up, apply transforms, simplify geometry, reduce texture sizes and set a floor origin. Collect a known width/depth/height: a single image does not establish metric scale or unseen geometry.
5. Register an editable furniture Item and let the user position, rotate and resize it. Save an asset ID plus transforms; resolve that ID to a durable asset URL on reload.

**Renderer decision:** modernize the engine to current Three.js + GLTFLoader before a durable GLB integration. Port Geometry operations, material handling, controls, raycasting, wall cutouts, bounding boxes and Item selection/collision behavior. Upgrading only the npm version will break the current code.

For an isolated feasibility demo, an offline converter can flatten a static GLB into legacy geometry JSON and bake base-color textures. This sacrifices PBR, hierarchy and possibly material fidelity; it is a temporary bridge, not the preferred long-term format. A modern preview canvas beside the old editor can assess generation quality before engine migration, but it does not make the asset placeable in Blueprint3D.

## Floor-plan image -> editable home map

Suggested baseline: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for labels, OCR, dimension interpretation and an initial graph proposal. Its documented spatial grounding/OCR capabilities do not establish reliable architectural reconstruction. Evaluate on representative plans before relying on its coordinates.

For more reliable boundaries, combine a wall/door/window detector with the VLM. [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) supplies a floorplan-specific multi-task segmentation/vectorization baseline and dataset, but its original Python/PyTorch stack is old; it is a research starting point rather than a ready modern service.

Pipeline:

1. Accept a floor-plan image, deskew it, and preserve the image as an editable tracing background. Large plans may need overlapping tiles.
2. Detect wall centerlines/junctions, rooms, doors/windows and text. Let the VLM interpret room names and dimension labels and return a constrained JSON structure. Keep observations, guesses and confidence separate.
3. Recover scale from explicit dimensions, or ask the user to mark a known-length segment. Without calibration, return pixel-space geometry and require scale before import; never invent measurements.
4. Deterministically reconcile geometry: merge nearby junctions, split segments at intersections, remove duplicate/zero-length edges, retain intentional diagonals, and verify referenced corners and room cycles. Surface uncertain edges for review. Model confidence alone is not validation.
5. Convert pixels to centimeters, preserving aspect ratio and origin. Feed `{floorplan, items: []}` into the editor. Keep labels, uncertainty, source image and openings in a separate project document; the upstream serializer does not retain them.
6. Review the 2D overlay, correct topology/scale, then use Blueprint3D's wall extrusion for the 3D home. Doors/windows need the existing in-wall item system and compatible assets.

Suggested intermediate contract (before scale calibration and graph repair):

```json
{
  "schemaVersion": 1,
  "coordinateSpace": "image_pixels",
  "imageSize": {"width": 1600, "height": 1000},
  "scale": null,
  "corners": [{"id": "c1", "x": 100, "y": 100}],
  "walls": [],
  "rooms": [],
  "openings": [],
  "dimensionObservations": [],
  "uncertainties": ["No verified scale"]
}
```

Require strict schema validation, finite coordinates, valid IDs, bounded image coordinates and explicit scale provenance. Add source pixel spans and confidence to observations. Define room polygons and openings against stable wall IDs, then preserve or remap them when graph repair splits walls.

`example/ai-samples/two-room.blueprint3d` is a hand-authored, centimeter-scale fixture compatible with **Load Plan**. It demonstrates the target import format, not model extraction. It intentionally contains no doors/windows/furniture.

## Proposed implementation order and evidence

1. Establish floorplan schema + validator + adapter and overlay review UI. Use hand-labeled plans to verify imports, then add a VLM backend. Acceptance: topology, room count, scale error and human correction effort.
2. Add backend generation jobs and a GLB preview using one furniture provider. Acceptance: actual completion/failure behavior, durable downloads, dimensions, visual similarity and triangle/texture budgets.
3. Modernize rendering and add glTF Item support; verify drag, rotate, resize, collision, selection, persistence and door cutouts on existing and generated assets.
4. Evaluate VLM-only extraction against segmentation + VLM. Track missing/extra walls, junction error, openings, room closure and dimension error. Test generated chairs/sofas/tables for silhouette, backside plausibility, dimensions and browser frame time.

No exact-home reconstruction can be validated from one interior photograph; that is a separate multi-view capture/reconstruction problem. Both pipelines should produce editable drafts. For floor plans, geometric walls should be constructed from validated vectors; diffusion/flow-based generation belongs in the furniture pipeline.
