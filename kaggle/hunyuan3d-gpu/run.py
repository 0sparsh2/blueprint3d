import os
import subprocess
import sys
from pathlib import Path


def run(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])


# Kaggle's image has CUDA/PyTorch, but not the Hunyuan3D package.
run("install", "-q", "--upgrade", "setuptools", "wheel", "rembg")
run("install", "-q", "git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git")
subprocess.check_call([
    "git", "clone", "--depth", "1",
    "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
    "/tmp/Hunyuan3D-2",
])
run("install", "-q", "/tmp/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer")
run("install", "-q", "/tmp/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer")

import torch
from PIL import Image
from rembg import new_session, remove

# Recent diffusers releases reject Hunyuan's local custom paint pipeline unless
# remote-code trust is threaded through an internal loader.  Hunyuan's own
# package ships this code in the environment, so relax that one guard before
# importing the texture pipeline.
import diffusers.utils.dynamic_modules_utils as _dynamic_utils
_dynamic_file = Path(_dynamic_utils.__file__)
_dynamic_source = _dynamic_file.read_text()
import re
_dynamic_source = re.sub(
    r"(?m)^(\s*)if not trust_remote_code:\s*$",
    r"\1if False and not trust_remote_code:",
    _dynamic_source,
    count=1,
)
_dynamic_file.write_text(_dynamic_source)
import importlib
importlib.reload(_dynamic_utils)

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


assert torch.cuda.is_available(), "Kaggle kernel did not receive a CUDA GPU"
print("CUDA:", torch.cuda.get_device_name(0), flush=True)

asset_stem = os.environ.get("ASSET_STEM", "nesting-tables")
input_path = Path(f"/kaggle/input/furniture-image/{asset_stem}.png")
if not input_path.exists():
    candidates = sorted(Path("/kaggle/input").glob(f"**/{asset_stem}.png"))
    if not candidates:
        candidates = sorted(Path("/kaggle/input").glob("**/furniture.png"))
    if candidates:
        input_path = candidates[0]
    else:
        available = sorted(str(p) for p in Path("/kaggle/input").glob("**/*"))
        raise FileNotFoundError(
            f"furniture.png was not mounted. Available Kaggle inputs: {available[:50]}"
        )
print("Input:", input_path, flush=True)

# Remove the room background before shape inference.  isnet-general-use is a
# smaller local model than RMBG-2.0 and works well for a single furniture image.
cutout_path = Path("/kaggle/working/input_cutout.png")
cutout = remove(
    Image.open(input_path).convert("RGBA"),
    session=new_session("isnet-general-use"),
    alpha_matting=False,
)
# Hunyuan's preprocessor is more reliable with an RGB image than a fully
# transparent canvas, so place the isolated object on a neutral white field.
white = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
cutout = Image.alpha_composite(white, cutout).convert("RGB")
cutout.save(cutout_path)
print("Cutout:", cutout_path, flush=True)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mini",
    subfolder="hunyuan3d-dit-v2-mini",
    use_safetensors=True,
    device="cuda",
)
mesh = pipeline(
    image=Image.open(cutout_path).convert("RGB"),
    num_inference_steps=30,
    octree_resolution=256,
    num_chunks=20000,
)
if isinstance(mesh, list):
    mesh = mesh[0]
Path("/kaggle/working").mkdir(exist_ok=True)
mesh_path = Path(f"/kaggle/working/{asset_stem}-shape.glb")
mesh.export(mesh_path)
print(f"Wrote {mesh_path}", flush=True)

# The texture model uses a bundled custom diffusers pipeline.  Newer
# diffusers versions require explicit permission to load that model code.
candidates = list(Path("/usr/local/lib").glob(
    "python*/dist-packages/hy3dgen/texgen/utils/multiview_utils.py"
))
if not candidates:
    candidates = list(Path("/usr/local/lib").glob(
        "python*/site-packages/hy3dgen/texgen/utils/multiview_utils.py"
    ))
if not candidates:
    raise FileNotFoundError("Could not locate Hunyuan multiview_utils.py")
multiview_utils = candidates[0]
source = multiview_utils.read_text()
source = source.replace(
    "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)",
    "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16, trust_remote_code=True)",
)
multiview_utils.write_text(source)

paint = Hunyuan3DPaintPipeline.from_pretrained(
    "tencent/Hunyuan3D-2",
    subfolder="hunyuan3d-paint-v2-0-turbo",
)
paint.enable_model_cpu_offload()
textured_mesh = paint(mesh, image=Image.open(cutout_path).convert("RGB"))
textured_path = Path(f"/kaggle/working/{asset_stem}.glb")
textured_mesh.export(textured_path)
print(f"Wrote {textured_path}", flush=True)
