# core/pipeline.py
import cv2, os
import numpy as np
from nav_vlm.adapters.depth_anything import DepthAnythingEstimator
from nav_vlm.adapters.sam2_seg import SAM2BoxesSegmenter
from nav_vlm.core.summarize import summarize

def _save_depth(depth: np.ndarray, out_path: str, color=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = (d - d.min()) / max(1e-6, (d.max() - d.min()))
    d8 = (d_norm * 255).astype(np.uint8)
    if color:
        img = cv2.applyColorMap(d8, cv2.COLORMAP_PLASMA)
    else:
        img = d8
    cv2.imwrite(out_path, img)

def run(
    image_path,
    ckpt_depth,
    ckpt_sam2,
    sam2_cfg,
    lang="ENG",
    save_depth_png: str | None = None,
    color_depth: bool = True
):
    rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    depth = DepthAnythingEstimator(ckpt_depth, encoder="vitb").infer(rgb)
    if save_depth_png:
        _save_depth(depth, save_depth_png, color=color_depth)
    masks = SAM2BoxesSegmenter(ckpt_sam2, model_cfg=sam2_cfg).masks(rgb)
    return summarize(depth, masks)