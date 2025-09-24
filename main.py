#fix the random seed for CUDA-based operations on GPUs
def set_deterministic(seed=42):
    import os, random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

set_deterministic(42)


from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from nav_vlm.adapters.qwen_narrator import QwenNarrator
import os, pathlib

img_path = "D:/Python/Python Project/nav_vlm/img/demo05.jpg"

#Generate a save path for the depth map corresponding to the input image.
out_dir = r"D:\Python\Python Project\nav_vlm\outputs"
os.makedirs(out_dir, exist_ok=True)
out_depth = os.path.join(out_dir, f"{pathlib.Path(img_path).stem}_depth.png")

#If Hydra has already been initialized, clear the existing instance to avoid conflicts.
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
#Specify Hydra's configuration directory.
initialize_config_dir(
    config_dir="D:/Python/repo/sam2/sam2/configs",
    version_base=None
)

import yaml, json
from core.pipeline import run
from dataclasses import asdict

#read the config file
with open("D:/Python/Python Project/nav_vlm/conf/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

#Call the main workflow.
summary = run(
    image_path=img_path,
    ckpt_depth=cfg["depth"]["ckpt"],
    ckpt_sam2=cfg["seg"]["ckpt"],
    sam2_cfg=cfg["seg"]["cfg"],
    lang="ENG",
    save_depth_png=out_depth,
    color_depth=True
)
print(f"\nDepth saved to: {out_depth}")
print("JSON:", json.dumps(asdict(summary), ensure_ascii=False, indent=2))

text = QwenNarrator().speak(img_path, summary, lang="ENG")
print("\nVLM:", text)