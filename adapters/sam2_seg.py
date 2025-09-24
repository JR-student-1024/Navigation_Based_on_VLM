# adapters/sam2_seg.py
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np, torch, os

class SAM2BoxesSegmenter:
    def __init__(self, ckpt: str, model_cfg: str, device: str | None = None):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.isabs(model_cfg) and os.path.isfile(model_cfg):
            # 绝对路径写法
            cfg_name = os.path.basename(model_cfg)       # sam2_hiera_s.yaml
            cfg_dir = os.path.dirname(model_cfg)         # D:/.../configs/sam2.1
            model = build_sam2(
                config_file=cfg_name,
                config_path=cfg_dir,
                ckpt=ckpt,
                device=device,
            )
        else:
            # 相对路径写法（Hydra 会从 pkg://sam2 开始找）
            model = build_sam2(config_file=model_cfg, ckpt=ckpt, device=device)

        self.pred = SAM2ImagePredictor(model)

    def masks(self, rgb: np.ndarray):
        H, W, _ = rgb.shape
        y1, y2 = int(H * 0.55), int(H * 0.95)
        xs = [0, W // 3, 2 * W // 3, W]
        self.pred.set_image(rgb)
        out = {}
        for name, i in zip(["left", "center", "right"], [0, 1, 2]):
            box = np.array([xs[i] + 10, y1, xs[i + 1] - 10, y2])
            m = self.pred.predict(box=box)[0]
            m = m[0] if m.ndim == 3 else m
            out[name] = (m > 0.5)
        return out
