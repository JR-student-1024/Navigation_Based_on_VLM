import numpy as np, torch
from depth_anything_v2.dpt import DepthAnythingV2 # type: ignore[attr-defined]

class DepthAnythingEstimator:
    def __init__(self, ckpt: str, encoder="vitl", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfgs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = DepthAnythingV2(**cfgs[encoder])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(device).eval()
        self.device = device

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        raw = self.model.infer_image(rgb).astype(np.float32)
        rmin, rmax = np.percentile(raw, 5), np.percentile(raw, 95)
        norm = (raw - rmin) / max(1e-6, (rmax - rmin))
        return 1.0 + 9.0 * (1.0 - norm)
