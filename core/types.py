from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Literal, Protocol

Direction = Literal["left", "center", "right", "front", "rear", "unknown"]

@dataclass
class Hazard:
    type: str
    distance_m: float
    direction: Direction

@dataclass
class SceneSummary:
    hazards: List[Hazard]
    zones_m: Dict[str, float]
    recommendation: str  # "left" / "right" / "straight" / "wait"

class DepthEstimator(Protocol):
    def infer(self, rgb: np.ndarray) -> np.ndarray: ...

class Segmenter(Protocol):
    def masks(self, rgb: np.ndarray) -> Dict[str, np.ndarray]: ...

class Narrator(Protocol):
    def speak(self, img_path: str, summary: SceneSummary, lang: str) -> str: ...
