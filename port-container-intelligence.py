"""
PORT BULK CARRIER INTELLIGENCE  v6.0
Dev: Anas Maqsood
- NEW: Colored segmentation overlay — Ship (MAGENTA), Cranes (LIME GREEN)
- NEW: Water masking — sea/ocean pixels NOT colored
- NEW: Smooth filled polygon segmentation (like SAM-style coloring)
- Faster: detect_every=4, scale=0.45
- Better HUD with real-time progress bar
"""

import subprocess, sys, os, time, warnings, glob, threading, math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore")

_REQUIRED = ["opencv-python", "numpy", "matplotlib", "tqdm"]

def _ensure():
    for pkg in _REQUIRED:
        mod = pkg.replace("-", "_").split("[")[0]
        try:
            __import__(mod)
        except ImportError:
            print(f"  [AUTO-INSTALL] {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

print("=" * 70)
print("  PORT BULK CARRIER INTELLIGENCE  |  Anas Maqsood  |  v6.0")
print("  Segmentation: Ship=MAGENTA  Cranes=LIME  Water=NO COLOR")
print("=" * 70)
_ensure()

import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

CFG = {
    "output_video":       "port_output_v6.mp4",
    "output_dashboard":   "port_dashboard_v6.png",
    "output_dir":         "port_output_v6",
    "detect_every":       4,
    "detect_scale":       0.45,
    "max_detect_dim":     640,
    "min_crane_area":     700,
    "min_ship_area":      6000,
    "iou_match_thresh":   0.18,
    "max_age_frames":     12,
    "trail_length":       35,
    "hud_width":          310,
    "heatmap_decay":      0.983,
    # Segmentation overlay alphas
    "seg_alpha_ship":     0.52,
    "seg_alpha_crane":    0.48,
}

# Colors in BGR
C = {
    "crane":   (50,  255,  50),      # LIME GREEN
    "ship":    (200,  50, 200),      # MAGENTA / PURPLE-PINK
    "accent":  (0,   210, 255),      # CYAN
    "bg":      (8,    10,  16),
    "panel":   (12,   15,  24),
    "grey":    (80,   85, 100),
    "white":   (235, 238, 245),
    "dark":    (20,   24,  36),
    "green":   (30,  220,  80),
    "amber":   (0,   165, 255),
}

# Segmentation fill colors (BGR)
SEG_SHIP_COLOR  = (180, 40, 190)    # magenta
SEG_CRANE_COLOR = (40,  240, 40)    # lime green


@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self) -> int:   return (self.x1 + self.x2) // 2
    @property
    def cy(self) -> int:   return (self.y1 + self.y2) // 2
    @property
    def w(self) -> int:    return self.x2 - self.x1
    @property
    def h(self) -> int:    return self.y2 - self.y1
    @property
    def area(self) -> float: return float(self.w * self.h)
    @property
    def center(self) -> Tuple[int, int]: return (self.cx, self.cy)

    def iou(self, o: "BBox") -> float:
        ix1 = max(self.x1, o.x1); iy1 = max(self.y1, o.y1)
        ix2 = min(self.x2, o.x2); iy2 = min(self.y2, o.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / (self.area + o.area - inter + 1e-6)


@dataclass
class Det:
    cls: str; bbox: BBox; conf: float; mask: Optional[np.ndarray] = None


@dataclass
class Track:
    tid: int; cls: str; bbox: BBox; conf: float
    mask: Optional[np.ndarray] = None
    trail: deque = field(default_factory=lambda: deque(maxlen=CFG["trail_length"]))
    missed: int = 0; age: int = 0
    velocity: Tuple[float, float] = (0., 0.)
    _pcx: float = 0.; _pcy: float = 0.

    def update(self, bbox: BBox, conf: float, mask=None):
        dx = bbox.cx - self._pcx; dy = bbox.cy - self._pcy
        a = 0.35
        self.velocity = (a * dx + (1 - a) * self.velocity[0],
                         a * dy + (1 - a) * self.velocity[1])
        self._pcx = float(bbox.cx); self._pcy = float(bbox.cy)
        self.bbox = bbox; self.conf = conf; self.mask = mask
        self.trail.append((bbox.cx, bbox.cy))
        self.missed = 0; self.age += 1


# ──────────────────────────────────────────────────────────────────────
#  WATER MASK  – exclude sea/ocean from coloring
# ──────────────────────────────────────────────────────────────────────
def build_water_mask(frame_small: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask (uint8, 255=water) at the small frame resolution.
    Detects teal/blue-green water using HSV.
    """
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    # Teal-blue-green water range
    m1 = cv2.inRange(hsv, (85,  40, 60), (105, 220, 220))   # teal
    m2 = cv2.inRange(hsv, (78,  30, 50), (115, 255, 210))   # broader blue-green
    water = cv2.bitwise_or(m1, m2)
    # Morphological cleaning
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k, iterations=3)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN,  k, iterations=2)
    return water


# ──────────────────────────────────────────────────────────────────────
#  DETECTOR
# ──────────────────────────────────────────────────────────────────────
class BulkPortDetector:
    def __init__(self, full_W: int, full_H: int):
        scale = min(CFG["detect_scale"], CFG["max_detect_dim"] / max(full_W, full_H))
        self.scale = scale
        self.dW = int(full_W * scale); self.dH = int(full_H * scale)
        self.inv = 1. / scale
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=36, detectShadows=False)
        print(f"  [DETECT] {full_W}x{full_H} -> {self.dW}x{self.dH}  scale={scale:.2f}")

    def _scale_area(self, a: int) -> int:
        return max(1, int(a * self.scale * self.scale))

    @staticmethod
    def _blobs(mask, min_area, max_n=12):
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = sorted(range(1, nb),
                       key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        out = []
        for i in blobs[:max_n]:
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out.append((stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                             labels == i))
        return out

    def _clean(self, mask, min_area, dil=2, ero=1):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if ero:  mask = cv2.erode(mask,  k, iterations=ero)
        if dil:  mask = cv2.dilate(mask, k, iterations=dil)
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def detect(self, frame: np.ndarray) -> List[Det]:
        small = cv2.resize(frame, (self.dW, self.dH), interpolation=cv2.INTER_LINEAR)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        inv   = self.inv
        dets  = []

        # ── WATER MASK (do NOT color water) ──────────────────────────
        water_mask = build_water_mask(small)
        not_water  = cv2.bitwise_not(water_mask)

        # ── CRANE: white/light-grey + yellow machinery + edge density ─
        m_white  = cv2.inRange(hsv, (0,  0, 175), (180, 40, 255))
        m_lgrey  = cv2.inRange(hsv, (0,  0, 145), (180, 35, 215))
        m_yellow = cv2.inRange(hsv, (18, 90, 110), (35, 255, 255))
        crane_color = cv2.bitwise_or(cv2.bitwise_or(m_white, m_lgrey), m_yellow)
        crane_color = cv2.bitwise_and(crane_color, not_water)
        edges    = cv2.Canny(gray, 45, 130)
        edge_dil = cv2.dilate(edges, np.ones((9, 9), np.uint8))
        crane_mask = cv2.bitwise_and(crane_color, edge_dil)
        s_crane  = self._scale_area(CFG["min_crane_area"])
        crane_mask = self._clean(crane_mask, s_crane, dil=3, ero=1)

        for x1, y1, w_, h_, blob_lbl in self._blobs(crane_mask, s_crane, max_n=10):
            ar = max(w_, h_) / max(min(w_, h_), 1)
            if ar < 1.2 and w_ * h_ < s_crane * 3: continue
            fx1 = int(x1 * inv); fy1 = int(y1 * inv)
            fx2 = int((x1 + w_) * inv); fy2 = int((y1 + h_) * inv)
            # Upscale mask to full frame resolution
            seg = cv2.resize(blob_lbl.astype(np.uint8) * 255,
                             (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            dets.append(Det(cls="Crane", bbox=BBox(fx1, fy1, fx2, fy2),
                            conf=0.85, mask=seg))

        # ── SHIP BODY: large, bright-colored or grey vessel hull ──────
        # Ships appear as large regions that are NOT water and NOT crane.
        # Use color + large area approach.
        m_hull_grey  = cv2.inRange(hsv, (0,   0,  80), (180,  50, 200))
        m_hull_color = cv2.inRange(hsv, (0,  30,  60), (180, 255, 240))
        fg = self._bg.apply(small, learningRate=0.005)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_wide = cv2.dilate(fg, np.ones((20, 20), np.uint8))

        # Ship = hull color AND not water, either fg or solid presence
        ship_cand = cv2.bitwise_or(m_hull_grey, m_hull_color)
        ship_cand = cv2.bitwise_and(ship_cand, not_water)
        # Exclude very bright (sky/white crane parts already handled)
        m_vbright = cv2.inRange(hsv, (0, 0, 240), (180, 20, 255))
        ship_cand = cv2.bitwise_and(ship_cand, cv2.bitwise_not(m_vbright))

        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        ship_cand = cv2.morphologyEx(ship_cand, cv2.MORPH_CLOSE, close_k)
        s_ship = self._scale_area(CFG["min_ship_area"])
        ship_cand = self._clean(ship_cand, s_ship, dil=5, ero=2)

        for x1, y1, w_, h_, blob_lbl in self._blobs(ship_cand, s_ship, max_n=4):
            ar = max(w_, h_) / max(min(w_, h_), 1)
            if ar > 8: continue
            if w_ * h_ < s_ship: continue
            fx1 = int(x1 * inv); fy1 = int(y1 * inv)
            fx2 = int((x1 + w_) * inv); fy2 = int((y1 + h_) * inv)
            seg = cv2.resize(blob_lbl.astype(np.uint8) * 255,
                             (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            dets.append(Det(cls="Ship", bbox=BBox(fx1, fy1, fx2, fy2),
                            conf=0.82, mask=seg))

        return self._nms(dets)

    @staticmethod
    def _nms(dets, thresh=0.45):
        keep = []; groups = defaultdict(list)
        for d in dets: groups[d.cls].append(d)
        for cls, grp in groups.items():
            grp.sort(key=lambda d: d.conf, reverse=True)
            sup = set()
            for i, d in enumerate(grp):
                if i in sup: continue
                keep.append(d)
                for j in range(i + 1, len(grp)):
                    if grp[i].bbox.iou(grp[j].bbox) > thresh: sup.add(j)
        return keep
