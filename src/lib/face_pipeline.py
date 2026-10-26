"""
Shared face pipeline: detection, alignment and preprocessing.

Used by the backend (FaceService) and by the notebooks, so that training and
inference see exactly the same crops and the same pixel normalization.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils import face_align

# ArcFace convention: RGB, (x - 127.5) / 127.5, CHW, batch dimension.
INPUT_MEAN = 127.5
INPUT_STD = 127.5


def load_detector(
    detector_path: Path, det_score_threshold: float = 0.5, det_input_size: int = 640
) -> SCRFD:
    """Load the SCRFD detector (e.g. det_10g.onnx from buffalo_l) on CPU."""
    detector = SCRFD(model_file=str(detector_path))
    detector.prepare(
        ctx_id=-1,
        det_thresh=det_score_threshold,
        input_size=(det_input_size, det_input_size),
    )
    return detector


def detect(
    detector: SCRFD, image: np.ndarray, min_face_size: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the detector on a BGR uint8 image.
    Returns (bboxes, kpss): bboxes is (N, 5) with x1, y1, x2, y2, score and kpss is
    (N, 5, 2) in absolute image coordinates, sorted by score (highest first).
    Faces whose shorter side is below min_face_size are dropped.
    """
    bboxes, kpss = detector.detect(image)
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    keep = np.minimum(w, h) >= min_face_size
    bboxes, kpss = bboxes[keep], kpss[keep]
    order = np.argsort(-bboxes[:, 4])
    return bboxes[order], kpss[order]


def align(image: np.ndarray, kps: np.ndarray, face_size: int = 112) -> np.ndarray:
    """Similarity transform of the 5 landmarks to the ArcFace template (BGR in, BGR out)."""
    return face_align.norm_crop(image, landmark=np.asarray(kps, np.float32), image_size=face_size)


def preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Aligned BGR face -> float32 tensor (1, 3, H, W), RGB, scaled to [-1, 1]."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - INPUT_MEAN) / INPUT_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding
