import os
import cv2
import numpy as np
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


def _read_grayscale(path: str, size: int = 64) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Falha ao ler imagem: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized


class PixelFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, size: int = 64):
        self.size = size

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        feats = []
        for p in X:
            img = _read_grayscale(p, self.size)
            # Normaliza para [0,1] e achata
            vec = (img.astype(np.float32) / 255.0).reshape(-1)
            feats.append(vec)
        return np.vstack(feats)


class MorphFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, size: int = 64):
        self.size = size

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        feats = []
        for p in X:
            img = _read_grayscale(p, self.size)
            # Binarização Otsu (fundo branco, texto preto invertido para foreground=1)
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            h, w = th.shape
            total_pixels = h * w
            fg = np.count_nonzero(th)
            area_ratio = fg / float(total_pixels) if total_pixels > 0 else 0.0

            # Componentes conectados com estatísticas
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
            # Exclui background (label 0)
            num_components = max(0, num_labels - 1)

            largest_area_ratio = 0.0
            smallest_area_ratio = 0.0
            centroid_y_diff_norm = 0.0

            if num_components > 0:
                # Áreas dos componentes (label >= 1)
                areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
                largest_area = areas.max() if areas.size > 0 else 0.0
                smallest_area = areas.min() if areas.size > 0 else 0.0
                largest_area_ratio = (largest_area / fg) if fg > 0 else 0.0
                smallest_area_ratio = (smallest_area / fg) if fg > 0 else 0.0

                # Diferença vertical entre centroides, se houver pelo menos 2 componentes
                if num_components >= 2:
                    # Ordena por coordenada y (cima -> baixo)
                    c = centroids[1:]  # remove background
                    ys = c[:, 1]
                    top_idx = np.argmin(ys)
                    bottom_idx = np.argmax(ys)
                    dy = abs(ys[bottom_idx] - ys[top_idx])
                    centroid_y_diff_norm = dy / float(h)

            # Aspect ratio do bounding box do foreground total
            ys, xs = np.where(th > 0)
            if ys.size > 0 and xs.size > 0:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                bbox_h = max(1, y_max - y_min + 1)
                bbox_w = max(1, x_max - x_min + 1)
                bbox_aspect_ratio = bbox_h / float(bbox_w)
            else:
                bbox_aspect_ratio = 0.0

            feats.append([
                float(num_components),
                float(area_ratio),
                float(largest_area_ratio),
                float(bbox_aspect_ratio),
                float(smallest_area_ratio),
                float(centroid_y_diff_norm),
            ])
        return np.array(feats, dtype=np.float32)