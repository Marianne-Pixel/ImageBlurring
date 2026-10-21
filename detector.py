"""
detector.py
Personen-Detektion mit YOLO11-Segmentierung (ultralytics) für:
  - normale (kaum verzerrte) Fotos
  - Equirectangular-Panoramen (z.B. Ricoh Theta Z1)

Es wird ein YOLO11-SEGMENTIERUNGS-Modell verwendet (yolo11s-seg.pt),
Dadurch steht pro erkannter Person die Silhouette (Polygon) zur Verfügung 
Das ermöglicht es, beim Verwischen nur den
Personen-Umriss (unregelmässige Form) zu treffen statt grossflächig auch
den Hintergrund mitzuverwischen.

Equirectangular-Bilder, wie ursprünglich vorgegeben:
  1. Panorama in mehrere überlappende, perspektivische Tiles zerlegen
  2. Segmentierung pro Tile durchführen
  3. Erkannte Silhouetten zurück ins Panorama-Koordinatensystem transformieren
  4. Ergebnis als Maske zurückgeben (Blur/Pixelierung, siehe processing.py)
"""

import math
import numpy as np
import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO-Klasse "person"


class PersonDetector:
    def __init__(self, model_path="yolo11s-seg.pt", conf_threshold=0.35, device=None):
        """
        model_path: YOLO11-Segmentierungsgewichte (werden beim ersten Aufruf
                    automatisch heruntergeladen, sofern eine
                    Internetverbindung besteht). Varianten (schneller ↔
                    genauer): 'yolo11n-seg.pt' < 'yolo11s-seg.pt' <
                    'yolo11m-seg.pt' < 'yolo11l-seg.pt' < 'yolo11x-seg.pt'.
                    Für präzisere Erkennung z.B. 'yolo11m-seg.pt' verwenden.
        conf_threshold: Mindest-Konfidenz für eine gültige Detektion.
        device: 'cpu', 'cuda:0', ... (None = automatische Wahl durch ultralytics)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    # ---------------------------------------------------------------
    # Normale Bilder
    # ---------------------------------------------------------------
    def detect_masks_normal(self, img_bgr):
        """
        Liefert eine Liste von Polygonen (je ein Nx2-int32-Array) für die
        Silhouette jeder erkannten Person, in Bildkoordinaten. Dies ist
        deutlich präziser als eine Bounding-Box, da nur die tatsächliche
        Personenform berücksichtigt wird.
        """
        results = self.model.predict(
            img_bgr,
            conf=self.conf_threshold,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        polygons = []
        for r in results:
            if r.masks is None:
                continue
            for poly in r.masks.xy:  # bereits in Original-Bildkoordinaten
                if poly is None or len(poly) < 3:
                    continue
                polygons.append(np.round(poly).astype(np.int32))
        return polygons

    # ---------------------------------------------------------------
    # Equirectangular-Bilder (Theta Z1)
    # ---------------------------------------------------------------
    def detect_mask_equirect(self, img_bgr, tile_size=900, fov_deg=100, margin_px=None):
        """
        Zerlegt das Panorama in überlappende perspektivische Tiles, führt die
        Segmentierung pro Tile durch und projiziert die erkannten
        Personen-Silhouetten zurück ins Equirectangular-Koordinatensystem.

        Rückgabe: 1-Kanal-Maske (uint8, Werte 0/255) in der Grösse des
        Panoramas, die die tatsächlichen Personen-Silhouetten markiert
        (keine rechteckigen Bereiche).
        """
        eq_h, eq_w = img_bgr.shape[:2]
        mask = np.zeros((eq_h, eq_w), dtype=np.uint8)

        if margin_px is None:
            margin_px = max(2, round(0.006 * tile_size))

        for yaw, pitch in self._tile_directions():
            map_x, map_y = _equirect_to_persp_maps(
                eq_w, eq_h, tile_size, tile_size, fov_deg, yaw, pitch
            )
            tile = cv2.remap(
                img_bgr, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )

            polygons = self.detect_masks_normal(tile)
            if not polygons:
                continue

            tile_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            for poly in polygons:
                cv2.fillPoly(tile_mask, [poly.reshape(-1, 1, 2)], 255)

            if margin_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
                )
                tile_mask = cv2.dilate(tile_mask, kernel, iterations=1)

            # Rück-Transformation: für jedes maskierte Tile-Pixel kennen wir
            # über map_x/map_y bereits die zugehörige Panorama-Koordinate.
            ys, xs = np.where(tile_mask > 0)
            if len(xs) == 0:
                continue
            eq_xs = map_x[ys, xs]
            eq_ys = map_y[ys, xs]
            eq_xs = np.mod(np.round(eq_xs).astype(np.int32), eq_w)
            eq_ys = np.clip(np.round(eq_ys).astype(np.int32), 0, eq_h - 1)
            mask[eq_ys, eq_xs] = 255

        # Kleine Lücken aus der Rück-Projektion schliessen (Rand bewusst klein halten)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    @staticmethod
    def _tile_directions():
        """
        Blickrichtungen (yaw, pitch) in Grad, die die gesamte Kugel mit
        Überlappung abdecken (Äquatorband, oberes/unteres Band, Zenit/Nadir).
        """
        dirs = []
        for yaw in range(0, 360, 60):
            dirs.append((yaw, 0))
        for yaw in range(0, 360, 90):
            dirs.append((yaw, 45))
            dirs.append((yaw, -45))
        dirs.append((0, 90))
        dirs.append((0, -90))
        return dirs


def _equirect_to_persp_maps(eq_w, eq_h, out_w, out_h, fov_deg, yaw_deg, pitch_deg):
    """
    Erstellt map_x/map_y (float32) für cv2.remap: für jedes Pixel des
    perspektivischen Tiles wird die entsprechende Koordinate im
    Equirectangular-Bild berechnet (klassische Pinhole-Projektion mit
    Yaw-/Pitch-Rotation).
    """
    fov = math.radians(fov_deg)
    f = (out_w / 2) / math.tan(fov / 2)

    x = np.arange(out_w) - out_w / 2 + 0.5
    y = np.arange(out_h) - out_h / 2 + 0.5
    xx, yy = np.meshgrid(x, -y)
    zz = np.full_like(xx, f, dtype=np.float64)

    norm = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    xx, yy, zz = xx / norm, yy / norm, zz / norm

    # Pitch-Rotation (um die X-Achse)
    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    y1 = yy * cp - zz * sp
    z1 = yy * sp + zz * cp
    x1 = xx

    # Yaw-Rotation (um die Y-Achse)
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x2 = x1 * cy + z1 * sy
    z2 = -x1 * sy + z1 * cy
    y2 = y1

    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(y2, -1.0, 1.0))

    map_x = (lon / (2 * np.pi) + 0.5) * eq_w
    map_y = (0.5 - lat / np.pi) * eq_h

    return map_x.astype(np.float32), map_y.astype(np.float32)
