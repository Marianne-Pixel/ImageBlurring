"""
detector.py
Personen-Detektion mit YOLO11-Segmentierung (ultralytics) für:
  - normale (kaum verzerrte) Fotos
  - Equirectangular-Panoramen (z.B. Ricoh Theta Z1)
  - Fisheye-Bilder (zirkulares Fisheye, z.B. eine Linse einer 360°-Kamera),
    siehe detect_mask_fisheye()

Es wird ein YOLO11-SEGMENTIERUNGS-Modell verwendet (yolo11s-seg.pt).
Dadurch steht pro erkannter Person die Silhouette (Polygon) zur Verfügung.
Das ermöglicht es, beim Verwischen nur den Personen-Umriss (unregelmässige
Form) zu treffen statt grossflächig auch den Hintergrund mitzuverwischen.

Equirectangular-Bilder, wie ursprünglich vorgegeben:
  1. Panorama in mehrere überlappende, perspektivische Tiles zerlegen
  2. Segmentierung pro Tile durchführen
  3. Erkannte Silhouetten zurück ins Panorama-Koordinatensystem transformieren
  4. Ergebnis als Maske zurückgeben (Blur/Pixelierung, siehe processing.py)

Geschwindigkeit (praktisch die gesamte Rechenzeit entfällt auf YOLO):
  - Rechenwerk wird automatisch gewählt: NVIDIA-GPU (CUDA, halbe
    Genauigkeit FP16) > OpenVINO auf der CPU (ca. 2.5x schneller als
    PyTorch auf der CPU) > PyTorch auf der CPU.
  - Alle Tiles eines Bildes werden gebündelt (Batches) an YOLO übergeben.
  - Fisheye: Tiles werden zuerst nur in 0° und 90° segmentiert, die
    Drehungen 180°/270° nur noch für Tiles, in denen etwas gefunden wurde.
  - Die Abbildungs-Tabellen (Tile <- Bild) werden zwischengespeichert und
    bei gleich grossen Bildern wiederverwendet.
"""

import math
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO-Klasse "person"
DEFAULT_IMGSZ = 640  # YOLO-Eingabegrösse für normale Bilder und Panorama-Tiles

# --- Fisheye-Standardwerte (an das verwendete Objektiv anpassen) ---
# Bildwinkel über den Durchmesser des Bildkreises in Grad. Die Kamera des
# Beispielbildes sieht einen Teil ihres eigenen Gehäuses, der Bildwinkel
# ist also grösser als 180°.
FISHEYE_FOV_DEG = 200
# Abbildungsfunktion: "equidistant" (r = f·θ, bei den meisten 360°-Kameras),
# "equisolid" (r = 2f·sin(θ/2)) oder "stereographic" (r = 2f·tan(θ/2)).
FISHEYE_PROJECTION = "equidistant"
# Drehungen (Grad), in denen jedes Tile segmentiert wird. Nötig, weil die
# Kamera gedreht montiert sein kann (in den Beispielbildern um ±90°,
# Personen liegen dann quer im Bild).
#  - Stufe 1 (alle Tiles): 0° und 90° decken stehende, kopfstehende und
#    liegende Personen bereits ab.
#  - Stufe 2 (nur Tiles mit einem Treffer in Stufe 1, auch unsicheren ab
#    FISHEYE_REFINE_TRIGGER_CONF): 180° und 270° verfeinern die Silhouetten.
# Ist die Kamera immer gleich montiert, z.B. FISHEYE_ROTATIONS = (270,)
# und FISHEYE_REFINE_ROTATIONS = () setzen (nochmals deutlich schneller).
FISHEYE_ROTATIONS = (0, 90)
FISHEYE_REFINE_ROTATIONS = (180, 270)
FISHEYE_REFINE_TRIGGER_CONF = 0.2

# Anzahl Tiles pro YOLO-Aufruf (GPU / CPU). Grössere Batches brauchen mehr
# Arbeitsspeicher.
BATCH_SIZE_GPU = 8
BATCH_SIZE_CPU = 4

# Zwischenspeicher für das OpenVINO-Modell (wird beim ersten Start einmalig
# aus der .pt-Datei erzeugt, falls nicht bereits mitgeliefert).
OPENVINO_CACHE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"),
    "ImageBlurring_YOLO11",
)


def _fp16_predict_kwargs():
    """
    Argument für FP16-Inferenz auf der GPU. Neuere ultralytics-Versionen
    verwenden `quantize=16` (das alte `half=True` ist dort veraltet und gibt
    eine Warnung aus), ältere Versionen kennen nur `half=True`.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _openvino_available():
    try:
        import openvino  # noqa: F401
        return True
    except Exception:
        return False


def _load_openvino_model(model_path):
    """
    Lädt die OpenVINO-Variante des Modells. Gesucht wird zuerst ein
    mitgelieferter Ordner '<name>_openvino_model' neben der .pt-Datei, dann
    im Zwischenspeicher. Fehlt beides, wird das Modell einmalig exportiert
    (dauert einige Sekunden). Wirft eine Exception, wenn das nicht klappt.
    """
    base = os.path.splitext(os.path.basename(model_path))[0]
    folder = base + "_openvino_model"
    for candidate in (os.path.join(os.path.dirname(os.path.abspath(model_path)), folder),
                      os.path.join(OPENVINO_CACHE_DIR, folder)):
        if os.path.isfile(os.path.join(candidate, base + ".xml")):
            return YOLO(candidate, task="segment")

    os.makedirs(OPENVINO_CACHE_DIR, exist_ok=True)
    cached_pt = os.path.join(OPENVINO_CACHE_DIR, os.path.basename(model_path))
    if not os.path.isfile(cached_pt):
        shutil.copy2(model_path, cached_pt)
    exported = YOLO(cached_pt).export(format="openvino", dynamic=True, imgsz=1024, verbose=False)
    return YOLO(exported, task="segment")


class PersonDetector:
    def __init__(self, model_path="yolo11s-seg.pt", conf_threshold=0.35, device=None, backend="auto"):
        """
        model_path: YOLO11-Segmentierungsgewichte. Varianten (schneller ↔
                    genauer): 'yolo11n-seg.pt' < 'yolo11s-seg.pt' <
                    'yolo11m-seg.pt' < 'yolo11l-seg.pt' < 'yolo11x-seg.pt'.
        conf_threshold: Mindest-Konfidenz für eine gültige Detektion.
        device: 'cpu', 'cuda:0', ... (None = automatische Wahl)
        backend: "auto" (GPU > OpenVINO > PyTorch-CPU), "openvino" oder "torch".
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.half = False
        self.backend_name = "PyTorch (CPU)"
        self._map_cache = {}

        use_gpu = _cuda_available() and (device is None or str(device).startswith("cuda"))
        model = None
        if backend in ("auto", "openvino") and not use_gpu and _openvino_available():
            try:
                model = _load_openvino_model(model_path)
                self.device = None
                self.backend_name = "OpenVINO (CPU)"
            except Exception:
                model = None  # Fallback auf PyTorch
        if model is None:
            model = YOLO(model_path)
            if use_gpu:
                self.half = True
                self._fp16_kwargs = _fp16_predict_kwargs()
                self.backend_name = "PyTorch (GPU, FP16)"
        self.model = model
        self.batch_size = BATCH_SIZE_GPU if use_gpu else BATCH_SIZE_CPU

    # ---------------------------------------------------------------
    # YOLO-Aufruf (gebündelt)
    # ---------------------------------------------------------------
    def _segment_batch(self, images, imgsz=None, min_conf=None):
        """
        Segmentiert mehrere Bilder, in Batches zu self.batch_size.
        Rückgabe: pro Bild ein Tupel (polygone, max_konfidenz).
          polygone: Silhouetten mit Konfidenz >= conf_threshold
          max_konfidenz: höchste Konfidenz aller Kandidaten >= min_conf
                         (0.0, wenn keine), z.B. als Auslöser für Stufe 2
        imgsz=None = Standard-Eingabegrösse DEFAULT_IMGSZ (640).
        """
        min_conf = self.conf_threshold if min_conf is None else min(min_conf, self.conf_threshold)
        # Eingabegrösse immer explizit setzen: ein OpenVINO-Modell hätte sonst
        # die Export-Grösse (1024) als Standard statt 640 wie PyTorch.
        kwargs = {"imgsz": int(imgsz) if imgsz is not None else DEFAULT_IMGSZ}
        if self.half:
            kwargs.update(self._fp16_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device

        out = []
        images = list(images)
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            results = self.model.predict(
                chunk,
                conf=min_conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                batch=len(chunk),
                **kwargs,
            )
            for r in results:
                polygons, max_conf = [], 0.0
                if r.boxes is not None and len(r.boxes) > 0:
                    confs = r.boxes.conf.cpu().numpy()
                    max_conf = float(confs.max())
                    if r.masks is not None:
                        for poly, c in zip(r.masks.xy, confs):  # Original-Bildkoordinaten
                            if c < self.conf_threshold or poly is None or len(poly) < 3:
                                continue
                            polygons.append(np.round(poly).astype(np.int32))
                out.append((polygons, max_conf))
        return out

    # ---------------------------------------------------------------
    # Normale Bilder
    # ---------------------------------------------------------------
    def detect_masks_normal(self, img_bgr, imgsz=None):
        """
        Liefert eine Liste von Polygonen (je ein Nx2-int32-Array) für die
        Silhouette jeder erkannten Person, in Bildkoordinaten. Dies ist
        deutlich präziser als eine Bounding-Box, da nur die tatsächliche
        Personenform berücksichtigt wird.
        """
        return self._segment_batch([img_bgr], imgsz=imgsz)[0][0]

    # ---------------------------------------------------------------
    # Equirectangular-Bilder (Theta Z1)
    # ---------------------------------------------------------------
    def detect_mask_equirect(self, img_bgr, tile_size=900, fov_deg=100, margin_px=None):
        """
        Zerlegt das Panorama in überlappende perspektivische Tiles, führt die
        Segmentierung für alle Tiles gebündelt durch und projiziert die
        erkannten Personen-Silhouetten zurück ins Equirectangular-
        Koordinatensystem.

        Rückgabe: 1-Kanal-Maske (uint8, Werte 0/255) in der Grösse des
        Panoramas, die die tatsächlichen Personen-Silhouetten markiert
        (keine rechteckigen Bereiche).
        """
        eq_h, eq_w = img_bgr.shape[:2]
        mask = np.zeros((eq_h, eq_w), dtype=np.uint8)

        if margin_px is None:
            margin_px = max(2, round(0.006 * tile_size))
        kernel = None
        if margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
            )

        key = ("equirect", eq_w, eq_h, tile_size, fov_deg)
        maps = self._map_cache.get(key)
        if maps is None:
            maps = [
                _equirect_to_persp_maps(eq_w, eq_h, tile_size, tile_size, fov_deg, yaw, pitch)
                for yaw, pitch in self._tile_directions()
            ]
            self._remember_maps(key, maps)

        tiles = [
            cv2.remap(img_bgr, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT101)
            for map_x, map_y in maps
        ]
        results = self._segment_batch(tiles)

        for (map_x, map_y), (polygons, _) in zip(maps, results):
            if not polygons:
                continue

            tile_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            for poly in polygons:
                cv2.fillPoly(tile_mask, [poly.reshape(-1, 1, 2)], 255)
            if kernel is not None:
                tile_mask = cv2.dilate(tile_mask, kernel, iterations=1)

            # Rück-Transformation: für jedes maskierte Tile-Pixel kennen wir
            # über map_x/map_y bereits die zugehörige Panorama-Koordinate.
            ys, xs = np.where(tile_mask > 0)
            if len(xs) == 0:
                continue
            eq_xs = np.mod(np.round(map_x[ys, xs]).astype(np.int32), eq_w)
            eq_ys = np.clip(np.round(map_y[ys, xs]).astype(np.int32), 0, eq_h - 1)
            mask[eq_ys, eq_xs] = 255

        # Kleine Lücken aus der Rück-Projektion schliessen (Rand bewusst klein halten)
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel5, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
        return mask

    def _remember_maps(self, key, maps):
        """Hält nur die Tabellen der zuletzt verwendeten Bildgrösse pro Modus
        im Speicher (ca. 75-100 MB)."""
        for k in [k for k in self._map_cache if k[0] == key[0]]:
            del self._map_cache[k]
        self._map_cache[key] = maps

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

    # ---------------------------------------------------------------
    # Fisheye-Bilder
    # ---------------------------------------------------------------
    def detect_mask_fisheye(
        self,
        img_bgr,
        fov_deg=FISHEYE_FOV_DEG,
        projection=FISHEYE_PROJECTION,
        center=None,
        radius=None,
        tile_fov_deg=100,
        tile_size=None,
        rotations=FISHEYE_ROTATIONS,
        refine_rotations=FISHEYE_REFINE_ROTATIONS,
        margin_px=None,
    ):
        """
        Fisheye-Workflow (analog zu den Equirectangular-Bildern):
          a) Bild schneiden: Der Bildkreis wird in überlappende, entzerrte
             perspektivische Tiles zerlegt (ein Tile in Blickrichtung der
             Kamera plus ein Ring von Tiles rundherum bis zum Bildrand).
          b) Segmentieren: YOLO11-Segmentierung aller Tiles gebündelt. Da
             die Kamera gedreht montiert sein kann (Personen liegen dann im
             Bild), wird jedes Tile in den Drehungen `rotations` segmentiert;
             Tiles mit einem (auch unsicheren) Treffer zusätzlich in
             `refine_rotations`. Die Ergebnisse werden vereinigt.
          c) Wieder zusammenfügen: Die Silhouetten jedes Tiles werden exakt
             (dicht, per inverser Abbildung) ins Fisheye-Bild
             zurückprojiziert und zu einer Gesamtmaske vereinigt.

        fov_deg:      Bildwinkel über den Durchmesser des Bildkreises (Grad).
        projection:   Abbildungsfunktion des Objektivs ("equidistant",
                      "equisolid" oder "stereographic").
        center/radius: Mittelpunkt (x, y) und Radius des Bildkreises in
                      Pixeln; None = Bildmitte bzw. halbe kürzere Bildseite.
        tile_size:    Kantenlänge der Tiles in Pixeln; None = automatisch
                      aus der Auflösung des Fisheye-Bildes (640–1024 px).

        Hinweis: Vorwärts- und Rückprojektion verwenden dasselbe Modell.
        Ein ungenauer Bildwinkel führt daher nur zu leicht verzerrten
        Tiles, nicht zu verschobenen Masken.

        Rückgabe: 1-Kanal-Maske (uint8, 0/255) in der Grösse des Bildes.
        """
        img_h, img_w = img_bgr.shape[:2]
        geo = FisheyeGeometry(img_w, img_h, fov_deg, projection, center, radius)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if tile_size is None:
            tile_size = geo.native_tile_size(tile_fov_deg)
        if margin_px is None:
            margin_px = max(2, round(0.006 * tile_size))
        kernel = None
        if margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
            )

        # a) Schneiden (Abbildungs-Tabellen pro Bildgrösse zwischenspeichern)
        key = ("fisheye", img_w, img_h, fov_deg, projection,
               None if center is None else tuple(center), radius, tile_fov_deg, tile_size)
        views = self._map_cache.get(key)
        if views is None:
            views = []
            for forward, up in geo.tile_views(tile_fov_deg):
                basis = _tile_basis(forward, up)
                map_x, map_y = geo.tile_to_fisheye_maps(basis, tile_size, tile_fov_deg)
                views.append((basis, map_x, map_y))
            self._remember_maps(key, views)

        tiles = [
            cv2.remap(img_bgr, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            for _, map_x, map_y in views
        ]

        # b) Segmentieren, Stufe 1: alle Tiles in `rotations`
        tile_masks = [np.zeros((tile_size, tile_size), dtype=np.uint8) for _ in tiles]
        hits = [False] * len(tiles)
        self._segment_tiles_rotated(tiles, list(range(len(tiles))), rotations,
                                    tile_size, tile_masks, hits)
        # Stufe 2: nur Tiles mit (auch unsicherem) Treffer in `refine_rotations`
        refine_idx = [i for i, hit in enumerate(hits) if hit]
        if refine_rotations and refine_idx:
            self._segment_tiles_rotated(tiles, refine_idx, refine_rotations,
                                        tile_size, tile_masks, None)

        # c) Zusammenfügen
        for (basis, map_x, map_y), tile_mask in zip(views, tile_masks):
            if not np.any(tile_mask):
                continue
            if kernel is not None:
                tile_mask = cv2.dilate(tile_mask, kernel, iterations=1)
            # Nur Pixel innerhalb des Bildkreises sind gültig
            tile_mask[map_x == -1] = 0

            # Bereich im Fisheye-Bild bestimmen, den die Tile-Maske abdeckt,
            # und dort für jedes Pixel exakt nachschlagen, ob es im Tile
            # maskiert ist.
            ys, xs = np.nonzero(tile_mask)
            if len(xs) == 0:
                continue
            fx, fy = map_x[ys, xs], map_y[ys, xs]
            pad = margin_px + 4
            x0 = max(0, int(np.floor(fx.min())) - pad)
            x1 = min(img_w, int(np.ceil(fx.max())) + pad + 1)
            y0 = max(0, int(np.floor(fy.min())) - pad)
            y1 = min(img_h, int(np.ceil(fy.max())) + pad + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            u_map, v_map = geo.fisheye_to_tile_maps(basis, tile_size, tile_fov_deg, x0, x1, y0, y1)
            sampled = cv2.remap(
                tile_mask, u_map, v_map,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            np.maximum(mask[y0:y1, x0:x1], sampled, out=mask[y0:y1, x0:x1])

        # Ausserhalb des Bildkreises gibt es keine Bildinhalte
        mask[~geo.inside_circle_mask()] = 0
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=1)
        return mask

    def _segment_tiles_rotated(self, tiles, indices, rotations, imgsz, tile_masks, hits):
        """
        Segmentiert die Tiles `indices` in allen `rotations` (gebündelt),
        dreht die Masken zurück und vereinigt sie in tile_masks. Ist `hits`
        eine Liste, wird dort vermerkt, welche Tiles einen Kandidaten ab
        FISHEYE_REFINE_TRIGGER_CONF enthalten.
        """
        jobs = [(i, deg) for i in indices for deg in rotations]
        if not jobs:
            return
        images = [_rotate(tiles[i], deg) for i, deg in jobs]
        results = self._segment_batch(images, imgsz=imgsz,
                                      min_conf=FISHEYE_REFINE_TRIGGER_CONF)
        for (i, deg), img, (polygons, max_conf) in zip(jobs, images, results):
            if hits is not None and max_conf >= FISHEYE_REFINE_TRIGGER_CONF:
                hits[i] = True
            if not polygons:
                continue
            vmask = np.zeros(img.shape[:2], dtype=np.uint8)
            for poly in polygons:
                cv2.fillPoly(vmask, [poly.reshape(-1, 1, 2)], 255)
            np.maximum(tile_masks[i], _rotate(vmask, -deg), out=tile_masks[i])


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


# =====================================================================
# Fisheye-Geometrie
# =====================================================================
# Kamerakoordinaten: x nach rechts, y nach unten (wie die Bildachsen),
# z entlang der optischen Achse (Blickrichtung). Ein Bildpunkt im
# Abstand r vom Kreismittelpunkt entspricht einem Strahl mit dem Winkel
# θ zur optischen Achse, gemäss der Abbildungsfunktion des Objektivs.

class FisheyeGeometry:
    def __init__(self, img_w, img_h, fov_deg, projection="equidistant", center=None, radius=None):
        if projection not in ("equidistant", "equisolid", "stereographic"):
            raise ValueError(f"Unbekannte Fisheye-Projektion: {projection}")
        if not 0 < fov_deg < 360:
            raise ValueError("Bildwinkel muss zwischen 0 und 360 Grad liegen")
        self.img_w, self.img_h = img_w, img_h
        self.cx, self.cy = center if center is not None else ((img_w - 1) / 2.0, (img_h - 1) / 2.0)
        self.radius = float(radius) if radius is not None else min(img_w, img_h) / 2.0
        self.projection = projection
        self.theta_max = math.radians(fov_deg) / 2.0
        self._norm = self._r(self.theta_max)

    # Abbildungsfunktion r(θ) (ohne Brennweite) und deren Umkehrung
    def _r(self, theta):
        if self.projection == "equisolid":
            return 2.0 * np.sin(theta / 2.0)
        if self.projection == "stereographic":
            return 2.0 * np.tan(theta / 2.0)
        return theta

    def _theta(self, r):
        if self.projection == "equisolid":
            return 2.0 * np.arcsin(np.clip(r / 2.0, -1.0, 1.0))
        if self.projection == "stereographic":
            return 2.0 * np.arctan(r / 2.0)
        return r

    def theta_to_rho(self, theta):
        """Winkel zur optischen Achse -> Radius relativ zum Bildkreis (0..1)."""
        return self._r(theta) / self._norm

    def rho_to_theta(self, rho):
        return self._theta(rho * self._norm)

    def native_tile_size(self, tile_fov_deg, min_size=640, max_size=1024):
        """Tile-Grösse, deren Auflösung in der Tile-Mitte etwa der des
        Fisheye-Bildes in der Bildmitte entspricht (begrenzt, auf 32 gerundet)."""
        px_per_rad = self.radius / self._norm  # dr/dθ bei θ = 0 (für alle Modelle 1·f)
        size = 2.0 * math.tan(math.radians(tile_fov_deg) / 2.0) * px_per_rad
        size = int(round(size / 32.0)) * 32
        return int(np.clip(size, min_size, max_size))

    def tile_views(self, tile_fov_deg):
        """
        Blickrichtungen der Tiles: eines entlang der optischen Achse und
        ein Ring (bei sehr grossem Bildwinkel zwei Ringe), der bis zum Rand
        des Bildkreises reicht. Die "Oben"-Richtung jedes Tiles entspricht
        der Bild-Oben-Richtung (Drehungen werden separat behandelt).
        Rückgabe: Liste von (forward, up) als 3D-Vektoren.
        """
        half_tile = math.radians(tile_fov_deg) / 2.0
        views = [(np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))]

        rings = []
        alpha = min(self.theta_max - 0.6 * half_tile, math.radians(70))
        if self.theta_max > half_tile * 0.95 and alpha > math.radians(20):
            rings.append((alpha, 8))
        if self.theta_max - alpha > half_tile * 1.1:   # sehr grosser Bildwinkel
            rings.append((min(self.theta_max - 0.5 * half_tile, math.radians(120)), 12))

        for alpha, count in rings:
            for k in range(count):
                phi = 2 * math.pi * k / count + (math.pi / count if alpha > math.radians(90) else 0.0)
                fwd = np.array([math.sin(alpha) * math.cos(phi),
                                math.sin(alpha) * math.sin(phi),
                                math.cos(alpha)])
                up = np.array([0.0, -1.0, 0.0])
                if abs(np.dot(fwd, up)) > 0.97:
                    # Blick fast genau nach "oben": radiale Richtung als Oben
                    up = np.array([0.0, 0.0, -1.0])
                views.append((fwd, up))
        return views

    def tile_to_fisheye_maps(self, basis, tile_size, tile_fov_deg):
        """map_x/map_y (float32) für cv2.remap: Tile-Pixel -> Fisheye-Pixel.
        Punkte ausserhalb des Bildkreises erhalten -1."""
        right, down, fwd = basis
        f = (tile_size / 2.0) / math.tan(math.radians(tile_fov_deg) / 2.0)
        c = np.arange(tile_size) - tile_size / 2.0 + 0.5
        gx, gy = np.meshgrid(c, c)
        dx = right[0] * gx + down[0] * gy + fwd[0] * f
        dy = right[1] * gx + down[1] * gy + fwd[1] * f
        dz = right[2] * gx + down[2] * gy + fwd[2] * f
        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / norm, dy / norm, dz / norm

        theta = np.arccos(np.clip(dz, -1.0, 1.0))
        rho = self.theta_to_rho(theta)
        planar = np.sqrt(dx * dx + dy * dy)
        planar[planar < 1e-12] = 1.0
        map_x = self.cx + self.radius * rho * dx / planar
        map_y = self.cy + self.radius * rho * dy / planar
        outside = theta > self.theta_max
        map_x[outside] = -1
        map_y[outside] = -1
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def fisheye_to_tile_maps(self, basis, tile_size, tile_fov_deg, x0, x1, y0, y1):
        """Inverse Abbildung für den Ausschnitt [y0:y1, x0:x1] des Fisheye-
        Bildes: Fisheye-Pixel -> Tile-Pixel (float32, -1 = ausserhalb)."""
        right, down, fwd = basis
        f = (tile_size / 2.0) / math.tan(math.radians(tile_fov_deg) / 2.0)
        xs = (np.arange(x0, x1) - self.cx) / self.radius
        ys = (np.arange(y0, y1) - self.cy) / self.radius
        nx, ny = np.meshgrid(xs, ys)
        rho = np.sqrt(nx * nx + ny * ny)
        theta = self.rho_to_theta(rho)
        safe = np.where(rho < 1e-12, 1.0, rho)
        st = np.sin(theta)
        dx, dy, dz = st * nx / safe, st * ny / safe, np.cos(theta)

        tx = dx * right[0] + dy * right[1] + dz * right[2]
        ty = dx * down[0] + dy * down[1] + dz * down[2]
        tz = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
        valid = (tz > 1e-6) & (rho <= 1.0)
        tz_safe = np.where(valid, tz, 1.0)
        u = tx / tz_safe * f + tile_size / 2.0 - 0.5
        v = ty / tz_safe * f + tile_size / 2.0 - 0.5
        u[~valid] = -1
        v[~valid] = -1
        return u.astype(np.float32), v.astype(np.float32)

    def inside_circle_mask(self):
        yy, xx = np.mgrid[0:self.img_h, 0:self.img_w]
        r2 = (xx - self.cx) ** 2 + (yy - self.cy) ** 2
        return r2 <= self.radius ** 2


def _tile_basis(forward, up_hint):
    """Orthonormale Tile-Basis (rechts, unten, vorne) aus Blickrichtung und
    gewünschter Oben-Richtung."""
    fwd = forward / np.linalg.norm(forward)
    up = up_hint - np.dot(up_hint, fwd) * fwd
    up = up / np.linalg.norm(up)
    down = -up
    right = np.cross(down, fwd)
    return right, down, fwd


def _rotate(img, deg):
    """Dreht ein Bild um ein Vielfaches von 90 Grad (positiv = im Uhrzeigersinn)."""
    deg = deg % 360
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img
