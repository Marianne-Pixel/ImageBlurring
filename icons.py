"""
icons.py
Erzeugt schematische Graustufen-Icons für die drei Bildtypen im GUI.

Jedes Icon zeigt denselben Raum (Innenansicht eines Würfels mit Gitter
auf allen Wänden), abgebildet mit der jeweiligen Projektion:
  - Equirectangular: ganze Kugel, 2:1-Format, Gitterlinien gekrümmt
  - Fisheye:         runder Bildkreis (180°, equidistant)
  - Normal:          Lochkamera-Perspektive, Gitterlinien gerade

Die Icons werden beim Programmstart berechnet (keine Bilddateien nötig)
und als RGBA-PIL-Bilder zurückgegeben (ausserhalb des Fisheye-Kreises
transparent).
"""

import math

import cv2
import numpy as np
from PIL import Image

# Grauwerte der Würfelflächen (vorne, hinten, links, rechts, Decke, Boden)
_FACE_GRAY = {
    "front": 175, "back": 150, "left": 205, "right": 135,
    "top": 225, "bottom": 105,
}
_GRID_GRAY = 70         # feine Gitterlinien
_EDGE_GRAY = 25         # Würfelkanten
_GRID_DIVISIONS = 4     # Gitterfelder pro Würfelfläche
_SUPERSAMPLE = 4        # Kantenglättung


def _shade(dx, dy, dz):
    """Grauwert für Blickrichtungen (x rechts, y unten, z vorne) im Würfel."""
    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)
    out = np.zeros(dx.shape, dtype=np.float32)

    def face(sel, a, b, gray):
        # a, b: Koordinaten auf der Fläche im Bereich -1..1
        g = np.full(a.shape, gray, dtype=np.float32)
        grid_a = np.abs((a + 1) * _GRID_DIVISIONS / 2 - np.round((a + 1) * _GRID_DIVISIONS / 2))
        grid_b = np.abs((b + 1) * _GRID_DIVISIONS / 2 - np.round((b + 1) * _GRID_DIVISIONS / 2))
        g[(grid_a < 0.05) | (grid_b < 0.05)] = _GRID_GRAY
        g[(np.abs(a) > 0.985) | (np.abs(b) > 0.985)] = _EDGE_GRAY
        out[sel] = g[sel]

    with np.errstate(divide="ignore", invalid="ignore"):
        sel = (az >= ax) & (az >= ay)
        m = np.where(sel, az, 1.0)
        face(sel & (dz > 0), dx / m, dy / m, _FACE_GRAY["front"])
        face(sel & (dz <= 0), dx / m, dy / m, _FACE_GRAY["back"])
        sel = (ax > az) & (ax >= ay)
        m = np.where(sel, ax, 1.0)
        face(sel & (dx < 0), dz / m, dy / m, _FACE_GRAY["left"])
        face(sel & (dx >= 0), dz / m, dy / m, _FACE_GRAY["right"])
        sel = (ay > az) & (ay > ax)
        m = np.where(sel, ay, 1.0)
        face(sel & (dy < 0), dx / m, dz / m, _FACE_GRAY["top"])
        face(sel & (dy >= 0), dx / m, dz / m, _FACE_GRAY["bottom"])
    return out


def _finish(gray, alpha, size):
    """Herunterskalieren (Kantenglättung) und als RGBA-PIL-Bild zurückgeben."""
    w, h = size
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    g = np.clip(gray, 0, 255).astype(np.uint8)
    a = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([g, g, g, a]), mode="RGBA")


def _grid(w, h):
    s = _SUPERSAMPLE
    xs = (np.arange(w * s) + 0.5) / (w * s)
    ys = (np.arange(h * s) + 0.5) / (h * s)
    return np.meshgrid(xs, ys)


def equirect_icon(height=48):
    w, h = 2 * height, height
    u, v = _grid(w, h)
    lon = (u - 0.5) * 2 * math.pi
    lat = (0.5 - v) * math.pi
    dx = np.cos(lat) * np.sin(lon)
    dy = -np.sin(lat)
    dz = np.cos(lat) * np.cos(lon)
    gray = _shade(dx, dy, dz)
    return _finish(gray, np.ones_like(gray), (w, h))


def fisheye_icon(size=48, fov_deg=180):
    u, v = _grid(size, size)
    nx, ny = u * 2 - 1, v * 2 - 1
    rho = np.sqrt(nx * nx + ny * ny)
    theta = rho * math.radians(fov_deg) / 2          # equidistant
    safe = np.where(rho < 1e-9, 1.0, rho)
    dx = np.sin(theta) * nx / safe
    dy = np.sin(theta) * ny / safe
    dz = np.cos(theta)
    gray = _shade(dx, dy, dz)
    inside = (rho <= 1.0).astype(np.float32)
    gray[inside == 0] = 0
    # dunkler Rand des Bildkreises
    gray[(rho > 0.96) & (rho <= 1.0)] = _EDGE_GRAY
    return _finish(gray, inside, (size, size))


def normal_icon(size=48, fov_deg=125):
    u, v = _grid(size, size)
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    dx, dy = u * 2 - 1, v * 2 - 1
    dz = np.full_like(dx, f)
    gray = _shade(dx, dy, dz)
    return _finish(gray, np.ones_like(gray), (size, size))


def projection_icons(height=48):
    """Dict bildtyp -> PIL.Image (RGBA) für 'equirect', 'fisheye', 'normal'."""
    return {
        "equirect": equirect_icon(height),
        "fisheye": fisheye_icon(height),
        "normal": normal_icon(height),
    }
