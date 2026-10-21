"""
processing.py
Bildbearbeitung:
  - Zwei wählbare Verwisch-Methoden ("gaussian" / "pixelate"), angewendet
    auf eine Maske, die exakt der Silhouette jeder erkannten Person
    entspricht (kein Rechteck oder Oval).
  - Zählen der verwischten Bereiche (für die Abschluss-Statistik im GUI).
  - Automatische Belichtungskorrektur: hellt Schatten auf und dunkelt
    Lichter ab (Tonwertkurve auf dem L-Kanal, LAB-Farbraum), mit zwei
    unabhängig einstellbaren Reglern (Schatten-/Lichter-Intensität) und
    Sicherheitsprüfung gegen Überbelichtung.
"""

import numpy as np
import cv2

BLUR_METHODS = ("gaussian", "pixelate")


def _gaussian_kernel_from_strength(strength, max_dim=None):
    """Wandelt die Stärke (1..30) in eine ungerade Kernelgrösse für Gaussian Blur um."""
    k = max(3, int(round(strength)) * 2 + 1)
    if max_dim is not None and max_dim >= 3:
        limit = max_dim if max_dim % 2 == 1 else max_dim - 1
        k = min(k, limit)
    if k % 2 == 0:
        k += 1
    return max(3, k)


def _pixel_block_from_strength(strength):
    """Wandelt die Stärke (1..30) in eine Blockgrösse für die Pixelierung um."""
    return max(2, int(round(strength)))


def apply_blur_effect(roi, method="gaussian", strength=12):
    """Wendet die gewählte Verwisch-Methode auf einen Bildausschnitt an."""
    h, w = roi.shape[:2]
    if h < 1 or w < 1:
        return roi
    if method == "pixelate":
        block = _pixel_block_from_strength(strength)
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:  # "gaussian" (Standard)
        k = _gaussian_kernel_from_strength(strength, max_dim=min(h, w))
        return cv2.GaussianBlur(roi, (k, k), 0)


def build_mask_from_polygons(img_shape, polygons, margin_px=None):
    """
    Erstellt eine binäre Maske exakt entlang der erkannten Personen-
    Silhouetten (kein Rechteck, kein Oval). margin_px ist ein sehr kleiner,
    fester Sicherheitsrand in Pixeln (z.B. für Haarsträhnen am Rand), damit
    möglichst wenig Hintergrund mitverwischt wird.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)

    if margin_px is None:
        margin_px = max(2, round(0.004 * min(h, w)))

    if margin_px > 0 and np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def blur_mask(img_bgr, mask, method="gaussian", strength=18):
    """
    Wendet die gewählte Verwisch-Methode auf alle Pixel an, die in der
    Maske gesetzt sind (Personen-Silhouette(n) bzw. Panorama-Maske aus
    detector.detect_mask_equirect). Der Übergang am Maskenrand wird sanft
    ausgeblendet, damit keine harte Kante entsteht.
    """
    if mask is None or not np.any(mask):
        return img_bgr.copy()

    effect_full = apply_blur_effect(img_bgr, method=method, strength=strength)
    feather = max(3, (max(img_bgr.shape[:2]) // 150) | 1)
    soft_mask = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
    soft_mask3 = soft_mask[..., None]
    out = (effect_full * soft_mask3 + img_bgr * (1 - soft_mask3)).astype(np.uint8)
    return out


def count_regions(mask):
    """Zählt die Anzahl zusammenhängender, tatsächlich verwischter Bereiche."""
    if mask is None or not np.any(mask):
        return 0
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return max(0, num_labels - 1)  # Hintergrund (Label 0) nicht mitzählen


def auto_correct_exposure(
    img_bgr,
    shadow_intensity=22,
    highlight_intensity=22,
    max_mean_gain=55,
    max_overexposed_gain=0.03,
):
    """
    Hellt Schattenbereiche auf und dunkelt sehr helle Bereiche ab, über
    eine Tonwertkurve auf dem L-Kanal (LAB-Farbraum). Beide Effekte sind
    unabhängig voneinander über je einen eigenen Regler steuerbar:

    shadow_intensity (0..100): Stärke der Schattenaufhellung.
    0 = keine Aufhellung, 100 = maximale Wirkung.

    highlight_intensity (0..100): Stärke der Lichter-Abdunklung
    (Kompression sehr heller Bereiche). 0 = keine Abdunklung,
    100 = maximale Wirkung.

    Vorgehen:
      1. Dunkle Pixel werden über eine Potenzfunktion angehoben; die
         Stärke des Effekts nimmt zu helleren Tonwerten hin weich ab
         (voller Effekt bei sehr dunklen Pixeln, praktisch kein Effekt
         mehr ab einem mittleren Tonwert).
      2. Sehr helle Pixel werden weich in Richtung Mittelton komprimiert,
         unabhängig von Schritt 1.
      3. Sicherheitsprüfung: Steigt die mittlere Helligkeit oder der
         Anteil überbelichteter Pixel (L > 245) durch die Korrektur zu
         stark an, wird die Korrektur automatisch abgeschwächt.

    Rückgabe: (korrigiertes_bild, helligkeits_differenz)
    helligkeits_differenz = mittlere Helligkeit nachher minus vorher
    (positiv = aufgehellt, negativ = abgedunkelt, ~0 = kaum verändert).
    """
    shadow_strength = float(np.clip(shadow_intensity, 0, 100)) / 100.0
    highlight_strength = float(np.clip(highlight_intensity, 0, 100)) / 100.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32) / 255.0

    orig_mean = float(l.mean())
    orig_over = float(np.mean(l > 245))

    if shadow_strength <= 0.0 and highlight_strength <= 0.0:
        return img_bgr.copy(), 0.0

    # --- Schatten aufhellen ---
    # Potenzkurve 1-(1-x)^gamma: hebt dunkle Werte an, lässt sehr helle
    # Werte (nahe 1) fast unverändert.
    gamma_shadow = 1.0 + 3.5 * shadow_strength
    lifted = 1.0 - np.power(1.0 - l_f, gamma_shadow)
    # Gewichtung: voller Effekt in dunklen Bereichen, weiches Ausblenden
    # bis zum Mittelton (~0.6), damit bereits helle Bildteile unangetastet bleiben.
    shadow_weight = np.clip(1.0 - l_f / 0.6, 0.0, 1.0)
    l_shadow = l_f * (1.0 - shadow_weight) + lifted * shadow_weight

    # --- Lichter abdunkeln / komprimieren (unabhängig von der Schattenstärke) ---
    highlight_weight = np.clip((l_shadow - 0.6) / 0.4, 0.0, 1.0)
    compressed = 0.6 + (l_shadow - 0.6) * (1.0 - highlight_strength)
    l_final = l_shadow * (1.0 - highlight_weight) + compressed * highlight_weight

    l_new = np.clip(l_final * 255.0, 0, 255).astype(np.uint8)

    new_mean = float(l_new.mean())
    new_over = float(np.mean(l_new > 245))

    if (new_mean > orig_mean + max_mean_gain) or (new_over > orig_over + max_overexposed_gain):
        for alpha in (0.8, 0.6, 0.4, 0.2):
            candidate = cv2.addWeighted(l_new, alpha, l, 1 - alpha, 0)
            c_mean, c_over = float(candidate.mean()), float(np.mean(candidate > 245))
            if c_mean <= orig_mean + max_mean_gain and c_over <= orig_over + max_overexposed_gain:
                l_new = candidate
                new_mean = c_mean
                break
        else:
            l_new = l
            new_mean = orig_mean

    merged = cv2.merge([l_new, a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out, (new_mean - orig_mean)
