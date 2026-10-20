# Blueprint: Personen-Anonymisierung (Theta Z1 & normale Bilder)

> Kompakte Referenz, um dieses Tool in einem neuen Chat identisch
> nachzubauen. Enthält Ziel, Tech-Stack, Architektur, Edge Cases und einen
> fertigen Generierungs-Prompt (Abschnitt 5).

---

## 1. Ziel & Funktion

Ein Desktop-Tool mit GUI, das Personen in Fotos automatisch erkennt und
unkenntlich macht (Anonymisierung), gedacht für zwei Bildarten:

- **Normale Fotos** (kaum verzerrt)
- **Equirectangular-360°-Panoramen der Ricoh Theta Z1**

Ablauf aus Nutzersicht:
1. Quellordner wählen → wird rekursiv nach `.jpg`/`.jpeg`/`.png` durchsucht.
2. Speicherort wählen → bearbeitete Bilder werden **dorthin** geschrieben
   (Originale bleiben unverändert), mit gleicher Unterordner-Struktur wie
   im Quellordner.
3. Gefundene Bilder in einer Liste auswählen (Mehrfachauswahl), daneben
   Thumbnail-Vorschau des markierten Bildes.
4. Verwisch-Methode (Gaussian Blur ODER Pixelieren/Mosaik) und Stärke
   (Regler 1–30) wählen, mit Live-Vorschau an einem fiktiven Beispielbild.
5. Optional: automatische Belichtungskorrektur aktivieren, mit zwei
   unabhängigen Reglern für "Schatten aufhellen" und "Lichter abdunkeln"
   einstellen.
6. Checkbox: Bilder sind Theta-Z1-Panoramen (Standard: **aktiviert**) —
   steuert den Detektions-Modus.
7. "Start" → Bestätigungsdialog (Ja/Nein/Abbrechen) mit Anzahl der Bilder
   → Verarbeitung im Hintergrund-Thread mit Fortschrittsbalken.
8. Abschluss-Zusammenfassung: Anzahl verwischter Bereiche/Bilder, Anzahl
   auf-/abgehellter Bilder, ggf. Fehlerliste.

Kernprinzip: Es wird **nur die tatsächliche Personen-Silhouette**
verwischt (kein Rechteck/Oval), so klein wie möglich gehalten.

---

## 2. Tech-Stack

- **Sprache:** Python 3.9+
- **GUI:** `tkinter` / `tkinter.ttk` (Standardbibliothek)
- **Bildverarbeitung:** `opencv-python` (>=4.9.0) — `cv2`
- **Numerik:** `numpy` (>=1.26.0)
- **Bild-I/O für GUI-Vorschauen:** `Pillow` (>=10.0.0) — `PIL.Image`,
  `PIL.ImageTk`
- **Personendetektion:** `ultralytics` (>=8.3.0), Modell
  **YOLO11-Segmentierung** (`yolo11s-seg.pt`, COCO-Klasse "person" = ID 0)
  — nicht das reine Detektionsmodell, sondern die Segmentierungsvariante,
  da pixelgenaue Silhouetten (Polygone) statt Bounding-Boxes benötigt
  werden.
- Keine Web-/Cloud-Abhängigkeit zur Laufzeit ausser dem einmaligen
  automatischen Download der Modellgewichte beim ersten Start.

Dateistruktur:
```
personen_anonymisierer/
├── main.py            # GUI, Ablaufsteuerung, Verarbeitung
├── detector.py         # YOLO11-Segmentierung, Equirectangular-Tiling
├── processing.py        # Blur/Pixelierung, Maskenaufbau, Belichtung
└── requirements.txt       # ultralytics, opencv-python, numpy, Pillow
```

---

## 3. Architektur & Logik

### `detector.py` — `PersonDetector`

- `__init__(model_path="yolo11s-seg.pt", conf_threshold=0.35, device=None)`
  lädt `YOLO(model_path)`.
- `detect_masks_normal(img_bgr)` → Liste von Polygonen (int32-Arrays,
  Nx2) je erkannter Person, aus `model.predict(..., classes=[0]).masks.xy`
  (bereits in Original-Bildkoordinaten).
- `detect_mask_equirect(img_bgr, tile_size=900, fov_deg=100, margin_px=None)`
  → 1-Kanal-Maske (uint8, Grösse = Panorama):
  1. **Tiling:** ~16 Blickrichtungen (yaw/pitch) decken die Kugel
     überlappend ab: Äquatorband (yaw 0/60/…/300°, pitch 0°), oberes/
     unteres Band (yaw 0/90/180/270°, pitch ±45°), Zenit/Nadir (pitch
     ±90°).
  2. Für jede Blickrichtung: `_equirect_to_persp_maps()` berechnet
     `map_x`/`map_y` (klassische Pinhole-Projektion mit Yaw-/Pitch-
     Rotationsmatrizen) → `cv2.remap()` erzeugt das perspektivische Tile.
  3. Segmentierung pro Tile → Polygone → in Tile-Maske gefüllt
     (`cv2.fillPoly`) + kleiner Dilatations-Rand.
  4. **Rückprojektion:** Da `map_x`/`map_y` für jedes Tile-Pixel bereits
     die zugehörige Panorama-Koordinate liefern, werden maskierte
     Tile-Pixel direkt (Scatter) in die Panorama-Maske eingetragen
     (`mask[eq_y, eq_x] = 255`, x-Koordinate modulo `eq_w` für den
     360°-Umbruch).
  5. Kleine Lücken aus der Rückprojektion per Dilate+Close schliessen.

### `processing.py`

- `apply_blur_effect(roi, method, strength)` — zwei Modi:
  - `"gaussian"`: `cv2.GaussianBlur` mit Kernel `k = 2·strength+1`
    (ungerade, an ROI-Grösse geclippt).
  - `"pixelate"`: Downscale auf `w//strength × h//strength`
    (`INTER_LINEAR`), dann Upscale zurück mit `INTER_NEAREST`.
- `build_mask_from_polygons(img_shape, polygons, margin_px=None)` —
  füllt alle Polygone in eine leere Maske, dilatiert um einen sehr
  kleinen festen Pixel-Rand (Default: `max(2, 0.004·min(h,w))`), damit
  möglichst wenig Hintergrund verwischt wird.
- `blur_mask(img_bgr, mask, method, strength)` — berechnet den Effekt
  einmal auf dem Gesamtbild, blendet ihn über eine weichgezeichnete
  (gefeatherte) Version der Maske ein (kein harter Rand).
- `count_regions(mask)` — `cv2.connectedComponents` minus Hintergrund;
  liefert Anzahl getrennter verwischter Bereiche.
- `auto_correct_exposure(img_bgr, shadow_intensity=22, highlight_intensity=22, max_mean_gain=55, max_overexposed_gain=0.03)`
  → `(bild, helligkeits_differenz)`:
  1. LAB-Farbraum, nur L-Kanal wird verändert.
  2. `shadow_strength = shadow_intensity/100`, `highlight_strength =
     highlight_intensity/100` — **vollständig unabhängig** voneinander
     (nicht mehr aneinander gekoppelt).
  3. Schatten: `lifted = 1 - (1-L)^gamma` mit `gamma = 1 + 3.5·shadow_strength`,
     Gewichtung `shadow_weight = clip(1 - L/0.6, 0, 1)` (voller Effekt bei
     L=0, ausgeblendet ab Mittelton ~0.6).
  4. Lichter: sanfte Kompression Richtung 0.6 oberhalb dieses Mitteltons,
     mit `highlight_strength` skaliert — unabhängig von Schritt 3.
  5. Sicherheitsprüfung: steigt mittlere Helligkeit oder Anteil
     überbelichteter Pixel (L>245) zu stark, wird das Ergebnis stufenweise
     (`alpha` 0.8/0.6/0.4/0.2) mit dem Original rückgemischt.

### `main.py` — GUI (`tkinter.ttk`)

**Fenster-Layout (von oben nach unten):**
1. Zeile "Ordner:" + readonly Entry + Button "Durchsuchen…" →
   `filedialog.askdirectory()` → `scan_images()` (rekursiv, `os.walk`,
   Endungen `.jpg/.jpeg/.png`, case-insensitive) → befüllt Listbox mit
   relativen Pfaden.
2. Zeile "Speicherort:" + readonly Entry + Button "Durchsuchen…" (separate
   `StringVar`).
3. Content-Bereich (nebeneinander):
   - `Listbox` (`selectmode="extended"`) + Scrollbar + Buttons
     "Alle auswählen"/"Auswahl aufheben"; `<<ListboxSelect>>`-Binding
     aktualisiert die Thumbnail-Vorschau rechts.
   - `LabelFrame` "Vorschau": `ttk.Label` mit `PIL.Image.open(path).thumbnail((190,150))`
     → `ImageTk.PhotoImage`.
4. `LabelFrame` "Verwisch-Einstellungen": Radiobuttons Methode
   (`gaussian`/`pixelate`), `ttk.Scale` Stärke (1–30, Default **22**),
   Live-Vorschau-Label rechts, aktualisiert bei jeder Reglerbewegung über
   ein fiktives Beispielbild (`_create_fictive_sample()`: 240×160 Bild mit
   Gitter-Hintergrund + gezeichneter Kopf/Schulter-Form + zugehöriges
   unregelmässiges Silhouetten-Polygon).
5. `LabelFrame` "Belichtungskorrektur": Checkbox aktivieren + zwei
   **unabhängige** `ttk.Scale`-Regler: "Schatten aufhellen" (0–100,
   Default **22**) und "Lichter abdunkeln" (0–100, Default **22**). Kein
   Vorschaubild.
6. `LabelFrame` "Bildtyp": Checkbox "Theta Z1 (Equirectangular)" —
   Standardwert **aktiviert** (`True`).
7. Statuszeile + `ttk.Progressbar` + Button "Start" (rechtsbündig).

**Custom-Dialog `ConfirmDialog(tk.Toplevel)`** — genau 3 Buttons "Ja" /
"Nein" / "Abbrechen" (kein Standard-`messagebox`, da dessen Labels nicht
frei benennbar sind), `result` ∈ `{"yes","no","cancel"}`.

**Start-Ablauf (`on_start_clicked`):**
1. Keine Auswahl → `messagebox.showinfo`, Abbruch.
2. Kein Speicherort gewählt → `messagebox.showinfo`, Abbruch.
3. `ConfirmDialog` mit Text: *"Achtung. Falls im Speicherort bereits
   Dateien mit gleichem Namen vorhanden sind, werden diese überschrieben.
   Es werden {N} Bild(er) bearbeitet. Soll der Prozess gestartet werden?"*
4. Bei "yes": `settings`-Dict sammeln (Theta-Flag, Belichtung an/aus +
   Schatten-/Lichter-Intensität, Methode + Stärke, Quell-/Zielordner) →
   Verarbeitung in `threading.Thread(..., daemon=True)`.

**`process_images(paths, settings)`** (läuft im Hintergrund-Thread):
- Modell wird lazy beim ersten Aufruf geladen (`PersonDetector(...)`).
- Pro Bild: `cv2.imread` → je nach Theta-Flag `detect_mask_equirect()`
  oder `detect_masks_normal()` + `build_mask_from_polygons()` →
  `count_regions()` → falls >0: `blur_mask()` anwenden, Zähler erhöhen →
  falls Belichtungskorrektur aktiv: `auto_correct_exposure()`, Zähler
  aufgehellt/abgedunkelt (`mean_diff` > 1 / < -1) → Ziel-Pfad =
  `output_folder + relpath(quellpfad, quellordner)`,
  `os.makedirs(..., exist_ok=True)`, `cv2.imwrite(dest_path, img)`.
- Alle UI-Updates (Status, Fortschritt, Abschluss-Dialog) über
  `self.after(0, lambda: ...)` (Thread-Sicherheit für Tkinter).
- Abschluss: `messagebox.showinfo`/`showwarning` mit
  Zusammenfassungstext (Anzahl Bilder, Bereiche, auf-/abgehellte Bilder,
  ggf. Fehlerliste).

---

## 4. Edge Cases & Fehlerbehandlung

- **Keine Bildauswahl / kein Speicherort:** Start wird mit `messagebox`
  verhindert, kein stiller Fehlschlag.
- **Beschädigte/unlesbare Bilddatei:** `cv2.imread` liefert `None` →
  `RuntimeError` wird abgefangen, Dateiname landet in der Fehlerliste,
  Verarbeitung der übrigen Bilder läuft weiter.
- **Speichern schlägt fehl** (`cv2.imwrite` → `False`): ebenfalls
  abgefangen und protokolliert statt Absturz.
- **Keine Person erkannt:** Maske ist leer (`np.any(mask)` False) →
  `blur_mask`/`count_regions` geben unverändertes Bild bzw. 0 zurück,
  kein Fehler.
- **Gaussian-Kernel zu gross für kleine ROI:** Kernelgrösse wird auf
  `min(ROI-Höhe, ROI-Breite)` geclippt und auf ungerade Zahl korrigiert
  (OpenCV verlangt ungerade Kernel).
- **Equirectangular Rückprojektions-Lücken:** Da die Vorwärts-Abbildung
  Tile→Panorama nicht 1:1 dicht ist, entstehen beim Scatter kleine
  Löcher in der Maske → werden mit Dilate+Morphological-Close
  geschlossen.
- **180°/-180°-Naht des Panoramas:** x-Koordinaten werden `modulo eq_w`
  berechnet, damit der Längengrad-Umbruch keine Lücke erzeugt.
- **Belichtungskorrektur überbelichtet das Bild:** Sicherheitsprüfung
  vergleicht mittlere Helligkeit und Anteil Pixel > 245 vor/nach; bei
  Überschreitung von `max_mean_gain`/`max_overexposed_gain` wird das
  Ergebnis stufenweise mit dem Original rückgemischt statt es zu
  verwerfen.
- **Threading & GUI:** Alle Tkinter-Widget-Änderungen aus dem
  Worker-Thread ausschliesslich über `self.after(0, ...)` — direkte
  Aufrufe aus einem Fremd-Thread sind in Tkinter nicht sicher.
- **Unterordner-Struktur am Speicherort:** `os.path.relpath()` +
  `os.makedirs(dirname, exist_ok=True)` vor jedem `imwrite`, damit
  verschachtelte Quellordner nicht zu Pfadfehlern führen.
- **Ordnerauswahl abgebrochen:** `filedialog.askdirectory()` liefert
  leeren String → Funktion kehrt sofort zurück, keine Zustandsänderung.
- **Mehrfachauswahl in der Liste:** Thumbnail-Vorschau zeigt bei genau
  1 Auswahl das Bild, bei >1 nur die Anzahl, bei 0 "Keine Auswahl".
- **Nur unterstützte Formate:** Groschreibung wird ignoriert
  (`f.lower().endswith((".jpg", ".jpeg", ".png"))`), andere Formate
  werden beim Scan ignoriert.

---

## 5. Finaler Generierungs-Prompt

> Alles zwischen den Trennlinien direkt in einen neuen Chat kopieren.

---

```
Erstelle mir ein Python-Tool zum Unkenntlichmachen (Anonymisieren) von
Personen in Bildern, aufgeteilt in main.py, detector.py, processing.py
und requirements.txt.

ZIEL: Ein Desktop-GUI-Tool (tkinter), das Personen in Fotos automatisch
per YOLO11-Segmentierung erkennt und ihre exakte Silhouette (kein
Rechteck/Oval) mit Gaussian Blur oder Pixelierung verwischt. Unterstützt
zwei Bildarten: normale Fotos und Equirectangular-360°-Panoramen der
Ricoh Theta Z1.

TECH-STACK: Python 3.9+, tkinter/ttk (GUI), opencv-python, numpy,
Pillow (für GUI-Bildvorschauen), ultralytics mit YOLO11-
Segmentierungsmodell "yolo11s-seg.pt" (COCO-Klasse "person", ID 0).

--- detector.py: Klasse PersonDetector ---
- __init__(model_path="yolo11s-seg.pt", conf_threshold=0.35, device=None)
- detect_masks_normal(img_bgr): läuft model.predict(img, conf=conf_threshold,
  classes=[0], verbose=False) und gibt eine Liste von Polygonen
  (int32 Nx2-Arrays aus results[i].masks.xy) zurück – die tatsächliche
  Personen-Silhouette, keine Bounding-Box.
- detect_mask_equirect(img_bgr, tile_size=900, fov_deg=100, margin_px=None):
  zerlegt das Panorama in ~16 überlappende perspektivische Tiles
  (Blickrichtungen yaw/pitch: Äquatorband yaw 0,60,...,300° bei pitch 0°;
  oberes/unteres Band yaw 0,90,180,270° bei pitch ±45°; Zenit/Nadir bei
  pitch ±90°), berechnet für jedes Tile via klassischer Pinhole-Projektion
  (Yaw-/Pitch-Rotationsmatrizen) eine Vorwärts-Abbildung map_x/map_y
  (Tile-Pixel -> Equirect-Koordinate), holt das Tile per cv2.remap,
  detektiert Personen-Polygone im Tile, füllt sie in eine Tile-Maske
  (cv2.fillPoly + kleine Dilatation), projiziert die maskierten
  Tile-Pixel direkt über map_x/map_y zurück ins Panorama-Koordinatensystem
  (Scatter, x-Koordinate modulo Panorama-Breite für den 360°-Umbruch),
  akkumuliert alle Tiles in einer gemeinsamen Panorama-Maske und schliesst
  kleine Lücken aus der Rückprojektion mit Dilate+MorphologicalClose.
  Gibt eine uint8-Maske (0/255) in Panorama-Grösse zurück.

--- processing.py ---
- apply_blur_effect(roi, method="gaussian", strength=12): bei "gaussian"
  cv2.GaussianBlur mit Kernel k=2*strength+1 (ungerade, auf ROI-Grösse
  geclippt); bei "pixelate" Downscale auf Breite/Höhe // strength
  (INTER_LINEAR) und Upscale zurück (INTER_NEAREST).
- build_mask_from_polygons(img_shape, polygons, margin_px=None): füllt
  alle Polygone in eine leere Maske (cv2.fillPoly), dilatiert um einen
  sehr kleinen festen Pixel-Rand (Default max(2, 0.004*min(h,w))), damit
  möglichst wenig Hintergrund verwischt wird.
- blur_mask(img_bgr, mask, method="gaussian", strength=18): berechnet den
  gewählten Effekt einmal auf dem Gesamtbild, blendet ihn über eine
  weichgezeichnete (gefeatherte) Version der Maske ein, damit kein harter
  Rand entsteht. Gibt bei leerer Maske das unveränderte Bild zurück.
- count_regions(mask): zählt über cv2.connectedComponents die Anzahl
  getrennter zusammenhängender Bereiche in der Maske (Hintergrund nicht
  mitgezählt).
- auto_correct_exposure(img_bgr, shadow_intensity=22, highlight_intensity=22,
  max_mean_gain=55, max_overexposed_gain=0.03): Belichtungskorrektur als
  Tonwertkurve auf dem L-Kanal im LAB-Farbraum (a/b-Kanäle unverändert).
  shadow_intensity und highlight_intensity (je 0-100) sind VOLLSTÄNDIG
  UNABHÄNGIG voneinander (nicht aneinander gekoppelt): shadow_strength =
  shadow_intensity/100, highlight_strength = highlight_intensity/100.
  Schatten-Aufhellung über Potenzkurve lifted = 1-(1-L)^gamma mit
  gamma = 1+3.5*shadow_strength, gewichtet mit shadow_weight =
  clip(1-L/0.6, 0, 1) (voller Effekt bei L=0, ausgeblendet ab Tonwert
  ~0.6). Lichter werden unabhängig davon weich Richtung 0.6 komprimiert,
  skaliert mit highlight_strength. Sicherheitsprüfung: wenn mittlere
  Helligkeit oder Anteil Pixel>245 zu stark steigt (über max_mean_gain /
  max_overexposed_gain), wird das Ergebnis stufenweise (alpha
  0.8/0.6/0.4/0.2) mit dem Original rückgemischt. Gibt
  (korrigiertes_bild, helligkeits_differenz) zurück (Differenz = neue
  minus alte mittlere Helligkeit).

--- main.py: GUI (tkinter/ttk) ---
Fenstertitel "Personen-Anonymisierung – Theta Z1 / Normale Bilder".
Layout von oben nach unten:
1. Zeile "Ordner:" mit readonly Entry + Button "Durchsuchen…"
   (filedialog.askdirectory) -> durchsucht den Ordner rekursiv
   (os.walk) nach .jpg/.jpeg/.png (case-insensitive) und befüllt eine
   Listbox mit den relativen Pfaden.
2. Zeile "Speicherort:" mit eigenem readonly Entry + Button
   "Durchsuchen…" -> separater Ausgabeordner, in den die bearbeiteten
   Bilder gespeichert werden (Originale bleiben unverändert).
3. Content-Bereich nebeneinander: links Listbox (selectmode="extended")
   mit Scrollbar und Buttons "Alle auswählen"/"Auswahl aufheben"; rechts
   ein LabelFrame "Vorschau" mit einem Label, das bei Auswahl genau
   eines Listeneintrags per PIL (Image.open(path).thumbnail((190,150)))
   ein Thumbnail anzeigt (bei 0 Auswahl "Keine Auswahl", bei >1 Auswahl
   die Anzahl als Text).
4. LabelFrame "Verwisch-Einstellungen": Radiobuttons für Methode
   ("Gaussian Blur" / "Pixelieren (Mosaik)"), ttk.Scale für Stärke
   (1 bis 30, Startwert 22) mit Zahlenanzeige daneben. Rechts ein Label
   mit Live-Vorschau: ein kleines fiktives Beispielbild (240x160px,
   Gitter-Hintergrund, gezeichneter Kopf/Schulter aus Rechteck+Ellipse)
   mit einem dazu passenden unregelmässigen Silhouetten-Polygon (kein
   Rechteck/Oval) wird bei jeder Reglerbewegung/Methodenwahl live mit
   dem aktuellen Verwisch-Effekt neu gerendert und angezeigt.
5. LabelFrame "Belichtungskorrektur": Checkbox "aktivieren" + zwei
   VOLLSTÄNDIG UNABHÄNGIGE ttk.Scale-Regler nebeneinander: "Schatten
   aufhellen" (0 bis 100, Startwert 22) und "Lichter abdunkeln" (0 bis
   100, Startwert 22), jeweils mit Zahlenanzeige daneben. Kein
   Vorschaubild.
6. LabelFrame "Bildtyp": Checkbox "Bilder stammen von der Ricoh Theta Z1
   (Equirectangular / 360°)" — Startwert STANDARDMÄSSIG AKTIVIERT (True).
7. Statuszeile + ttk.Progressbar + Button "Start" (rechtsbündig unten).

Custom-Dialog ConfirmDialog(tk.Toplevel): eigener Dialog mit genau drei
Buttons "Ja" / "Nein" / "Abbrechen" (nicht der Standard-messagebox, da
dessen Beschriftungen fix sind), speichert das Ergebnis in self.result
als "yes"/"no"/"cancel", modal (grab_set, wait_window).

Beim Klick auf "Start" (on_start_clicked):
- Wenn keine Bilder in der Liste ausgewählt sind: messagebox.showinfo
  und abbrechen.
- Wenn kein Speicherort gewählt wurde: messagebox.showinfo und
  abbrechen.
- ConfirmDialog anzeigen mit Text: "Achtung. Falls im Speicherort
  bereits Dateien mit gleichem Namen vorhanden sind, werden diese
  überschrieben. Es werden {N} Bild(er) bearbeitet. Soll der Prozess
  gestartet werden?" (N = Anzahl ausgewählter Bilder).
- Nur bei "yes" fortfahren: Settings (Theta-Flag, Belichtung an/aus +
  Schatten-Intensität + Lichter-Intensität, Methode + Stärke, Quell- und
  Zielordner) sammeln, Start-Button deaktivieren, Progressbar
  zurücksetzen, Verarbeitung in einem threading.Thread(daemon=True)
  starten.

Verarbeitung (process_images, im Hintergrund-Thread):
- PersonDetector wird beim ersten Aufruf lazy geladen (nicht beim
  Programmstart, spart Startzeit).
- Für jedes ausgewählte Bild: cv2.imread; falls None -> Fehler
  protokollieren und mit dem nächsten Bild weitermachen (kein Abbruch
  der gesamten Verarbeitung). Je nach Theta-Checkbox entweder
  detect_mask_equirect() aufrufen, oder detect_masks_normal() +
  build_mask_from_polygons(). Mit count_regions() prüfen, ob überhaupt
  etwas zu verwischen ist; falls ja, blur_mask() anwenden und Zähler für
  "Bereiche insgesamt" sowie "Bilder mit Verwischung" erhöhen. Falls
  Belichtungskorrektur aktiviert ist, auto_correct_exposure() mit den
  beiden unabhängigen Reglerwerten (shadow_intensity, highlight_intensity)
  aufrufen und je nach Rückgabewert (mean_diff > 1 bzw. < -1) einen Zähler
  für "aufgehellt" bzw. "abgedunkelt" erhöhen. Zielpfad berechnen als
  output_folder + os.path.relpath(quellpfad, quellordner), Zielordner
  mit os.makedirs(..., exist_ok=True) anlegen, mit cv2.imwrite
  speichern; falls das fehlschlägt, ebenfalls als Fehler protokollieren.
  Nach jedem Bild die Fortschrittsanzeige aktualisieren.
- Alle GUI-Aktualisierungen aus dem Hintergrund-Thread ausschliesslich
  über self.after(0, lambda: ...) vornehmen (Tkinter-Thread-Sicherheit).
- Am Ende einen Zusammenfassungsdialog anzeigen mit: Anzahl verarbeiteter
  Bilder, Anzahl verwischter Bereiche insgesamt und Anzahl betroffener
  Bilder, sowie (falls Belichtungskorrektur aktiv war) Anzahl auf-/
  abgehellter Bilder. Falls Fehler aufgetreten sind, diese zusätzlich
  namentlich auflisten (messagebox.showwarning statt showinfo).

WICHTIGE EDGE CASES, die im Code behandelt sein müssen:
- Keine Auswahl / kein Speicherort -> Start wird verhindert, kein
  stiller Fehlschlag.
- Beschädigte/unlesbare Bilddatei oder fehlgeschlagenes Speichern ->
  abfangen, protokollieren, mit den übrigen Bildern weitermachen statt
  abzustürzen.
- Leere Maske (keine Person erkannt) -> Bild bleibt unverändert, kein
  Fehler.
- Gaussian-Kernelgrösse an ROI-Grösse clippen und auf ungerade Zahl
  runden (OpenCV verlangt ungerade Kernel).
- Equirectangular: Lücken aus der Rückprojektions-Streuung per
  Dilate+Close schliessen; x-Koordinaten modulo Panorama-Breite für den
  180°/-180°-Meridian-Umbruch.
- Belichtungskorrektur: Sicherheitsprüfung gegen Überbelichtung mit
  stufenweisem Rückmischen zum Original statt hartem Abbruch.
- Unterordner-Struktur des Quellordners beim Speichern im Zielordner
  nachbilden (os.path.relpath + os.makedirs).
- Nur .jpg/.jpeg/.png (case-insensitive) werden beim Scan berücksichtigt.

Gib mir den vollständigen, direkt lauffähigen Code für main.py,
detector.py, processing.py und requirements.txt.
```

---
