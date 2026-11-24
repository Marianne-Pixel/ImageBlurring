# Blueprint: Personen-Anonymisierung (Theta Z1, Fisheye & normale Bilder)

> Kompakte Referenz, um dieses Tool in einem neuen Chat identisch
> nachzubauen. Enthält Ziel, Tech-Stack, Architektur, Edge Cases und einen
> fertigen Generierungs-Prompt (Abschnitt 5).

---

## 1. Ziel & Funktion

Ein Desktop-Tool mit GUI, das Personen in Fotos automatisch erkennt und
unkenntlich macht (Anonymisierung), gedacht für drei Bildarten:

- **Equirectangular-360°-Panoramen der Ricoh Theta Z1**
- **Fisheye-Bilder** (zirkularer Bildkreis, z. B. eine Linse einer
  360°-Kamera, Kamera ggf. gedreht montiert)
- **Normale Fotos** (kaum verzerrt)

Ablauf aus Nutzersicht:
1. Quellordner wählen → wird rekursiv nach `.jpg`/`.jpeg`/`.png` durchsucht.
2. Speicherort wählen → bearbeitete Bilder werden **dorthin** geschrieben
   (Originale bleiben unverändert), mit gleicher Unterordner-Struktur wie
   im Quellordner. Alternativ Checkbox "Bilder im Quellordner ersetzen" (Standard
   aus) → Originale im Quellordner werden ersetzt.
3. Gefundene Bilder in einer Liste auswählen (Mehrfachauswahl), daneben
   Thumbnail-Vorschau des markierten Bildes.
4. Verwisch-Methode (Gaussian Blur ODER Pixelieren/Mosaik) und Stärke
   (Regler 1–30) wählen, mit Live-Vorschau an einem fiktiven Beispielbild.
5. Optional: automatische Belichtungskorrektur aktivieren, mit zwei
   unabhängigen Reglern für "Schatten aufhellen" und "Lichter abdunkeln"
   einstellen.
6. Radio-Buttons "Bildtyp": Equirectangular (Standard) / Fisheye /
   Normale Perspektive — steuert den Detektions-Modus.
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
- **Beschleunigung:** `openvino` (optional, empfohlen ohne GPU).
  Rechenwerk automatisch: CUDA-GPU mit FP16 (`quantize=16`, bei älteren
  ultralytics-Versionen `half=True`) > OpenVINO-CPU
  (Modell einmalig mit `export(format="openvino", dynamic=True,
  imgsz=1024)` umgewandelt, Ordner `<name>_openvino_model` neben der .pt
  oder in `%LOCALAPPDATA%\ImageBlurring_YOLO11`) > PyTorch-CPU; bei
  Fehlern Fallback auf PyTorch.
- Keine Web-/Cloud-Abhängigkeit zur Laufzeit.

Dateistruktur:
```
personen_anonymisierer/
├── main.py            # GUI, Ablaufsteuerung, Verarbeitung
├── detector.py         # YOLO11-Segmentierung, Equirectangular-/Fisheye-Tiling
├── processing.py        # Blur/Pixelierung, Maskenaufbau, Belichtung
├── icons.py           # Graustufen-Icons der Bildtypen
├── image_io.py          # Unicode-sicheres Bild-Lesen/-Schreiben, Metadaten, Ausrichtung
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
- `detect_mask_fisheye(img_bgr, fov_deg=FISHEYE_FOV_DEG (200),
  projection=FISHEYE_PROJECTION ("equidistant"), center=None, radius=None,
  tile_fov_deg=100, tile_size=None, rotations=FISHEYE_ROTATIONS (0, 90),
  refine_rotations=FISHEYE_REFINE_ROTATIONS (180, 270), margin_px=None)`
  → 1-Kanal-Maske:
  1. **Geometrie (`FisheyeGeometry`):** Kamerakoordinaten x rechts,
     y unten, z optische Achse. Bildkreis-Mittelpunkt = Bildmitte, Radius =
     halbe kürzere Seite (überschreibbar). Normierter Radius
     ρ = r(θ)/r(θmax) mit r(θ) = θ (equidistant), 2·sin(θ/2) (equisolid)
     oder 2·tan(θ/2) (stereographic); θmax = fov/2.
  2. **Schneiden:** 1 Tile entlang der optischen Achse + Ring aus 8 Tiles
     bei α = min(θmax − 0.6·Tile-Halbwinkel, 70°); bei sehr grossem
     Bildwinkel zweiter Ring (12 Tiles). Tile-"Oben" = Bild-Oben
     (Komponente von (0,−1,0) senkrecht zur Blickrichtung). Tile-Grösse
     automatisch: 2·tan(Tile-FOV/2)·(Pixel pro Radiant in Bildmitte), auf
     32 gerundet, begrenzt auf 640–1024. Tile per `cv2.remap` (Punkte
     ausserhalb θmax = −1 → schwarz).
  3. **Segmentieren:** jedes Tile in allen `rotations` (cv2.rotate, Batch
     über `_segment_batch`, `imgsz` = Tile-Grösse), Masken zurückdrehen
     und vereinigen; kleine Dilatation.
  4. **Zusammenfügen:** Bounding-Box der maskierten Tile-Pixel im
     Fisheye-Bild (über map_x/map_y) bestimmen, dort für jedes
     Fisheye-Pixel inverse Abbildung (Fisheye → Strahl → Tile-Pixel)
     berechnen und die Tile-Maske per `cv2.remap` (INTER_NEAREST)
     abtasten → dicht, keine Lücken. Ausserhalb des Bildkreises nullen,
     leichtes Close (3×3).
- `detect_masks_normal(img_bgr, imgsz=None)` und `_segment_batch(images,
  imgsz=None)` (mehrere Bilder pro `model.predict`-Aufruf).

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
- `blur_mask(img_bgr, mask, method, strength, relative_strength=True)` —
  bearbeitet jeden zusammenhängenden Maskenbereich einzeln
  (`connectedComponentsWithStats`) in einem um Kernel-Radius + Feather
  erweiterten Ausschnitt und blendet den Effekt über eine weichgezeichnete
  (gefeatherte) Version der Maske ein (kein harter Rand).
- `effective_strength(strength, region_size, relative)` — skaliert die
  Stärke mit der Bereichsgrösse (`max(Breite, Höhe)`):
  `max(strength, strength · size / 128)`. Sehr grosse Gaussian-Kernel
  (> 61) werden auf einer verkleinerten Kopie berechnet (Performance).
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
   `StringVar`). Darunter Zeile mit `ttk.Checkbutton` "Bilder
   überschreiben" (`BooleanVar overwrite`, Standard `False`) und grauem
   Hinweis-Label; bei aktivierter Checkbox Entry und Button `disabled`.
3. Content-Bereich (nebeneinander):
   - `LabelFrame` mit Titel "Gefundene Bilder: ##" (Anzahl nach jedem
     Scan aktualisiert), darin oben eine Zeile "Sortieren nach:" mit
     readonly `ttk.Combobox` ("Name (A–Z)" Standard, "Bildgrösse (klein →
     gross)", "Bildgrösse (gross → klein)"; Grösse = Breite × Höhe in
     Pixeln, bei Gleichstand nach Name, unlesbare Dateien am Ende; Auswahl
     bleibt beim Umsortieren erhalten). Listeneinträge "relpfad   (B × H)",
     Grösse per `PIL.Image.open(...).size` (nur Dateikopf, EXIF-Drehung
     5–8 vertauscht B/H). Darunter `Listbox` (`selectmode="extended"`) +
     Scrollbar + Buttons "Alle auswählen"/"Auswahl aufheben";
     `<<ListboxSelect>>`-Binding aktualisiert die Thumbnail-Vorschau
     rechts und die Statuszeile unten ("## Bilder ausgewählt", bei 1
     "1 Bild ausgewählt"; nicht während der Verarbeitung).
   - `LabelFrame` "Vorschau": `ttk.Label` mit `PIL.Image.open(path).thumbnail((190,150))`
     → `ImageTk.PhotoImage`.
4. `LabelFrame` "Verwisch-Einstellungen": Radiobuttons Methode
   (`gaussian`/`pixelate`), `ttk.Scale` Stärke (1–30, Default **22**),
   Live-Vorschau-Label rechts (darunter die Beschriftung "Vorschau", kein
   weiterer Erklärtext), aktualisiert bei jeder Reglerbewegung über
   ein fiktives Beispielbild (`_create_fictive_sample()`: 240×160 Bild nur in
   Grauwerten mit Gitter-Hintergrund + gezeichneter Kopf/Schulter-Form + zugehöriges
   unregelmässiges Silhouetten-Polygon).
5. `LabelFrame` "Belichtungskorrektur": Checkbox aktivieren + zwei
   **unabhängige** `ttk.Scale`-Regler: "Schatten aufhellen" (0–100,
   Default **22**) und "Lichter abdunkeln" (0–100, Default **22**). Kein
   Vorschaubild.
6. `LabelFrame` "Bildtyp": drei `ttk.Radiobutton` nebeneinander
   (`StringVar image_type`): "Equirectangular" (`"equirect"`, Standard),
   "Fisheye" (`"fisheye"`), "Normale Perspektive" (`"normal"`), jeweils
   mit Icon links vom Text (`image=..., compound="left"`). Icons aus
   `icons.projection_icons(48)`: Innenansicht eines Würfels mit Gitter
   (4×4 Felder pro Fläche, Grauwert je Fläche, dunkle Kanten), pro Pixel
   Blickrichtung → Würfelfläche, 4× Supersampling, abgebildet als
   Equirectangular (96×48), Fisheye 180° equidistant (48×48, rund,
   ausserhalb transparent) und Perspektive 125° (48×48).
7. `LabelFrame` "Metadaten (nur JPEG)": Checkbox "Metadaten übernehmen"
   (Standard an) und eingerückt "GPS-Position entfernen" (Standard aus,
   deaktiviert, wenn keine Metadaten übernommen werden).
**Scrollbarer Aufbau:** Punkte 1–7 liegen in einem `ScrollableFrame`
(`tk.Canvas` + `create_window` mit innerem `ttk.Frame body`, senkrechter
und waagrechter `ttk.Scrollbar`, nur bei Bedarf per `grid`/`grid_remove`
eingeblendet). Bedarf aus dem Platz des äusseren Frames berechnet (zwei
Durchgänge, da ein Balken dem anderen Platz nimmt; kein Flackern), bei
genug Platz `body` auf die volle Fläche strecken (Bildliste mit
`expand=True` wächst mit). Neuberechnung per `after_idle` bei
`<Configure>` von Frame und `body`. Mausrad über `bind_all`
(`<MouseWheel>`, `<Button-4/5>`, mit Shift waagrecht), ignoriert über
`Listbox`/`TCombobox`/`Text`, in anderen Fenstern und ausserhalb des
Bereichs. Punkt 8 ist eine feste Fusszeile (`pack(side="bottom")` vor dem
Scrollbereich, darüber `ttk.Separator`). Startgrösse
`min(840, Bildschirmbreite − 40) × min(1020, Bildschirmhöhe − 100)`,
`minsize(420, 300)`.

8. Statuszeile + `ttk.Progressbar` + Button "Start" (rechtsbündig), per
   `grid`, als feste Fusszeile; darunter (Spalte des Balkens) ein Label mit Zeitabschätzung:
   "Verstrichen: 2 min 14 sec · Restzeit: ca. 6 min 40 sec · fertig um ca. HH:MM Uhr"
   (Zeiten mit Einheiten: "45 sec", "2 min 14 sec", "1 h 02 min 05 sec")
   (Restzeit = Ø Dauer der fertigen Bilder × verbleibende Bilder, minus
   Zeit seit dem letzten fertigen Bild; Zeitmessung startet erst nach dem
   Laden des Modells; sekündliche Aktualisierung per `self.after(1000)`
   im GUI-Thread; am Ende "Gesamtdauer: … (ø … pro Bild)").

Im Bereich "Verwisch-Einstellungen" zusätzlich Checkbox "Stärke an
Personengrösse anpassen (empfohlen)" (Standard an).

**Custom-Dialog `ConfirmDialog(tk.Toplevel)`** — genau 3 Buttons "Ja" /
"Nein" / "Abbrechen" (kein Standard-`messagebox`, da dessen Labels nicht
frei benennbar sind), `result` ∈ `{"yes","no","cancel"}`.

**Start-Ablauf (`on_start_clicked`):**
1. Keine Auswahl → `messagebox.showinfo`, Abbruch.
2. Nur wenn "Bilder im Quellordner ersetzen" aus: kein Speicherort gewählt →
   `messagebox.showinfo`, Abbruch; Speicherort = Quellordner
   (`os.path.samefile`, Fallback `normcase(abspath)`) → Hinweis, Abbruch.
3. `ConfirmDialog` mit Text: *"Achtung. Falls im Speicherort bereits
   Dateien mit gleichem Namen vorhanden sind, werden diese überschrieben.
   Es werden {N} Bild(er) bearbeitet. Soll der Prozess gestartet werden?"*
   bzw. beim Überschreiben: *"Achtung. Die ausgewählten Originalbilder im
   Quellordner werden durch die bearbeiteten Bilder ersetzt. Das kann
   nicht rückgängig gemacht werden. Es werden {N} Bild(er) bearbeitet.
   Soll der Prozess gestartet werden?"*
4. Bei "yes": `settings`-Dict sammeln (Bildtyp, Belichtung an/aus +
   Schatten-/Lichter-Intensität, Methode + Stärke, Quell-/Zielordner,
   Überschreiben an/aus) →
   Verarbeitung in `threading.Thread(..., daemon=True)`.

**`process_images(paths, settings)`** (läuft im Hintergrund-Thread):
- Modell wird lazy beim ersten Aufruf geladen (`PersonDetector(...)`).
- Pro Bild: `cv2.imread` → je nach Bildtyp `detect_mask_equirect()`,
  `detect_mask_fisheye()` oder `detect_masks_normal()` +
  `build_mask_from_polygons()` →
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
- **Überschreiben / sicheres Speichern (`save_image_safely`):** immer
  zuerst `<name>.anonym-tmp<endung>` im Zielordner schreiben (gleiche
  Endung → richtiges Format), dann `os.replace` auf die Zieldatei; im
  `finally` eine verbliebene temporäre Datei löschen. Bei Fehler bleibt das
  Original/Ziel unversehrt; `os.replace`-Fehler (schreibgeschützt/geöffnet)
  als verständliche Meldung. Metadaten vor dem Ersetzen aus dem Original
  lesen. Dateien mit `.anonym-tmp` im Namen beim Scan ignorieren.
- **Überschreiben ohne Änderung:** keine Person, keine
  Belichtungskorrektur, Metadaten übernehmen an, GPS entfernen aus →
  Original nicht neu speichern (kein JPEG-Qualitätsverlust), als
  "unverändert" zählen. Fortschritt im `finally` aktualisieren, damit auch
  übersprungene Bilder zählen.
- **Speicherort = Quellordner im Kopiermodus:** Start verhindern (würde
  Originale überschreiben, ohne dass "Bilder im Quellordner ersetzen" aktiv ist).
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
- **Fisheye-Kamera gedreht montiert** (Personen liegen quer im Bild,
  z. B. im Beispielbild um 90°): YOLO erkennt liegende Personen schlecht
  → alle Tiles in 0°/90° segmentieren, Tiles mit einem Kandidaten ab
  Konfidenz 0.2 zusätzlich in 180°/270°, alles vereinigen.
- **Fisheye: ungenauer Bildwinkel:** Vorwärts- und Rückprojektion nutzen
  dasselbe Modell → Masken bleiben deckungsgleich, nur die Tiles sind
  leicht verzerrt.
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
- **Umlaute/Nicht-ASCII-Zeichen in Pfaden (Windows):** `cv2.imread`/
  `cv2.imwrite` scheitern daran still → stattdessen `image_io.read_image()`
  (`np.fromfile` + `cv2.imdecode`) und `image_io.write_image()`
  (`cv2.imencode` + `tofile`).
- **Metadaten (`image_io.read_metadata` / `write_image_with_meta`):**
  `cv2.imencode` schreibt kein EXIF/XMP → ohne Übernahme fehlen die
  GPano-Tags und 360°-Viewer zeigen das Panorama flach. Übernommen werden
  APP1 EXIF, APP1 XMP und APP2 ICC (direkt nach SOI, JFIF-APP0 von OpenCV
  entfernt). EXIF wird "in place" bearbeitet (MakerNote-Offsets bleiben
  gültig): IFD1-Vorschaubild genullt und ausgehängt (würde Personen
  unverwischt zeigen), Orientation = übergebener Wert (siehe
  Bildausrichtung), PixelX/YDimension = gespeicherte Abmessungen, optional
  GPS-IFD geleert und Werte genullt. XMP: `xmp:Thumbnails` und
  `HasExtendedXMP` entfernt, `tiff:Orientation` = gleicher Wert, optional
  `exif:GPS*`. Extended XMP, APP13 und
  MPF werden nie übernommen. Ist EXIF nicht parsebar, wird es weggelassen
  (Warnung in der Zusammenfassung).
- **Bildausrichtung (Hochformat, RiSCAN):** Kameras speichern Hochformat
  quer + EXIF-Orientation; RiSCAN nutzt die gespeicherten Pixel. Daher nie
  die Geometrie ändern: `read_image_with_orientation()` liest JPEG mit
  `cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION` und liefert das Tag
  (1–8, aus dem EXIF gelesen; EXIF unlesbar → Pixel wie angezeigt,
  Orientation 1). Erkennung auf `to_upright(img, o)`, Maske mit
  `from_upright(mask, o)` zurück (6↔8 vertauscht, übrige selbstinvers),
  Verwischen/Speichern auf den ungedrehten Pixeln,
  `write_image_with_meta(..., orientation=o)`. Ohne übernommenes EXIF und
  o ≠ 1: minimales EXIF nur mit Orientation. Vorschaubild mit
  `ImageOps.exif_transpose`.
- **Nahe/grosse Personen zu schwach verwischt:** Stärke pro Bereich an
  dessen Grösse anpassen (siehe `effective_strength`).
- **Nur unterstützte Formate:** Groschreibung wird ignoriert
  (`f.lower().endswith((".jpg", ".jpeg", ".png"))`), andere Formate
  werden beim Scan ignoriert.

---

## 5. Finaler Generierungs-Prompt

> Alles zwischen den Trennlinien direkt in einen neuen Chat kopieren.

---

```
Erstelle mir ein Python-Tool zum Unkenntlichmachen (Anonymisieren) von
Personen in Bildern, aufgeteilt in main.py, detector.py, processing.py,
image_io.py, icons.py und requirements.txt.

ZIEL: Ein Desktop-GUI-Tool (tkinter), das Personen in Fotos automatisch
per YOLO11-Segmentierung erkennt und ihre exakte Silhouette (kein
Rechteck/Oval) mit Gaussian Blur oder Pixelierung verwischt. Unterstützt
drei Bildarten: Equirectangular-360°-Panoramen der Ricoh Theta Z1,
Fisheye-Bilder (zirkularer Bildkreis, Kamera evtl. gedreht montiert) und
normale Fotos.

TECH-STACK: Python 3.9+, tkinter/ttk (GUI), opencv-python, numpy,
Pillow (für GUI-Bildvorschauen), ultralytics mit YOLO11-
Segmentierungsmodell "yolo11s-seg.pt" (COCO-Klasse "person", ID 0),
optional openvino (schnellere CPU-Inferenz).

--- detector.py: Klasse PersonDetector ---
- __init__(model_path="yolo11s-seg.pt", conf_threshold=0.35, device=None)
- detect_masks_normal(img_bgr, imgsz=None): läuft model.predict(img,
  conf=conf_threshold, classes=[0], verbose=False, ggf. imgsz) und gibt eine Liste von Polygonen
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
- Konstanten oben in detector.py: FISHEYE_FOV_DEG = 200,
  FISHEYE_PROJECTION = "equidistant", FISHEYE_ROTATIONS = (0, 90),
  FISHEYE_REFINE_ROTATIONS = (180, 270), FISHEYE_REFINE_TRIGGER_CONF = 0.2,
  BATCH_SIZE_GPU = 8, BATCH_SIZE_CPU = 4, DEFAULT_IMGSZ = 640.
- PersonDetector(model_path, conf_threshold=0.35, device=None,
  backend="auto"): wählt das Rechenwerk (CUDA-GPU mit FP16 >
  OpenVINO auf der CPU > PyTorch-CPU). OpenVINO: Ordner
  "<name>_openvino_model" neben der .pt oder im Cache
  (%LOCALAPPDATA%/ImageBlurring_YOLO11) laden, sonst die .pt dorthin
  kopieren und einmalig export(format="openvino", dynamic=True,
  imgsz=1024) ausführen; bei jedem Fehler Fallback auf PyTorch.
  FP16 über predict(quantize=16), falls "quantize" in
  ultralytics.cfg.DEFAULT_CFG_DICT vorhanden ist, sonst half=True (ältere
  Versionen; neuere melden bei half eine Veraltet-Warnung).
  Attribut backend_name für die Anzeige in der Zusammenfassung.
- _segment_batch(images, imgsz=None, min_conf=None): alle Bilder in
  Batches (batch_size) an model.predict, imgsz IMMER explizit setzen
  (Standard 640; ein OpenVINO-Modell hätte sonst 1024), Rückgabe pro Bild
  (Polygone mit Konfidenz >= conf_threshold, höchste Konfidenz aller
  Kandidaten >= min_conf). Equirectangular: alle 16 Tiles in einem
  Aufruf. Abbildungs-Tabellen (map_x/map_y) pro Bildgrösse
  zwischenspeichern.
- detect_mask_fisheye(img_bgr, fov_deg, projection, center=None,
  radius=None, tile_fov_deg=100, tile_size=None, rotations, margin_px=None):
  Workflow a) schneiden, b) segmentieren, c) zusammenfügen:
  a) Klasse FisheyeGeometry (x rechts, y unten, z optische Achse;
     Bildkreis-Mittelpunkt = Bildmitte, Radius = halbe kürzere Seite;
     Abbildung r(θ) = θ / 2sin(θ/2) / 2tan(θ/2) für equidistant /
     equisolid / stereographic, normiert auf r(θmax), θmax = fov/2).
     Tiles (je 100°): eines entlang der optischen Achse plus ein Ring aus
     8 Tiles bei α = min(θmax - 0.6*50°, 70°) (bei sehr grossem Bildwinkel
     zweiter Ring mit 12 Tiles). Tile-Oben = Bild-Oben. Tile-Grösse =
     2*tan(50°)*(Pixel pro Radiant in Bildmitte), auf 32 gerundet,
     begrenzt auf 640..1024. Tile per cv2.remap aus dem Fisheye holen
     (ausserhalb des Bildkreises schwarz).
  b) Zweistufig segmentieren (imgsz = Tile-Grösse, gebündelt): Stufe 1
     alle Tiles in 0° und 90° (cv2.rotate); Stufe 2 nur Tiles mit einem
     Kandidaten ab Konfidenz 0.2 zusätzlich in 180° und 270°. Masken
     zurückdrehen und vereinigen (Kamera kann gedreht montiert sein),
     kleine Dilatation.
  c) Für den Bereich (Bounding-Box) der maskierten Tile-Pixel im
     Fisheye-Bild die inverse Abbildung berechnen (Fisheye-Pixel ->
     Strahl -> Tile-Pixel) und die Tile-Maske per cv2.remap
     (INTER_NEAREST) abtasten; alle Tiles per Maximum vereinigen,
     ausserhalb des Bildkreises nullen, leichtes Close (3x3).

--- processing.py ---
- apply_blur_effect(roi, method="gaussian", strength=12): bei "gaussian"
  cv2.GaussianBlur mit Kernel k=2*strength+1 (ungerade, auf ROI-Grösse
  geclippt); bei "pixelate" Downscale auf Breite/Höhe // strength
  (INTER_LINEAR) und Upscale zurück (INTER_NEAREST).
- build_mask_from_polygons(img_shape, polygons, margin_px=None): füllt
  alle Polygone in eine leere Maske (cv2.fillPoly), dilatiert um einen
  sehr kleinen festen Pixel-Rand (Default max(2, 0.004*min(h,w))), damit
  möglichst wenig Hintergrund verwischt wird.
- effective_strength(strength, region_size, relative=True): gibt
  max(strength, strength * region_size / 128) zurück (bzw. strength, wenn
  relative=False). 128 px entspricht etwa der Figur in der GUI-Vorschau.
- blur_mask(img_bgr, mask, method="gaussian", strength=18,
  relative_strength=True): bearbeitet jeden zusammenhängenden Bereich der
  Maske einzeln (cv2.connectedComponentsWithStats). Pro Bereich: Stärke
  über effective_strength(strength, max(Breite, Höhe)) bestimmen,
  Ausschnitt um Kernel-Radius + Feather erweitern, Effekt im Ausschnitt
  berechnen und über die weichgezeichnete (gefeatherte) Bereichsmaske
  einblenden, damit kein harter Rand entsteht. Gibt bei leerer Maske das
  unveränderte Bild zurück.
- In apply_blur_effect: Gaussian-Kernel > 61 auf einer verkleinerten Kopie
  berechnen (resize INTER_AREA, Blur mit k=61, resize zurück), sonst wird
  es bei grossen Panoramen sehr langsam.
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
Fenstertitel "Personen-Anonymisierung YOLO11 – Theta Z1 / Fisheye / Normale Bilder".
Layout von oben nach unten:
1. Zeile "Ordner:" mit readonly Entry + Button "Durchsuchen…"
   (filedialog.askdirectory) -> durchsucht den Ordner rekursiv
   (os.walk) nach .jpg/.jpeg/.png (case-insensitive) und befüllt eine
   Listbox mit den relativen Pfaden.
2. Zeile "Speicherort:" mit eigenem readonly Entry + Button
   "Durchsuchen…" -> separater Ausgabeordner, in den die bearbeiteten
   Bilder gespeichert werden (Originale bleiben unverändert).
2b. Darunter Checkbox "Bilder im Quellordner ersetzen" (Startwert False) mit grauem
   Hinweistext; aktiviert -> Speicherort-Entry und -Button ausgegraut,
   die Originalbilder im Quellordner werden ersetzt.
3. Content-Bereich nebeneinander: links ein LabelFrame mit dem Titel
   "Gefundene Bilder: ##" (Anzahl gefundener Bilder, nach jedem Scan
   aktualisiert), darin oben "Sortieren nach:" mit readonly Combobox
   ("Name (A–Z)" Standard, "Bildgrösse (klein → gross)", "Bildgrösse
   (gross → klein)"; Grösse = Breite × Höhe in Pixeln aus dem Dateikopf
   per PIL, EXIF-Drehung 5–8 vertauscht B/H, bei Gleichstand nach Name,
   unlesbare Dateien am Ende; Auswahl bleibt beim Umsortieren erhalten;
   Einträge "relpfad   (B × H)"), darunter Listbox (selectmode="extended")
   mit Scrollbar und Buttons "Alle auswählen"/"Auswahl aufheben"; rechts
   ein LabelFrame "Vorschau" mit einem Label, das bei Auswahl genau
   eines Listeneintrags per PIL (Image.open(path).thumbnail((190,150)))
   ein Thumbnail anzeigt (bei 0 Auswahl "Keine Auswahl", bei >1 Auswahl
   die Anzahl als Text).
4. LabelFrame "Verwisch-Einstellungen": Radiobuttons für Methode
   ("Gaussian Blur" / "Pixelieren (Mosaik)"), ttk.Scale für Stärke
   (1 bis 30, Startwert 22) mit Zahlenanzeige daneben. Rechts ein Label
   mit Live-Vorschau, darunter ein kleines graues Label "Vorschau" (kein
   Erklärtext links): ein kleines fiktives Beispielbild (240x160px,
   nur Grauwerte, Gitter-Hintergrund, gezeichneter Kopf/Schulter aus Rechteck+Ellipse)
   mit einem dazu passenden unregelmässigen Silhouetten-Polygon (kein
   Rechteck/Oval) wird bei jeder Reglerbewegung/Methodenwahl live mit
   dem aktuellen Verwisch-Effekt neu gerendert und angezeigt.
5. LabelFrame "Belichtungskorrektur": Checkbox "aktivieren" + zwei
   VOLLSTÄNDIG UNABHÄNGIGE ttk.Scale-Regler nebeneinander: "Schatten
   aufhellen" (0 bis 100, Startwert 22) und "Lichter abdunkeln" (0 bis
   100, Startwert 22), jeweils mit Zahlenanzeige daneben. Kein
   Vorschaubild.
6. LabelFrame "Bildtyp": drei Radio-Buttons nebeneinander (nur eine Option
   aktiv): "Equirectangular" (Startwert), "Fisheye", "Normale
   Perspektive", jeweils mit einem schematischen Graustufen-Icon links
   vom Text. Icons in icons.py zur Laufzeit berechnen (keine
   Bilddateien): Innenansicht eines Würfelraums mit Gitter auf allen
   Wänden (4x4 Felder, je Fläche ein anderer Grauwert, dunkle Kanten),
   für jedes Pixel Blickrichtung bestimmen und die getroffene
   Würfelfläche einfärben, 4x Supersampling; Projektionen:
   Equirectangular 96x48, Fisheye 180° equidistant 48x48 (rund,
   ausserhalb transparent), Perspektive 125° 48x48.
7. LabelFrame "Metadaten (nur JPEG)": Checkbox "Metadaten übernehmen (EXIF,
   XMP inkl. 360°-Angaben, Farbprofil)" (Startwert True) und eingerückt
   "GPS-Position entfernen" (Startwert False, ausgegraut, wenn keine
   Metadaten übernommen werden).
Punkte 1–7 in einem scrollbaren Bereich (Klasse ScrollableFrame: Canvas
mit innerem Frame, senkrechter und waagrechter Scrollbalken nur bei
Bedarf; bei genug Platz inneren Frame auf volle Fläche strecken, damit die
Bildliste mitwächst; Bedarf aus dem Platz des äusseren Frames berechnen,
damit nichts flackert; Mausrad per bind_all inkl. Linux Button-4/5 und
Shift = waagrecht, aber nicht über Listbox/Combobox, in anderen Fenstern
oder über der Fusszeile). Startgrösse an den Bildschirm anpassen (max.
840x1020), minsize(420, 300).
8. Statuszeile + ttk.Progressbar + Button "Start" (rechtsbündig unten)
   als feste Fusszeile (vor dem Scrollbereich mit side="bottom" packen),
   immer sichtbar.
   Die Statuszeile zeigt ausserhalb der Verarbeitung die Anzahl
   ausgewählter Bilder ("## Bilder ausgewählt", bei 1 "1 Bild
   ausgewählt"), aktualisiert bei jeder Auswahländerung; während der
   Verarbeitung den Fortschritt ("Bearbeite i/N: datei").
   Direkt unter dem Balken ein Label mit Zeitabschätzung:
   "Verstrichen: 2 min 14 sec · Restzeit: ca. 6 min 40 sec · fertig um ca. HH:MM Uhr"
   (Zeiten mit Einheiten: "45 sec", "2 min 14 sec", "1 h 02 min 05 sec").
   Restzeit = (Zeit bis zum letzten fertigen Bild / Anzahl fertiger
   Bilder) × verbleibende Bilder − Zeit seit dem letzten fertigen Bild.
   Zeitmessung erst nach dem Laden des Modells starten. Vor dem ersten
   fertigen Bild "Restzeit: wird nach dem ersten Bild berechnet…".
   Sekündlich per self.after(1000, ...) im GUI-Thread aktualisieren; der
   Worker-Thread meldet nur "Bild fertig" (mit Zeitstempel) über
   self.after(0, ...). Am Ende "Gesamtdauer: … (ø … pro Bild)"; Dauer auch
   in der Abschluss-Zusammenfassung.
Im LabelFrame "Verwisch-Einstellungen" zusätzlich eine Checkbox "Stärke an
Personengrösse anpassen (empfohlen)" (Startwert True).

Custom-Dialog ConfirmDialog(tk.Toplevel): eigener Dialog mit genau drei
Buttons "Ja" / "Nein" / "Abbrechen" (nicht der Standard-messagebox, da
dessen Beschriftungen fix sind), speichert das Ergebnis in self.result
als "yes"/"no"/"cancel", modal (grab_set, wait_window).

Beim Klick auf "Start" (on_start_clicked):
- Wenn keine Bilder in der Liste ausgewählt sind: messagebox.showinfo
  und abbrechen.
- Nur wenn "Bilder im Quellordner ersetzen" aus ist: Wenn kein Speicherort gewählt
  wurde, messagebox.showinfo und abbrechen; wenn Speicherort = Quellordner,
  Hinweis und abbrechen (sonst würden Originale überschrieben).
- Bei "Bilder im Quellordner ersetzen" eigener Dialogtext: "Achtung. Die
  ausgewählten Originalbilder im Quellordner werden durch die bearbeiteten
  Bilder ersetzt. Das kann nicht rückgängig gemacht werden. Es werden {N}
  Bild(er) bearbeitet. Soll der Prozess gestartet werden?"
- ConfirmDialog anzeigen mit Text: "Achtung. Falls im Speicherort
  bereits Dateien mit gleichem Namen vorhanden sind, werden diese
  überschrieben. Es werden {N} Bild(er) bearbeitet. Soll der Prozess
  gestartet werden?" (N = Anzahl ausgewählter Bilder).
- Nur bei "yes" fortfahren: Settings (Bildtyp, Belichtung an/aus +
  Schatten-Intensität + Lichter-Intensität, Methode + Stärke, Quell- und
  Zielordner, Überschreiben an/aus) sammeln, Start-Button deaktivieren, Progressbar
  zurücksetzen, Verarbeitung in einem threading.Thread(daemon=True)
  starten.

Verarbeitung (process_images, im Hintergrund-Thread):
- PersonDetector wird beim ersten Aufruf lazy geladen (nicht beim
  Programmstart, spart Startzeit).
- Für jedes ausgewählte Bild: cv2.imread; falls None -> Fehler
  protokollieren und mit dem nächsten Bild weitermachen (kein Abbruch
  der gesamten Verarbeitung). Je nach Bildtyp detect_mask_equirect(),
  detect_mask_fisheye() oder detect_masks_normal() +
  build_mask_from_polygons() aufrufen. Mit count_regions() prüfen, ob überhaupt
  etwas zu verwischen ist; falls ja, blur_mask() anwenden und Zähler für
  "Bereiche insgesamt" sowie "Bilder mit Verwischung" erhöhen. Falls
  Belichtungskorrektur aktiviert ist, auto_correct_exposure() mit den
  beiden unabhängigen Reglerwerten (shadow_intensity, highlight_intensity)
  aufrufen und je nach Rückgabewert (mean_diff > 1 bzw. < -1) einen Zähler
  für "aufgehellt" bzw. "abgedunkelt" erhöhen. Zielpfad: beim
  Überschreiben der Quellpfad selbst (Bild ohne Person und ohne
  Belichtungs-/Metadaten-Änderung überspringen), sonst output_folder +
  os.path.relpath(quellpfad, quellordner) und Zielordner mit
  os.makedirs(..., exist_ok=True) anlegen. Speichern über
  save_image_safely(dest, img, read_metadata(quelle) oder None,
  strip_gps): image_io.write_image_with_meta in die temporäre Datei,
  dann os.replace; falls das fehlschlägt, als Fehler protokollieren
  (Metadaten-Warnungen ebenfalls in der Fehlerliste).
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
- Sicheres Speichern (beide Modi): in "<name>.anonym-tmp<endung>" im
  Zielordner schreiben, erst danach os.replace auf die Zieldatei; bei
  Fehler temporäre Datei löschen, Original bleibt unversehrt. Solche
  temporären Dateien beim Scan ignorieren.
- Beim Überschreiben Bilder ohne erkannte Person (und ohne Belichtungs-/
  Metadaten-Änderung) nicht neu speichern; Zusammenfassung nennt Anzahl
  ersetzter und unveränderter Bilder.
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
- Bilder NIE direkt mit cv2.imread/cv2.imwrite lesen/schreiben (scheitern
  unter Windows an Umlauten im Pfad), sondern über image_io.py:
  read_image(path) = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR),
  write_image(path, img) = cv2.imencode(ext, img) + buf.tofile(path).
- Bildgeometrie NIE verändern (Hochformat-Fotos sind oft quer gespeichert
  mit EXIF-Orientation; RiSCAN verwendet die gespeicherten Pixel):
  image_io.read_image_with_orientation(path) -> (img, orientation) liest
  JPEG mit IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION und das Tag aus dem
  EXIF (EXIF unlesbar oder PNG -> wie angezeigt, orientation 1);
  to_upright(img, o) / from_upright(img, o) für die Erkennung (Maske
  zurückdrehen); write_image_with_meta(..., orientation=o) schreibt o in
  EXIF und XMP (Standard 1), ohne EXIF und o != 1 ein minimales EXIF mit
  nur der Orientation.
- image_io.py zusätzlich: read_metadata(path) liest bei JPEG die Segmente
  APP1 EXIF, APP1 XMP und APP2 ICC; write_image_with_meta(path, img, meta,
  strip_gps) codiert JPEG mit Qualität 95, entfernt das JFIF-APP0 von
  OpenCV und setzt die Segmente direkt nach SOI ein. EXIF in place
  bearbeiten (MakerNote bleibt gültig): Vorschaubild (IFD1) nullen und
  aushängen, Orientation auf den übergebenen Wert, PixelX/YDimension
  auf die gespeicherte Grösse,
  bei strip_gps die GPS-IFD leeren und ihre Werte nullen. XMP:
  xmp:Thumbnails und xmpNote:HasExtendedXMP entfernen, tiff:Orientation
  auf denselben Wert, bei strip_gps alle exif:GPS*-Angaben entfernen. Extended XMP,
  APP13 und MPF nie übernehmen. Wenn EXIF nicht parsebar ist: weglassen
  und Warnung zurückgeben.

Gib mir den vollständigen, direkt lauffähigen Code für main.py,
detector.py, processing.py, image_io.py, icons.py und requirements.txt.
```

---
