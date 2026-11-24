# Anleitung: Personen-Anonymisierung (Theta Z1, Fisheye & normale Bilder)

> Diese Anleitung wird parallel zum Code aktuell gehalten. Stand: passend zu
> `main.py`, `detector.py`, `processing.py` (siehe Projektordner).

## Inhalt

1. [Überblick](#überblick)
2. [Voraussetzungen](#voraussetzungen)
3. [Installation](#installation)
4. [Programm starten](#programm-starten)
5. [Bedienung Schritt für Schritt](#bedienung-schritt-für-schritt)
6. [Funktionsweise im Detail](#funktionsweise-im-detail)
7. [Anpassbare Einstellungen](#anpassbare-einstellungen)
8. [Häufige Probleme (FAQ)](#häufige-probleme-faq)
9. [Grenzen des Tools](#grenzen-des-tools)
10. [Projektstruktur](#projektstruktur)
11. [Änderungsverlauf](#änderungsverlauf)

---

## Überblick

Das Tool erkennt Personen in Bildern automatisch mit YOLO11-**Segmentierung**
(nicht nur Bounding-Box-Detektion) und macht sie mittels Gaussian Blur oder
Pixelierung unkenntlich – exakt entlang ihrer Silhouette (unregelmässige
Form), nicht als Rechteck oder Oval. Es unterstützt drei Bildarten:

| Bildart | Radio-Button "Bildtyp" | Vorgehen |
|---|---|---|
| Equirectangular-Panoramen (z. B. Ricoh Theta Z1, 360°) | **Equirectangular (Standard)** | Bild wird in Tiles zerlegt, Segmentierung pro Tile, Rückprojektion der Silhouetten ins Panorama |
| Fisheye-Bilder (zirkularer Bildkreis, z. B. eine Linse einer 360°-Kamera) | Fisheye | Bildkreis wird in entzerrte Tiles zerlegt, Segmentierung pro Tile in mehreren Drehungen, Rückprojektion der Silhouetten ins Fisheye-Bild |
| Normale Fotos (kaum verzerrt) | Normale Perspektive | Direkte Segmentierung im Bild |

Zusätzlich kann optional eine automatische Belichtungskorrektur (Schatten
aufhellen, Lichter abdunkeln) angewendet werden, deren beide Effekte über
zwei **unabhängige** Regler ("Schatten aufhellen" / "Lichter abdunkeln")
separat gesteuert werden. Am Ende der Verarbeitung zeigt das Tool eine
Zusammenfassung mit der Anzahl verwischter Bereiche/Bilder sowie der
Anzahl auf- bzw. abgehellter Bilder.

**Wichtig:** Standardmässig werden die Originaldateien **nicht verändert**.
Die bearbeiteten Bilder werden unter einem separat wählbaren Speicherort
abgelegt (mit derselben relativen Unterordner-Struktur wie im Quellordner).
Sind dort bereits Dateien mit gleichem Namen vorhanden, werden diese
überschrieben — vor dem Start erscheint deshalb eine Sicherheitsabfrage.
Mit der Checkbox **"Bilder im Quellordner ersetzen"** werden stattdessen die
Originalbilder im Quellordner ersetzt.

---

## Voraussetzungen

- Python 3.9 oder neuer

- Optional: NVIDIA-GPU mit CUDA für deutlich schnellere Verarbeitung
  (funktioniert aber auch ohne GPU, nur langsamer)
- Empfohlen ohne GPU: Paket `openvino` (in `requirements.txt` enthalten),
  macht die Segmentierung auf der CPU etwa 2.5-mal schneller

## Installation

```bash
cd personen_anonymisierer
pip install -r requirements.txt
```

Enthaltene Abhängigkeiten (siehe `requirements.txt`):

- `ultralytics` – YOLO11-**Segmentierungs**modell und Inferenz
- `opencv-python` – Bildverarbeitung (Gaussian Blur, Pixelierung,
  Belichtungskorrektur, Tile-Projektion)
- `numpy` – numerische Berechnungen
- `Pillow` – Laden/Skalieren der Bild- und Verwisch-Vorschauen im GUI
  (u. a. JPEG-Unterstützung, die reines Tkinter nicht mitbringt)
- `openvino` – optional, schnellere Segmentierung auf der CPU

### NVIDIA-GPU nutzen

`pip install -r requirements.txt` installiert unter Windows die
**CPU-Version** von PyTorch. Für eine NVIDIA-GPU (z. B. RTX 2080 Ti) danach
die CUDA-Version installieren:

```bash
pip uninstall -y torch torchvision
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
```

Wichtig: `--index-url` verwenden, nicht `--extra-index-url` (sonst kann pip
wieder die CPU-Version nehmen). Prüfen:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Erwartet: `…+cu126 True`. Ein späteres `pip install -U ultralytics` kann
PyTorch wieder durch die CPU-Version ersetzen, danach erneut prüfen.

## Programm starten

```bash
python main.py
```

Es öffnet sich das GUI-Fenster "Personen-Anonymisierung YOLO11 – Theta Z1 / Fisheye / Normale
Bilder".

### Exe erstellen (PyInstaller)

```bash
pip install pyinstaller
pyinstaller ImageBlurring_YOLO11.spec
```

Ergebnis ist der **Ordner** `dist\ImageBlurring_YOLO11\` mit
`ImageBlurring_YOLO11.exe` und dem Unterordner `_internal`. Weitergegeben
wird immer der ganze Ordner (z. B. gezippt), die .exe allein läuft nicht.

- **Ordner statt Einzeldatei:** Mit CUDA ist PyTorch mehrere GB gross. Eine
  Einzeldatei-Exe müsste das bei jedem Start in einen Temp-Ordner
  entpacken. Der Ordner-Build startet direkt.
- **Ohne UPX:** UPX-Kompression beschädigt häufig CUDA-/PyTorch-DLLs.
- **GPU in der Exe:** Die Exe enthält die PyTorch-Version, die beim Build
  installiert ist. Der Build meldet in der Konsole, ob PyTorch mit CUDA
  eingepackt wird (`[spec] PyTorch … mit CUDA …`) oder eine WARNUNG, wenn
  nur die CPU-Version installiert ist. Auf Rechnern ohne NVIDIA-GPU läuft
  die GPU-Exe trotzdem, dann mit OpenVINO bzw. PyTorch auf der CPU.
- **OpenVINO-Modell mitliefern (optional):** Vor dem Build einmal
  `python -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt').export(format='openvino', dynamic=True, imgsz=1024)"`
  ausführen. Der Ordner `yolo11s-seg_openvino_model` wird dann
  automatisch eingepackt.

---

## Bedienung Schritt für Schritt

**Kleine Bildschirme:** Das Fenster lässt sich mit der Maus beliebig
verkleinern. Passt der Inhalt nicht mehr hinein, erscheint rechts (und bei
schmalem Fenster auch unten) ein Scrollbalken; gescrollt wird mit dem
Balken oder dem Mausrad (Shift + Mausrad = waagrecht). Über der Bildliste
scrollt das Mausrad die Liste selbst. Die unterste Zeile mit Status,
Fortschrittsbalken und **"Start"** bleibt immer sichtbar am unteren
Fensterrand. Beim Programmstart passt sich die Fenstergrösse automatisch
an den Bildschirm an.

1. **Ordner mit den Quellbildern auswählen**
   Bei der Zeile **"Ordner"** auf **"Durchsuchen…"** klicken und den Ordner
   mit den Bildern wählen. Das Tool durchsucht den Ordner **inklusive
   Unterordner** nach `.jpg`, `.jpeg` und `.png` (Gross-/Kleinschreibung
   wird ignoriert).

2. **Speicherort auswählen**
   Bei der neuen Zeile **"Speicherort"** direkt darunter auf
   **"Durchsuchen…"** klicken und den Ordner wählen, in dem die bearbeiteten
   Bilder abgelegt werden sollen. Dies muss ein anderer Ordner sein als
   der Quellordner (z. B. ein separater "Ausgabe"-Ordner) — die
   Originaldateien werden dadurch nicht verändert. Wird versehentlich der
   Quellordner selbst gewählt, verhindert das Tool den Start mit einem
   Hinweis.

   **Alternative: "Bilder im Quellordner ersetzen"** (Checkbox direkt darunter,
   Standard: **aus**). Ist sie aktiviert, werden die Originalbilder im
   Quellordner durch die bearbeiteten Bilder ersetzt; die Zeile
   "Speicherort" ist dann ausgegraut und wird nicht verwendet. Dabei gilt:
   - Das Ersetzen kann **nicht rückgängig** gemacht werden. Vor dem Start
     erscheint eine entsprechende Warnung.
   - Jedes Bild wird zuerst in eine temporäre Datei daneben gespeichert
     (`<name>.anonym-tmp.jpg`) und ersetzt das Original erst, wenn das
     Speichern fehlerfrei geklappt hat. Bei einem Fehler (z. B. Datenträger
     voll, Datei in einem anderen Programm geöffnet oder schreibgeschützt)
     bleibt das Original unversehrt und der Fehler erscheint in der
     Zusammenfassung.
   - Bilder, in denen **keine Person** erkannt wurde, bleiben
     **unangetastet** (kein erneutes JPEG-Komprimieren ohne Nutzen). Das
     gilt nicht, wenn die Belichtungskorrektur aktiv ist, "Metadaten
     übernehmen" ausgeschaltet oder "GPS-Position entfernen" eingeschaltet
     ist, denn dann ändert sich auch an diesen Bildern etwas.

3. **Bilder auswählen**
   Die gefundenen Bilder erscheinen als Liste (Pfad relativ zum gewählten
   Quellordner). Der Titel der Liste zeigt die Anzahl, z. B.
   **"Gefundene Bilder: 33"**. Ganz unten in der Statuszeile steht die
   Anzahl der aktuell ausgewählten Bilder, z. B. **"12 Bilder
   ausgewählt"**. Hinter jedem Dateinamen steht die Bildgrösse in Pixeln,
   z. B. `(2576 × 2576)`.

   Über **"Sortieren nach"** oberhalb der Liste lässt sich die Reihenfolge
   ändern:
   - **Name (A–Z)** (Standard), inklusive Unterordner-Pfad
   - **Bildgrösse (klein → gross)** bzw. **(gross → klein)**: nach Anzahl
     Pixel (Breite × Höhe); bei gleicher Grösse nach Name. Unlesbare
     Dateien stehen immer am Ende ("Grösse unbekannt").

   Bereits ausgewählte Bilder bleiben beim Umsortieren ausgewählt. So
   lassen sich z. B. alle Bilder einer Grösse (etwa alle Fisheye-Bilder
   2576 × 2576) bequem per `Shift`-Klick am Stück auswählen.

   Mehrfachauswahl ist möglich:
   - `Strg` + Klick: einzelne zusätzliche Bilder auswählen
   - `Shift` + Klick: Bereich auswählen
   - Buttons **"Alle auswählen"** / **"Auswahl aufheben"** für schnellen
     Zugriff auf die gesamte Liste

   Ist genau **ein** Bild in der Liste ausgewählt, erscheint rechts daneben
   im Feld **"Vorschau"** ein kleines Thumbnail des Bildes. Bei
   Mehrfachauswahl wird stattdessen die Anzahl ausgewählter Bilder
   angezeigt.

4. **Verwisch-Einstellungen wählen**
   Im Bereich **"Verwisch-Einstellungen"**:
   - **Methode:** **Gaussian Blur** (klassische Unschärfe) oder
     **Pixelieren (Mosaik)** (grobe, blockartige Pixel wie bei klassischer
     Anonymisierung in TV-Bildern)
   - **Stärke:** Regler von 1 (schwach) bis 30 (sehr stark), Vorgabe **22**
   - **Stärke an Personengrösse anpassen** (Vorgabe: **aktiviert**): Grosse,
     nahe Personen werden entsprechend stärker verwischt als kleine,
     entfernte. Siehe [Verwisch-Methoden](#verwisch-methoden).

   Rechts daneben zeigt eine **Live-Vorschau an einem kleinen, fiktiven
   Beispielbild** in Grauwerten (keinem echten Foto) sofort, wie der gewählte Effekt bei
   der aktuellen Stärke aussieht (beschriftet mit **"Vorschau"**). So lässt
   sich die Einstellung testen, bevor echte Bilder verarbeitet werden.

5. **Belichtungskorrektur einstellen (optional)**
   Im Bereich **"Belichtungskorrektur"**:
   - Checkbox **"Automatische Belichtungskorrektur aktivieren"** einschalten,
     wenn Schatten aufgehellt und/oder Lichter abgedunkelt werden sollen.
   - **Schatten aufhellen:** Regler von 0 (kein Effekt) bis 100 (maximale
     Wirkung), Vorgabe **22**.
   - **Lichter abdunkeln:** Regler von 0 (kein Effekt) bis 100 (maximale
     Wirkung), Vorgabe **22**.

   Die beiden Regler wirken **unabhängig voneinander** — z. B. lässt sich
   "Schatten aufhellen" auf 0 stellen, um ausschliesslich helle Bereiche
   abzudunkeln, oder umgekehrt. Das Tool prüft dabei automatisch, dass das
   Bild dadurch nicht zu hell wird (siehe
   [Funktionsweise](#funktionsweise-im-detail)).

6. **Bildtyp festlegen**
   Im Bereich **"Bildtyp"** genau eine der drei Optionen wählen
   (Radio-Buttons). Neben jeder Option zeigt ein kleines Graustufen-Icon,
   wie ein Raum mit Gitter-Wänden in dieser Projektion aussieht:
   - **Equirectangular** (Standard, z. B. Ricoh Theta Z1, 360°):
     Panorama-Verarbeitung mit Tile-Zerlegung
   - **Fisheye:** runde Fisheye-Bilder (Bildkreis), z. B. die einzelnen
     Linsenbilder einer 360°-Kamera
   - **Normale Perspektive:** direkte Verarbeitung als normales Foto

   Diese Einstellung gilt für **alle** in diesem Durchgang ausgewählten
   Bilder. Sollen verschiedene Bildtypen verarbeitet werden, ist pro
   Bildtyp ein separater Durchgang nötig.

7. **Metadaten (nur JPEG)**
   Im Bereich **"Metadaten (nur JPEG)"**:
   - **"Metadaten übernehmen"** (Vorgabe: **aktiviert**): EXIF
     (Kamera, Aufnahmezeit usw.), XMP (inkl. der 360°-Angaben der Theta Z1)
     und das Farbprofil werden ins bearbeitete Bild übernommen. Ohne diese
     Angaben erkennen 360°-Viewer das Bild nicht mehr als Panorama.
   - **"GPS-Position entfernen"** (Vorgabe: deaktiviert): entfernt die
     Aufnahmeposition aus EXIF und XMP, z. B. wenn auch der Ort
     anonymisiert werden soll. Für Dienste wie Google Street View muss
     die Position dagegen erhalten bleiben.

   Eingebettete Vorschaubilder werden **immer** entfernt, da sie die
   Personen unverwischt zeigen würden.

8. **Start**
   Unten rechts auf **"Start"** klicken. Wurde noch kein Speicherort
   ausgewählt (und ist "Bilder im Quellordner ersetzen" aus), erscheint zuerst ein
   Hinweis, dies nachzuholen. Andernfalls erscheint die Sicherheitsabfrage
   mit der Anzahl der ausgewählten Bilder, z. B.:

   > *"Achtung. Falls im Speicherort bereits Dateien mit gleichem Namen
   > vorhanden sind, werden diese überschrieben. Es werden 12 Bild(er)
   > bearbeitet. Soll der Prozess gestartet werden?"*

   Bei aktivierter Checkbox "Bilder im Quellordner ersetzen" lautet sie stattdessen:

   > *"Achtung. Die ausgewählten Originalbilder im Quellordner werden durch
   > die bearbeiteten Bilder ersetzt. Das kann nicht rückgängig gemacht
   > werden. Es werden 12 Bild(er) bearbeitet. Soll der Prozess gestartet
   > werden?"*

   - **Ja** → Verarbeitung startet
   - **Nein** → Abbrechen
   - **Abbrechen** → Abbrechen

9. **Fortschritt & Ergebnis**
   Während der Verarbeitung zeigt die Statuszeile das aktuell bearbeitete
   Bild sowie eine Fortschrittsleiste. Direkt **unter dem
   Fortschrittsbalken** steht eine Zeitabschätzung, z. B.:

   > *Verstrichen: 2 min 14 sec  ·  Restzeit: ca. 6 min 40 sec  ·  fertig um ca. 14:32 Uhr*

   Die Restzeit wird nach dem ersten fertigen Bild aus der
   durchschnittlichen Dauer pro Bild hochgerechnet und läuft jede Sekunde
   herunter. Die einmalige Ladezeit des Modells wird dabei nicht
   mitgerechnet. Da Bilder unterschiedlich lange brauchen (z. B. viele
   oder wenige Personen), ist der Wert eine Schätzung und wird mit jedem
   weiteren Bild genauer. Nach Abschluss zeigt die Zeile die Gesamtdauer
   und die durchschnittliche Dauer pro Bild. Zeiten werden mit Einheiten
   angezeigt (`h` / `min` / `sec`), z. B. `45 sec`, `2 min 14 sec` oder
   `1 h 02 min 05 sec`.

   Am Ende erscheint eine Zusammenfassung mit:
   - Anzahl verarbeiteter Bilder, Gesamtdauer und verwendetes Rechenwerk
   - Anzahl verwischter Bereiche insgesamt sowie Anzahl Bilder, in denen
     überhaupt etwas verwischt wurde
   - (falls Belichtungskorrektur aktiv) Anzahl aufgehellter und Anzahl
     abgedunkelter Bilder

   Falls bei einzelnen Bildern Fehler auftraten (z. B. defekte Datei),
   werden diese zusätzlich namentlich aufgelistet; der Rest wird aber
   trotzdem fertig verarbeitet. Die bearbeiteten Bilder liegen danach im
   gewählten Speicherort, mit derselben relativen Unterordner-Struktur wie
   im Quellordner.

---

## Funktionsweise im Detail

### Personendetektion (Segmentierung statt Bounding-Box)

Beide Bildarten nutzen dieselbe YOLO11-**Segmentierung** (COCO-Klasse
"person", Standard-Konfidenzschwelle 0.35) über
`PersonDetector.detect_masks_normal()` in `detector.py`. Im Unterschied zu
einem reinen Detektionsmodell liefert ein Segmentierungsmodell nicht nur
eine grobe Bounding-Box, sondern für jede erkannte Person ein Polygon, das
ihre tatsächliche Silhouette (Kopf, Schultern, Körperumriss) beschreibt.
Das ist die Grundlage dafür, dass wirklich nur die Person selbst verwischt
wird und nicht zusätzlich ein grosser rechteckiger oder ovaler
Hintergrundbereich.

### Verwisch-Methoden

`processing.apply_blur_effect()` kennt zwei Methoden, beide mit einer
Stärke von 1 (schwach) bis 30 (sehr stark) einstellbar:

- **`gaussian`** – klassischer Gaussian Blur. Die Stärke steuert die
  Kernelgrösse (`k = 2·Stärke + 1`, auf ungerade Werte begrenzt und an die
  Grösse des Bildausschnitts angepasst).
- **`pixelate`** – Mosaik-Effekt. Der Bildausschnitt wird stark verkleinert
  (Blockgrösse = Stärke) und ohne Glättung wieder hochskaliert, wodurch
  grobe, blockartige Pixel entstehen. Diese Methode macht Details
  besonders zuverlässig unkenntlich, auch bei kleineren Personen.

**Stärke an Personengrösse anpassen:** Ist die Option aktiviert (Standard),
wird jeder verwischte Bereich einzeln bearbeitet und die Stärke mit seiner
Grösse skaliert (`processing.effective_strength()`): Die eingestellte
Stärke gilt für einen Bereich von 128 px (`REFERENCE_REGION_SIZE`, etwa
die Grösse der Figur in der Vorschau). Ein Bereich von 1280 px wird
entsprechend 10× stärker verwischt. Die eingestellte Stärke ist dabei die
Untergrenze, kleine Personen werden also nie schwächer verwischt als ohne
die Option. Ohne diese Anpassung wäre z. B. ein Kernel von 45 px bei einer
nahen Person auf einem 6720×3360-Panorama zu schwach, und Gesichter
könnten erkennbar bleiben. Sehr grosse Gaussian-Kernel werden intern auf
einer verkleinerten Kopie berechnet, damit die Verarbeitung schnell bleibt.

Beide Methoden lassen sich im GUI live an einem kleinen, frei erfundenen
Beispielbild (`App._create_fictive_sample()` in `main.py`) testen, bevor
echte Bilder verarbeitet werden. Die Vorschau nutzt dabei bewusst ein
unregelmässiges Silhouetten-Polygon (keine Box/kein Oval), damit sie zeigt,
wie es auch bei echten Bildern aussieht.

### Normale Bilder

1. `detect_masks_normal()` liefert eine Liste von Silhouetten-Polygonen.
2. `processing.build_mask_from_polygons()` füllt daraus eine Maske und
   erweitert sie nur um einen sehr kleinen, festen Pixel-Rand
   (`margin_px`, standardmässig ca. 0.4 % der kleineren Bildseite,
   mindestens 2 Pixel) – gerade genug, um z. B. einzelne Haarsträhnen am
   Rand mit abzudecken, ohne unnötig viel Hintergrund einzubeziehen.
3. `processing.blur_mask()` wendet die gewählte Verwisch-Methode (Gaussian
   Blur oder Pixelieren) mit der im GUI eingestellten Stärke nur auf diese
   Maske an, mit weichem Übergang am Rand (keine harte Kante).

### Equirectangular-Bilder (Theta Z1)

`detector.detect_mask_equirect()`, wie ursprünglich vorgegeben:

1. **Zerlegung in Tiles:** Das Panorama wird in ca. 16 überlappende,
   perspektivische Ausschnitte (Blickrichtungen) zerlegt, die zusammen die
   gesamte Kugel abdecken (Äquatorband, oberes/unteres Band, Zenit, Nadir).
2. **Segmentierung pro Tile:** YOLO11 läuft auf jedem der perspektivischen
   Tiles und liefert dort Silhouetten-Polygone (auf denen Personen kaum
   verzerrt und daher zuverlässig erkennbar sind).
3. **Rückprojektion:** Jedes Polygon wird zu einer Tile-Maske gefüllt (mit
   demselben minimalen Pixel-Rand wie bei normalen Bildern) und über die
   bekannte Projektionsabbildung direkt als Maske ins
   Panorama-Koordinatensystem zurückgeschrieben. Kleine Lücken, die durch
   die Rückprojektion entstehen können, werden morphologisch geschlossen.
4. **Verwischen:** `processing.blur_mask()` wendet die gewählte Methode nur
   auf die so entstandene, unregelmässig geformte Panorama-Maske an.

### Fisheye-Bilder

`detector.detect_mask_fisheye()` folgt demselben Prinzip wie die
Equirectangular-Verarbeitung, angepasst an die Fisheye-Abbildung:

1. **Bild schneiden:** Der Bildkreis wird in 9 überlappende, entzerrte
   perspektivische Tiles (je 100° Bildwinkel) zerlegt: eines in
   Blickrichtung der Kamera und ein Ring von 8 Tiles rundherum bis zum
   Rand des Bildkreises (bei sehr grossem Bildwinkel ein zweiter Ring).
   In den Tiles sind Personen kaum noch verzerrt. Die Tile-Grösse richtet
   sich nach der Auflösung des Fisheye-Bildes (640–1024 px), und YOLO
   arbeitet mit derselben Eingabegrösse, damit keine Details verloren
   gehen.
2. **Segmentieren:** Die Tiles werden in mehreren Drehungen segmentiert
   und die Ergebnisse vereinigt. Grund: Die Kamera kann gedreht montiert
   sein. In den Beispielbildern ist sie um ±90° gedreht, Personen liegen
   dort quer im Bild, und YOLO erkennt liegende Personen deutlich
   schlechter als stehende. Das geschieht in zwei Stufen:
   - **Stufe 1:** alle Tiles in 0° und 90°. Damit sind stehende,
     kopfstehende und liegende Personen bereits abgedeckt.
   - **Stufe 2:** nur Tiles, in denen Stufe 1 etwas gefunden hat (auch
     unsichere Treffer ab Konfidenz 0.2), zusätzlich in 180° und 270°.
     Das verfeinert die Silhouetten. Tiles ohne Personen werden so nur
     halb so oft gerechnet.
3. **Wieder zusammenfügen:** Die Silhouetten jedes Tiles werden exakt
   (dicht, über die inverse Abbildung) ins Fisheye-Bild zurückprojiziert
   und zu einer Gesamtmaske vereinigt. Ausserhalb des Bildkreises wird
   nichts maskiert.
4. **Verwischen:** wie bei den anderen Bildtypen über
   `processing.blur_mask()`.

**Objektiv-Parameter** (oben in `detector.py`):

- `FISHEYE_FOV_DEG = 200`: Bildwinkel über den Durchmesser des
  Bildkreises. Die Kamera des Beispielbildes sieht einen Teil ihres
  eigenen Gehäuses, der Bildwinkel liegt also über 180°. Gerade Kanten
  (Geländer, Mauern) erscheinen in den Tiles mit 200° gerade.
- `FISHEYE_PROJECTION = "equidistant"`: Abbildungsfunktion des Objektivs
  (alternativ `"equisolid"` oder `"stereographic"`).
- Bildkreis: Mittelpunkt = Bildmitte, Radius = halbe kürzere Bildseite
  (passend für quadratische Bilder, deren Kreis an den Rand stösst; sonst
  über `center`/`radius` in `detect_mask_fisheye()` anpassen).

Vorwärts- und Rückprojektion verwenden dasselbe Modell. Ein ungenauer
Bildwinkel führt deshalb nur zu leicht verzerrten Tiles, nicht zu
verschobenen Masken.

### Geschwindigkeit

Praktisch die gesamte Rechenzeit entfällt auf die YOLO-Segmentierung. Das
Tool beschleunigt sie automatisch:

- **Rechenwerk:** Mit NVIDIA-GPU (CUDA) wird auf der GPU mit halber
  Genauigkeit (FP16) gerechnet. Ohne GPU wird, falls das Paket `openvino`
  installiert ist, OpenVINO verwendet, das auf der CPU etwa 2.5-mal
  schneller ist als PyTorch. Sonst läuft PyTorch auf der CPU. Das
  verwendete Rechenwerk steht in der Abschluss-Zusammenfassung.
- **OpenVINO-Modell:** Beim allerersten Start ohne GPU wird das Modell
  einmalig in das OpenVINO-Format umgewandelt (wenige Sekunden) und unter
  `%LOCALAPPDATA%\ImageBlurring_YOLO11` gespeichert. Liegt ein Ordner
  `yolo11s-seg_openvino_model` neben `yolo11s-seg.pt` (z. B. in der Exe
  mitgeliefert), wird dieser direkt verwendet. Schlägt OpenVINO fehl,
  läuft das Tool automatisch mit PyTorch weiter.
- **Batches:** Alle Tiles eines Bildes werden gebündelt an YOLO übergeben
  (`BATCH_SIZE_GPU` = 8, `BATCH_SIZE_CPU` = 4).
- **Fisheye in zwei Stufen:** siehe [Fisheye-Bilder](#fisheye-bilder).
- **Zwischengespeicherte Abbildungs-Tabellen:** Die Tabellen für das
  Schneiden der Tiles werden nur einmal pro Bildgrösse berechnet.

Gemessen auf einem einzelnen CPU-Kern (Fisheye 2576 × 2576 px): vorher
etwa 45 s pro Bild, jetzt etwa 9–12 s mit OpenVINO bzw. 30–40 s mit
PyTorch. Die Masken sind mit PyTorch identisch zu vorher; OpenVINO
rechnet minimal anders, die Masken weichen um etwa 1 % ab (Randpixel).

### Bildausrichtung (Hochformat-Fotos, RiSCAN)

Viele Kameras speichern Hochformat-Fotos quer und vermerken nur im
EXIF-Tag "Orientation", dass das Bild beim Anzeigen gedreht werden muss.
Programme wie **RiSCAN** verwenden dagegen die Pixel so, wie die Kamera sie
gespeichert hat (die Kamerakalibrierung bezieht sich auf den Sensor).

Das Tool verändert die Bildgeometrie deshalb **nie**:

- Die Pixel werden ungedreht gelesen und ungedreht gespeichert, das
  Orientation-Tag bleibt erhalten (EXIF und XMP). Das bearbeitete Bild hat
  dieselben gespeicherten Abmessungen wie das Original.
- Für die Personenerkennung wird intern eine aufgerichtete Kopie
  verwendet (aufrechte Personen werden zuverlässiger erkannt), die Maske
  wird danach in die gespeicherte Ausrichtung zurückgedreht.
- Ist "Metadaten übernehmen" ausgeschaltet, wird ein minimales EXIF
  geschrieben, das nur das Orientation-Tag enthält, damit Bildbetrachter
  das Bild weiterhin richtig herum anzeigen.
- PNG-Dateien und JPEG ohne EXIF sind nicht betroffen.

### Anzahl verwischter Bereiche

`processing.count_regions()` zählt über eine Analyse zusammenhängender
Komponenten (`cv2.connectedComponents`), wie viele getrennte Bereiche in
der finalen Maske eines Bildes tatsächlich verwischt wurden. Diese Zahl
wird über alle verarbeiteten Bilder aufsummiert und am Ende als
Gesamtstatistik angezeigt (siehe [Bedienung](#bedienung-schritt-für-schritt),
Punkt 8).

### Automatische Belichtungskorrektur

`processing.auto_correct_exposure()` passt **ausschliesslich den L-Kanal**
(Helligkeit) im LAB-Farbraum an; die Farbkanäle a und b bleiben komplett
unverändert, damit keine Farbverschiebung entsteht. Die Korrektur erfolgt
über eine Tonwertkurve mit **zwei unabhängigen, im GUI einstellbaren
Reglern (je 0–100)**: `shadow_intensity` (Schatten aufhellen) und
`highlight_intensity` (Lichter abdunkeln). Anders als in einer früheren
Version sind diese beiden Effekte **nicht mehr aneinander gekoppelt** —
jeder Regler wirkt für sich:

1. **Schatten aufhellen** (`shadow_intensity`): Über eine Potenzfunktion
   (`1 - (1 - L)^gamma`, wobei `gamma` von `shadow_intensity` abhängt)
   werden dunkle Pixel angehoben; die Stärke des Effekts nimmt zu
   helleren Tonwerten hin weich ab (voller Effekt bei sehr dunklen
   Pixeln, praktisch kein Effekt mehr ab einem mittleren Tonwert). Bei
   `shadow_intensity = 0` bleiben dunkle Bereiche unverändert.
2. **Lichter abdunkeln** (`highlight_intensity`): Unabhängig von Schritt 1
   werden sehr helle Pixel weich in Richtung Mittelton komprimiert. Bei
   `highlight_intensity = 0` bleiben helle Bereiche unverändert.
3. **Sicherheitsprüfung:** Steigt die mittlere Helligkeit oder der Anteil
   überbelichteter Pixel (L > 245) durch die Korrektur trotzdem zu stark an,
   wird die Korrekturstärke automatisch in mehreren Stufen reduziert.
4. Die Funktion gibt zusätzlich die Helligkeitsdifferenz (nachher minus
   vorher) zurück. Im GUI wird daraus gezählt, wie viele Bilder aufgehellt
   (Differenz > 1) bzw. abgedunkelt (Differenz < -1) wurden — diese Zahlen
   erscheinen in der Abschluss-Zusammenfassung.

Da beide Regler unabhängig sind, lässt sich z. B. gezielt nur die
Schattenaufhellung nutzen (Regler "Lichter abdunkeln" auf 0), oder
umgekehrt — je nach Bildmaterial, ohne den Code ändern zu müssen.

---

## Anpassbare Einstellungen

### Im GUI einstellbar

| Einstellung | Wo im GUI | Bereich / Optionen | Vorgabewert |
|---|---|---|---|
| Speicherort für bearbeitete Bilder | Zeile "Speicherort" (unter "Ordner") | beliebiger Ordner ausser dem Quellordner (Pflichtfeld, wenn "Bilder im Quellordner ersetzen" aus ist) | leer (muss gewählt werden) |
| Bilder im Quellordner ersetzen | Checkbox unter "Speicherort" | an (Originale ersetzen) / aus (Kopie im Speicherort) | **aus** |
| Verwisch-Methode | Verwisch-Einstellungen → Methode | Gaussian Blur / Pixelieren (Mosaik) | Gaussian Blur |
| Verwisch-Stärke | Verwisch-Einstellungen → Regler | 1 (schwach) – 30 (sehr stark) | **22** |
| Belichtungskorrektur aktivieren | Belichtungskorrektur → Checkbox | an / aus | aus |
| Schatten aufhellen (Intensität) | Belichtungskorrektur → Regler | 0 (kein Effekt) – 100 (maximal) | **22** |
| Lichter abdunkeln (Intensität) | Belichtungskorrektur → Regler | 0 (kein Effekt) – 100 (maximal) | **22** |
| Stärke an Personengrösse anpassen | Verwisch-Einstellungen → Checkbox | an / aus | **an** |
| Bildtyp | Bildtyp → Radio-Buttons | Equirectangular / Fisheye / Normale Perspektive | **Equirectangular** |
| Metadaten übernehmen | Metadaten → Checkbox | an / aus | **an** |
| GPS-Position entfernen | Metadaten → Checkbox | an / aus | aus |

### Nur im Code anpassbar

Folgende Werte werden bewusst nicht im GUI angeboten, um die Oberfläche
übersichtlich zu halten, lassen sich aber direkt im Code ändern:

| Einstellung | Datei | Parameter | Standard |
|---|---|---|---|
| YOLO11-Segmentierungsmodell | `main.py` | `model_path` in `PersonDetector(...)` | `"yolo11s-seg.pt"` |
| Konfidenzschwelle | `main.py` | `conf_threshold` in `PersonDetector(...)` | `0.35` |
| Sicherheitsrand um Silhouette (normale Bilder) | `processing.py` | `margin_px` in `build_mask_from_polygons()` | `~0.4 %` der kürzeren Bildseite (min. 2 px) |
| Sicherheitsrand um Silhouette (Panorama) | `detector.py` | `margin_px` in `detect_mask_equirect()` | `~0.6 %` der Tile-Grösse (min. 2 px) |
| Tile-Grösse (Panorama) | `detector.py` | `tile_size` in `detect_mask_equirect()` | `900` |
| Field of View pro Tile | `detector.py` | `fov_deg` in `detect_mask_equirect()` | `100` |
| Referenzgrösse für angepasste Stärke | `processing.py` | `REFERENCE_REGION_SIZE` | `128` px |
| JPEG-Qualität beim Speichern | `image_io.py` | `JPEG_QUALITY` | `95` |
| Fisheye-Bildwinkel | `detector.py` | `FISHEYE_FOV_DEG` | `200` |
| Fisheye-Abbildungsfunktion | `detector.py` | `FISHEYE_PROJECTION` | `"equidistant"` |
| Fisheye-Drehungen Stufe 1 (alle Tiles) | `detector.py` | `FISHEYE_ROTATIONS` | `(0, 90)` |
| Fisheye-Drehungen Stufe 2 (Tiles mit Treffer) | `detector.py` | `FISHEYE_REFINE_ROTATIONS` | `(180, 270)` |
| Auslöse-Konfidenz für Stufe 2 | `detector.py` | `FISHEYE_REFINE_TRIGGER_CONF` | `0.2` |
| Tiles pro YOLO-Aufruf (GPU / CPU) | `detector.py` | `BATCH_SIZE_GPU` / `BATCH_SIZE_CPU` | `8` / `4` |
| Rechenwerk | `detector.py` | `backend` in `PersonDetector(...)` | `"auto"` (GPU > OpenVINO > PyTorch) |
| Fisheye-Bildkreis (Mittelpunkt/Radius) | `detector.py` | `center` / `radius` in `detect_mask_fisheye()` | Bildmitte / halbe kürzere Seite |
| Max. erlaubte Helligkeitszunahme | `processing.py` | `max_mean_gain` in `auto_correct_exposure()` | `55` |
| Max. erlaubte Zunahme überbelichteter Pixel | `processing.py` | `max_overexposed_gain` in `auto_correct_exposure()` | `0.03` |
| Wertebereich des Stärke-Reglers (Verwischen) | `main.py` | `from_` / `to` in `self.strength_scale` | `1` – `30` |
| Wertebereich des Schatten-Reglers (Belichtung) | `main.py` | `from_` / `to` in `self.shadow_scale` | `0` – `100` |
| Wertebereich des Lichter-Reglers (Belichtung) | `main.py` | `from_` / `to` in `self.highlight_scale` | `0` – `100` |
| Schwelle für "aufgehellt"/"abgedunkelt" in der Statistik | `main.py` | Vergleich `mean_diff` in `process_images()` | `> 1.0` / `< -1.0` |

Modellgrössen von YOLO11 (schneller ↔ genauer):
`yolo11n.pt` < `yolo11s.pt` < `yolo11m.pt` < `yolo11l.pt` < `yolo11x.pt`

---

## Häufige Probleme (FAQ)

**Das Modell wird nicht heruntergeladen / Fehler beim ersten Start**
→ Internetverbindung prüfen. Alternativ das Modell manuell herunterladen
und den Dateipfad in `model_path` anpassen.

**Verarbeitung ist sehr langsam**
→ In der Abschluss-Zusammenfassung nachsehen, welches Rechenwerk verwendet
wurde. Steht dort "PyTorch (CPU)", das Paket `openvino` installieren
(`pip install openvino`), das macht die Segmentierung etwa 2.5-mal
schneller. Weitere Möglichkeiten: GPU verwenden oder ein kleineres Modell
(`yolo11n-seg.pt`, erkennt aber weniger zuverlässig).

**Personen werden nicht erkannt**
→ `conf_threshold` in `main.py` senken (z. B. auf `0.25`), oder ein
grösseres Segmentierungsmodell verwenden (`yolo11m-seg.pt` oder
`yolo11l-seg.pt`) für höhere Erkennungsgenauigkeit (auf Kosten der
Geschwindigkeit).

**Fisheye-Verarbeitung ist langsam**
→ Die 9 Tiles werden in Stufe 1 in zwei Drehungen segmentiert, Tiles mit
Personen in Stufe 2 nochmals in zwei weiteren. Ist die Kamera immer gleich
montiert, kann man in `detector.py` `FISHEYE_ROTATIONS` auf die passende
Drehung und `FISHEYE_REFINE_ROTATIONS` auf `()` setzen (für das
Front-Beispielbild `(270,)`). Das ist nochmals 2- bis 4-mal schneller,
Personen in anderer Ausrichtung werden dann aber schlechter erkannt.

**Fisheye: Tiles wirken verzerrt / Personen am Bildrand werden nicht erkannt**
→ Bildwinkel (`FISHEYE_FOV_DEG`) und Abbildungsfunktion
(`FISHEYE_PROJECTION`) an das Objektiv anpassen (Datenblatt der Kamera).
Bei nicht quadratischen Bildern oder versetztem Bildkreis `center` und
`radius` in `detect_mask_fisheye()` angeben.

**Am Rand einer Person bleibt ein kleiner Streifen unverwischt**
→ `margin_px` in `build_mask_from_polygons()` (normale Bilder) bzw. in
`detect_mask_equirect()` (Panorama) leicht erhöhen.

**Am Bildrand/Nahtstelle des Panoramas (180°/-180° Meridian) bleibt eine
Person teilweise unverwischt**
→ Kann in Ausnahmefällen bei Personen genau auf der Naht vorkommen.
`tile_size` und/oder `fov_deg` erhöhen, um mehr Überlappung zu erzeugen.

**Bilder wirken nach der Belichtungskorrektur kaum verändert**
→ Zuerst den passenden Regler ("Schatten aufhellen" für dunkle Bereiche
bzw. "Lichter abdunkeln" für helle Bereiche) im GUI erhöhen. Ist das Bild
insgesamt sehr dunkel, könnte auch die Sicherheitsbremse gegriffen haben:
Wenn die volle Korrektur das Bild zu stark aufhellen würde, wird sie
automatisch abgeschwächt. `max_mean_gain` in `auto_correct_exposure()`
kann bei Bedarf im Code erhöht werden.

**Die Belichtungskorrektur wirkt zu stark**
→ Den betreffenden Regler ("Schatten aufhellen" bzw. "Lichter abdunkeln")
im GUI verringern (0 = kein Effekt). Da beide Regler unabhängig
voneinander wirken, lässt sich z. B. gezielt nur einer der beiden Effekte
abschalten.

**Person ist nach der Verarbeitung noch erkennbar**
→ Prüfen, ob "Stärke an Personengrösse anpassen" aktiviert ist. Danach
die Stärke des Verwisch-Reglers erhöhen, oder auf "Pixelieren (Mosaik)"
wechseln — diese Methode entfernt Details tendenziell zuverlässiger als
Gaussian Blur bei gleicher Stärke.

**Bearbeitete Fotos sind in RiSCAN um 90° gedreht (im Bildbetrachter aber richtig)**
→ Behoben (Version vom 18.11.2026): Das Tool behält jetzt die gespeicherte
Pixel-Ausrichtung und das EXIF-Tag "Orientation" des Originals bei, siehe
[Bildausrichtung](#bildausrichtung-hochformat-fotos-riscan). Bilder, die
mit einer älteren Version bearbeitet wurden, bitte erneut aus den
Originalen erzeugen.

**Das bearbeitete Panorama wird im 360°-Viewer flach angezeigt**
→ "Metadaten übernehmen" aktivieren. Die 360°-Angaben (GPano-Tags im XMP)
werden nur bei JPEG-Dateien übernommen.

**Hinweis "EXIF-Daten nicht lesbar, wurden weggelassen" in der Zusammenfassung**
→ Das Bild wurde trotzdem gespeichert, aber ohne EXIF, weil dessen Struktur
nicht sicher bearbeitet werden konnte (ein eingebettetes, unverwischtes
Vorschaubild hätte sonst nicht zuverlässig entfernt werden können).

**Das Fenster passt nicht auf meinen Bildschirm**
→ Fenster mit der Maus kleiner ziehen. Sobald der Inhalt nicht mehr
hineinpasst, erscheint ein Scrollbalken; der "Start"-Knopf bleibt unten
immer sichtbar.

**Die Bild-Vorschau neben der Liste erscheint nicht**
→ Es ist genau ein Bild in der Liste auszuwählen (bei Mehrfachauswahl wird
nur die Anzahl angezeigt). Bei sehr grossen oder beschädigten Dateien kann
das Laden der Vorschau fehlschlagen; es erscheint dann der Hinweis
"Vorschau nicht verfügbar", die Verarbeitung selbst ist davon nicht
betroffen.

**"Kein Speicherort" beim Klick auf Start**
→ Es muss zuerst über die Zeile "Speicherort" ein Zielordner für die
bearbeiteten Bilder ausgewählt werden (separat von der Zeile "Ordner", die
nur die Quellbilder festlegt). Alternativ "Bilder im Quellordner ersetzen"
aktivieren, dann werden die Originale ersetzt und kein Speicherort
benötigt.

**"Speicherort = Quellordner" beim Klick auf Start**
→ Als Speicherort wurde der Quellordner selbst gewählt, die Originale
würden also überschrieben, obwohl "Bilder im Quellordner ersetzen" aus ist. Einen
anderen Speicherort wählen, oder "Bilder im Quellordner ersetzen" bewusst
aktivieren.

**Fehler "Bild konnte nicht ersetzt werden" (beim Überschreiben)**
→ Die Originaldatei ist schreibgeschützt oder gerade in einem anderen
Programm geöffnet (z. B. in einem Bildbetrachter). Das Original ist
unverändert geblieben. Datei schliessen bzw. Schreibschutz entfernen und
das Bild erneut verarbeiten.

**Im Ordner liegt eine Datei "….anonym-tmp.jpg"**
→ Temporäre Datei aus einem abgebrochenen Speichervorgang (z. B. wenn das
Programm während des Speicherns gewaltsam beendet wurde). Sie kann
gelöscht werden; das Tool zeigt solche Dateien in der Bildliste nicht an.

**Ich finde meine bearbeiteten Bilder nicht**
→ Ist "Bilder im Quellordner ersetzen" aus, liegen sie nicht im Quellordner, sondern
im gewählten **Speicherort** (er wird auch in der Abschluss-Zusammenfassung
genannt), mit derselben relativen Unterordner-Struktur wie im
Quellordner (z. B. `Speicherort/unterordner/bild.jpg`, wenn das Original
unter `Quellordner/unterordner/bild.jpg` lag).

---

## Grenzen des Tools

- Es wird die gesamte erkannte Person (Silhouette) verwischt, nicht nur das
  Gesicht — das entspricht der ursprünglichen Anforderung
  "Unkenntlichmachen von Personen".
- Die Segmentierungs-Maske folgt der vom Modell erkannten Kontur; bei sehr
  unscharfen Konturen (z. B. wehende Kleidung, starke Bewegungsunschärfe)
  kann die Silhouette leicht ungenau sein — ein kleiner Sicherheitsrand
  (`margin_px`) gleicht das teilweise aus.
- Bei sehr kleinen, weit entfernten oder stark verdeckten Personen kann die
  Erkennung fehlschlagen (kein Detektor ist zu 100 % zuverlässig).
- Die Equirectangular-Verarbeitung ist rechenintensiv (16 Tiles pro Bild,
  zusätzlich rechenintensiver als reine Detektion, da pro Tile eine
  Segmentierung statt nur einer Bounding-Box-Erkennung läuft).
- Metadaten werden nur bei JPEG übernommen, nicht bei PNG. Die
  herstellerspezifischen Daten (MakerNote) werden unverändert übernommen.
- Es wird nur eine flache Prüfung auf Namenskollisionen im Speicherort
  vorgenommen: Existiert dort bereits eine Datei mit demselben relativen
  Pfad, wird sie beim Speichern überschrieben (siehe Sicherheitsabfrage
  vor dem Start). Ein Backup der bearbeiteten Bilder wird nicht automatisch
  angelegt.
- Mit "Bilder im Quellordner ersetzen" wird kein Backup der Originale angelegt.
  Wer die Originale später noch braucht, sollte vorher eine Kopie des
  Ordners anlegen oder im Standard-Modus (Kopie im Speicherort) arbeiten.
  Überschriebene JPEG-Bilder werden neu komprimiert (Qualität 95).

---

## Projektstruktur

```
personen_anonymisierer/
├── main.py              # GUI, Ablaufsteuerung, Dateiverwaltung
├── detector.py           # YOLO11-Segmentierung, Equirectangular-/Fisheye-Tiling & Rückprojektion
├── processing.py         # Gaussian Blur, Belichtungskorrektur
├── icons.py              # Schematische Graustufen-Icons der Bildtypen (werden berechnet)
├── image_io.py           # Unicode-sicheres Lesen/Schreiben, Metadaten, Bildausrichtung
├── requirements.txt       # Python-Abhängigkeiten
├── README.md              # Kurzübersicht
└── ANLEITUNG.md            # Diese ausführliche Anleitung
```

---

## Änderungsverlauf

| Datum | Änderung |
|---|---|
| 2026-08-19 | Erste Version: GUI, YOLO11-Detektion, Equirectangular-Tiling, Gaussian Blur, automatische Belichtungskorrektur |
| 2026-08-19 | Verwisch-Stärke im GUI per Regler einstellbar (mit Live-Vorschau an fiktivem Beispielbild); zweite Verwisch-Methode "Pixelieren (Mosaik)" ergänzt; Rand um erkannte Personen deutlich verkleinert (`margin_ratio` 0.15/0.12 → 0.04); kleine Bildvorschau neben der Liste ergänzt; neue Abhängigkeit `Pillow` |
| 2026-08-19 | Umstellung von Bounding-Box-Detektion auf YOLO11-**Segmentierung** (`yolo11s-seg.pt`): Es wird nur noch die tatsächliche, unregelmässige Personen-Silhouette verwischt statt Rechteck/Oval, mit minimalem Pixel-Sicherheitsrand statt prozentualem Box-Rand; Abschluss-Zusammenfassung mit Anzahl verwischter Bereiche/Bilder sowie Anzahl auf-/abgehellter Bilder ergänzt; Belichtungskorrektur von CLAHE auf eine gezielte Schatten-/Lichter-Tonwertkurve umgestellt, die dunkle Bereiche deutlich stärker aufhellt |
| 2026-08-19 | Belichtungskorrektur überarbeitet, da zu stark: jetzt reines LAB-basiertes Aufhellen des L-Kanals unterhalb eines im GUI per Regler einstellbaren Dunkelheits-Grenzwerts (0–255), mit Live-Vorschau an einem zweiten fiktiven Beispielbild (Helligkeitsverlauf mit Grenzwert-Markierung); Lichter-Abdunklung entfernt; Start-Bestätigungsdialog nennt neu die Anzahl der ausgewählten Bilder |
| 2026-08-19 | Belichtungskorrektur erneut überarbeitet: zurück zur Tonwertkurve (Schatten aufhellen + Lichter leicht abdunkeln, wie vor dem Dunkelheits-Grenzwert-Ansatz), aber jetzt mit einem einzigen, im GUI einstellbaren **Intensitäts-Regler (0–100)** statt eines Dunkelheits-Grenzwerts; das zweite fiktive Beispielbild für die Belichtungs-Vorschau (Helligkeitsverlauf mit Markierung) wurde entfernt |
| 2026-08-19 | Vorgabewerte der Regler für Verwisch-Stärke und Belichtungs-Intensität auf **22** gesetzt; neue Zeile "Speicherort" im GUI unterhalb der Ordner-Auswahl ergänzt — die bearbeiteten Bilder werden nicht mehr im Quellordner überschrieben, sondern unter dem gewählten Speicherort abgelegt (mit derselben relativen Unterordner-Struktur wie im Quellordner); Start ohne gewählten Speicherort wird verhindert; Sicherheitsabfrage vor dem Start entsprechend angepasst |
| 2026-08-19 | Belichtungskorrektur entkoppelt: statt eines einzigen Intensitäts-Reglers gibt es jetzt zwei **unabhängige** Regler "Schatten aufhellen" und "Lichter abdunkeln" (je 0–100, Vorgabe 22), die sich getrennt voneinander einstellen lassen (`auto_correct_exposure(shadow_intensity, highlight_intensity, ...)`); Checkbox "Theta Z1 (Equirectangular)" im Bildtyp-Bereich ist neu standardmässig aktiviert |
| 2026-11-11 | Bilder werden über `image_io.read_image()` / `write_image()` gelesen und gespeichert (`np.fromfile` + `cv2.imdecode` bzw. `cv2.imencode` + `tofile`) statt direkt über `cv2.imread` / `cv2.imwrite`. Dadurch funktionieren unter Windows auch Ordner- und Dateinamen mit Umlauten oder anderen Nicht-ASCII-Zeichen (z.B. "Zürich"), die vorher still als "Bild konnte nicht gelesen/gespeichert werden" fehlschlugen |
| 2026-11-11 | **Metadaten:** EXIF, XMP (inkl. GPano-Tags der Theta Z1) und ICC-Farbprofil werden bei JPEG ins bearbeitete Bild übernommen (`image_io.read_metadata()` / `write_image_with_meta()`), sodass 360°-Viewer das Panorama wieder erkennen. Eingebettete Vorschaubilder (EXIF-Thumbnail, `xmp:Thumbnails`, Extended XMP, APP13, MPF) werden aus Datenschutzgründen immer entfernt; EXIF-/XMP-Orientierung wird auf 1 gesetzt, da die Pixel bereits gedreht sind; neue Option "GPS-Position entfernen". JPEG-Qualität explizit auf 95 gesetzt. **Verwischen:** Stärke wird pro Bereich an die Personengrösse angepasst (`effective_strength()`, Referenz 128 px, neue Checkbox "Stärke an Personengrösse anpassen", Vorgabe an); `blur_mask()` arbeitet dazu pro zusammenhängendem Bereich; sehr grosse Gaussian-Kernel werden auf einer verkleinerten Kopie berechnet |
| 2026-11-11 | Zeitabschätzung unter dem Fortschrittsbalken: verstrichene Zeit, geschätzte Restzeit (Durchschnitt der fertigen Bilder, sekündlich heruntergezählt) und voraussichtliche Uhrzeit des Abschlusses; nach Abschluss Gesamtdauer und ø pro Bild; Dauer auch in der Abschluss-Zusammenfassung; Zeiten mit Einheiten (h / min / sec) |
| 2026-11-11 | Listentitel zeigt die Anzahl gefundener Bilder ("Gefundene Bilder: ##", Zusatz "(Mehrfachauswahl möglich)" entfernt); Statuszeile zeigt statt "## Bild(er) gefunden." laufend die Anzahl ausgewählter Bilder ("## Bilder ausgewählt", während der Verarbeitung weiterhin den Fortschritt) |
| 2026-11-11 | Erklärtext zur Verwisch-Vorschau ("Die Vorschau rechts zeigt …") aus dem GUI entfernt; stattdessen steht direkt unter dem fiktiven Beispielbild die Beschriftung "Vorschau" |
| 2026-11-11 | Neuer Bildtyp **Fisheye**: Bereich "Bildtyp" mit drei Radio-Buttons (Equirectangular / Fisheye / Normale Perspektive, Vorgabe Equirectangular) statt Checkbox; `detector.detect_mask_fisheye()` zerlegt den Bildkreis in 9 entzerrte Tiles (100°, Tile-Grösse 640–1024 px nach Auflösung), segmentiert jedes Tile in vier Drehungen (gedreht montierte Kamera) und projiziert die Silhouetten exakt ins Fisheye-Bild zurück; Objektiv-Parameter `FISHEYE_FOV_DEG` (200), `FISHEYE_PROJECTION` (equidistant), `FISHEYE_ROTATIONS`; `detect_masks_normal()` akzeptiert optional `imgsz` |
| 2026-11-11 | Bildtyp-Auswahl mit schematischen Graustufen-Icons (Würfelraum mit Gitter, je Projektion berechnet in neuem Modul `icons.py`: Equirectangular 2:1, Fisheye 180° rund, normale Perspektive); Zusatz "(Ricoh Theta Z1, 360°)" im Radio-Button-Text entfernt; Fenster 30 px höher |
| 2026-11-11 | **Schnellere Segmentierung:** automatische Wahl des Rechenwerks (GPU mit FP16 > OpenVINO auf der CPU, einmaliger Export in `%LOCALAPPDATA%\\ImageBlurring_YOLO11` > PyTorch-CPU; Anzeige in der Zusammenfassung); alle Tiles eines Bildes gebündelt (Batches); Fisheye zweistufig (alle Tiles in 0°/90°, 180°/270° nur für Tiles mit Treffer ab Konfidenz 0.2); Abbildungs-Tabellen pro Bildgrösse zwischengespeichert; YOLO-Eingabegrösse immer explizit (640 bzw. Tile-Grösse); `openvino` in `requirements.txt`; Spec-Datei bündelt OpenVINO und optional das vorab exportierte Modell. Fisheye-Bild auf einem CPU-Kern: ca. 45 s → 9–12 s |
| 2026-11-11 | PyInstaller-Spec auf Ordner-Build (onedir, `COLLECT`) ohne UPX umgestellt, damit die mehrere GB grosse CUDA-Version von PyTorch nicht bei jedem Start entpackt werden muss; Spec meldet beim Build, ob PyTorch mit CUDA eingepackt wird; Anleitung um "NVIDIA-GPU nutzen" und "Exe erstellen" ergänzt |
| 2026-11-11 | Fiktives Beispielbild der Verwisch-Vorschau in Grauwerten statt farbig (passend zu den Bildtyp-Icons) |
| 2026-11-11 | GPU-FP16 über `quantize=16` statt des in neueren ultralytics-Versionen veralteten `half=True` (behebt die Warnung "'half' is deprecated"); ältere Versionen erhalten automatisch weiterhin `half=True` |
| 2026-11-11 | Bildliste sortierbar ("Sortieren nach": Name / Bildgrösse klein → gross / gross → klein, nach Anzahl Pixel); Bildgrösse wird hinter jedem Dateinamen angezeigt (nur Dateikopf gelesen, EXIF-Drehung berücksichtigt); Auswahl bleibt beim Umsortieren erhalten |
| 2026-11-18 | Neue Checkbox **"Bilder überschreiben"** (Standard: aus) unter "Speicherort": ersetzt die Originalbilder im Quellordner statt sie in den Speicherort zu kopieren; Speicherort ist dann gesperrt, eigene Warnung im Start-Dialog. Sicheres Speichern in beiden Modi: zuerst temporäre Datei `<name>.anonym-tmp<endung>`, dann Ersetzen in einem Schritt (`os.replace`), bei Fehlern bleibt die Zieldatei unversehrt; temporäre Dateien werden beim Durchsuchen ignoriert. Beim Überschreiben bleiben Bilder ohne erkannte Person unangetastet (sofern keine Belichtungs-/Metadaten-Änderung). Start im Kopiermodus mit Speicherort = Quellordner wird verhindert. Zusammenfassung nennt Anzahl gespeicherter bzw. ersetzter und unveränderter Bilder |
| 2026-11-18 | Scrollbarer Hauptbereich (`ScrollableFrame`): beim Verkleinern des Fensters erscheinen bei Bedarf senkrechte/waagrechte Scrollbalken, Mausrad scrollt (ausser über Bildliste und Auswahlfeld, die selbst scrollen; Shift + Mausrad waagrecht); Status, Fortschritt und "Start" als feste Fusszeile immer sichtbar; Startgrösse an den Bildschirm angepasst (höchstens 840 × 1020), Mindestgrösse auf 420 × 300 gesenkt |
| 2026-11-18 | **Bildausrichtung korrigiert** (Fotos waren in RiSCAN um 90° gedreht): Pixel werden nicht mehr gemäss EXIF-Orientation gedreht gespeichert, das Orientation-Tag bleibt erhalten (EXIF und XMP), gespeicherte Abmessungen wie im Original. Neu in `image_io.py`: `read_image_with_orientation()`, `to_upright()`/`from_upright()`, Parameter `orientation` in `write_image_with_meta()` (Standard 1 = bisheriges Verhalten), minimales EXIF mit nur Orientation, wenn "Metadaten übernehmen" aus ist. Personenerkennung auf aufgerichteter Kopie, Maske zurückgedreht (alle Bildtypen). Vorschaubild in der Liste zeigt Hochformat-Fotos richtig herum |
| 2026-11-18 | Checkbox "Bilder überschreiben" umbenannt in **"Bilder im Quellordner ersetzen"** (inkl. Hinweisen beim Start und Hinweistext neben der Checkbox) |

*Dieser Änderungsverlauf wird bei künftigen Anpassungen am Code fortgeführt.*
