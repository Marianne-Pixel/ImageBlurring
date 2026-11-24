# Personen-Anonymisierung (Theta Z1, Fisheye & normale Bilder)

Tool zum automatischen Unkenntlichmachen von Personen in Bildern mittels
YOLO11-Segmentierung und wählbarem Verwisch-Effekt (Gaussian Blur oder
Pixelierung). Unterstützt normale, kaum verzerrte Fotos,
Equirectangular-Panoramen der Ricoh Theta Z1 sowie Fisheye-Bilder
(zirkularer Bildkreis). Die bearbeiteten Bilder
werden standardmässig an einem frei wählbaren Speicherort abgelegt, die
Originale bleiben unverändert. Optional können die Originale direkt
überschrieben werden (Checkbox "Bilder im Quellordner ersetzen").

## Installation

Python 3.9+ empfohlen.

```bash
cd personen_anonymisierer
pip install -r requirements.txt
```

Das YOLO11-Segmentierungsmodell (`yolo11s-seg.pt`) wurde am 20.10.2026 von Ultralytics heruntergeladen.
Die Datei liegt lokal und wird beim Ausführen des Tools verwendet.

Das Tool wählt das schnellste verfügbare Rechenwerk automatisch: eine
NVIDIA-GPU mit CUDA (halbe Genauigkeit), sonst OpenVINO auf der CPU (etwa
2.5-mal schneller als PyTorch, Paket `openvino` in `requirements.txt`),
sonst PyTorch auf der CPU. Beim allerersten Start ohne GPU wird das Modell
dafür einmalig in wenigen Sekunden umgewandelt.

Für eine NVIDIA-GPU muss PyTorch mit CUDA installiert sein (unter Windows
installiert `pip` standardmässig die CPU-Version), siehe
[ANLEITUNG.md](ANLEITUNG.md#nvidia-gpu-nutzen).

## Start

```bash
python main.py
```

Als Exe: `pyinstaller ImageBlurring_YOLO11.spec` erzeugt den Ordner
`dist\ImageBlurring_YOLO11\` (ganzen Ordner weitergeben), Details siehe
[ANLEITUNG.md](ANLEITUNG.md#exe-erstellen-pyinstaller).

## Bedienung

Auf kleinen Bildschirmen das Fenster einfach verkleinern: Es erscheinen
Scrollbalken (auch Mausrad), der "Start"-Knopf bleibt unten immer sichtbar.

1. **Durchsuchen…** (bei "Ordner") klicken und den Ordner mit den
   Quellbildern auswählen. Der Ordner wird (inkl. Unterordner) nach `.jpg`,
   `.jpeg` und `.png` durchsucht.
2. **Durchsuchen…** (bei "Speicherort") klicken und den Ordner auswählen,
   in dem die bearbeiteten Bilder gespeichert werden sollen. Die
   Unterordner-Struktur des Quellordners wird dabei am Speicherort
   nachgebildet. Alternativ die Checkbox **"Bilder im Quellordner ersetzen"**
   aktivieren (Standard: aus): Dann werden die Originalbilder ersetzt und
   kein Speicherort benötigt.
3. In der Liste die gewünschten Bilder auswählen (Mehrfachauswahl mit
   `Strg`/`Shift`-Klick, oder "Alle auswählen"). Über "Sortieren nach"
   lässt sich die Liste nach Name oder Bildgrösse ordnen. Bei Auswahl eines einzelnen
   Bildes erscheint rechts daneben eine kleine Vorschau.
4. Unter **"Verwisch-Einstellungen"** die Methode wählen (**Gaussian Blur**
   oder **Pixelieren/Mosaik**) und die Stärke per Regler einstellen. Die
   Vorschau rechts zeigt den Effekt live an einem kleinen, fiktiven
   Beispielbild.
5. Unter **"Belichtungskorrektur"** die Korrektur aktivieren und die
   beiden unabhängigen Regler **"Schatten aufhellen"** und
   **"Lichter abdunkeln"** (je 0–100) einstellen: 0 = kein Effekt,
   100 = maximale Wirkung. Die Regler wirken unabhängig voneinander.
6. Unter **"Bildtyp"** per Radio-Button **Equirectangular** (Standard),
   **Fisheye** oder **Normale Perspektive** wählen.
7. Unter **"Metadaten (nur JPEG)"** festlegen, ob EXIF/XMP/Farbprofil
   übernommen werden (Standard: ja, nötig für die 360°-Darstellung) und
   ob die GPS-Position entfernt werden soll (Standard: nein).
8. **Start** klicken. Es erscheint eine Sicherheitsabfrage mit der Anzahl
   der ausgewählten Bilder ("Es werden X Bild(er) bearbeitet."), da
   gleichnamige Dateien im Speicherort überschrieben werden ("Ja" =
   starten, "Nein"/"Abbrechen" = abbrechen).
9. Fortschritt wird unten angezeigt, darunter eine Zeitabschätzung
   (verstrichene Zeit, Restzeit, voraussichtliches Ende); am Ende erscheint eine
   Zusammenfassung (verwischte Bereiche/Bilder, Anzahl auf-/abgehellter
   Bilder, inkl. eventueller Fehler pro Bild).

## Funktionsweise

### Personendetektion
Beide Bildarten nutzen dieselbe YOLO11-**Segmentierung** (nicht nur eine
Bounding-Box) über `PersonDetector.detect_masks_normal()` in `detector.py`.
Dadurch steht pro erkannter Person die tatsächliche Silhouette zur
Verfügung, und es wird beim Verwischen nur diese unregelmässige Form
getroffen — nicht ein Rechteck oder Oval, das unnötig viel Hintergrund
mitverwischen würde.

### Normale Bilder
`processing.build_mask_from_polygons()` erzeugt aus den erkannten
Silhouetten eine Maske mit nur wenigen Pixeln Sicherheitsrand;
`processing.blur_mask()` wendet darauf die gewählte Methode
(**Gaussian Blur** oder **Pixelieren/Mosaik**, mit im GUI einstellbarer
Stärke) an, mit weichem Übergang am Rand. Standardmässig wird die Stärke
pro Person an deren Grösse angepasst, damit auch nahe, grosse Personen
zuverlässig unkenntlich werden.

### Equirectangular-Bilder (Theta Z1)
Wie zuvor: Panorama in überlappende Tiles zerlegen, Segmentierung pro Tile,
Silhouetten zurück ins Panorama-Koordinatensystem projizieren, dann
Verwischen über die resultierende Maske.

### Fisheye-Bilder
Analog zu den Panoramen: Bildkreis in 9 überlappende, entzerrte Tiles
schneiden, alle Tiles in 0°/90° segmentieren und Tiles mit Treffern
zusätzlich in 180°/270° (für gedreht montierte Kameras), Silhouetten exakt ins Fisheye-Bild zurückprojizieren.
Bildwinkel (Standard 200°) und Abbildungsfunktion (Standard
equidistant) lassen sich oben in `detector.py` anpassen.

### Belichtungskorrektur
Tonwertkurve auf dem L-Kanal (Helligkeit) im LAB-Farbraum: Schatten werden
angehoben, sehr helle Bereiche komprimiert. Zwei **unabhängige** Regler im
GUI — **"Schatten aufhellen"** und **"Lichter abdunkeln"** (je 0–100) —
steuern die beiden Effekte getrennt voneinander. Eine Sicherheitsprüfung
verhindert, dass das Bild dadurch insgesamt zu hell wird.

### Abschluss-Statistik
Nach der Verarbeitung zeigt das GUI eine Zusammenfassung: Anzahl verwischter
Bereiche und Anzahl betroffener Bilder, sowie (falls aktiviert) wie viele
Bilder auf- bzw. abgehellt wurden.

Details siehe [ANLEITUNG.md](ANLEITUNG.md).

### Metadaten
Bei JPEG werden EXIF, XMP (inkl. der 360°-Angaben der Theta Z1) und das
Farbprofil übernommen. Eingebettete Vorschaubilder werden immer entfernt,
da sie die Personen unverwischt zeigen würden. Optional lässt sich die
GPS-Position entfernen.

Die Bildgeometrie bleibt unverändert: Hochformat-Fotos, die die Kamera
quer mit EXIF-Tag "Orientation" speichert, werden auch quer mit demselben
Tag gespeichert. So erhalten Programme wie RiSCAN dieselbe Geometrie wie
beim Original.

## Hinweis

Standardmässig werden die Originaldateien **nicht verändert**. Die
bearbeiteten Bilder werden unter dem gewählten Speicherort abgelegt (mit
derselben relativen Unterordner-Struktur wie im Quellordner). Sind dort
bereits Dateien mit gleichem Namen vorhanden, werden diese überschrieben.

Mit **"Bilder im Quellordner ersetzen"** werden die Originale ersetzt (nicht
rückgängig zu machen). Jedes Bild wird zuerst in eine temporäre Datei
geschrieben und ersetzt das Original erst nach fehlerfreiem Speichern.
Bilder ohne erkannte Person bleiben dabei unangetastet.
