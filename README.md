# Personen-Anonymisierung (Theta Z1 & normale Bilder)

Tool zum automatischen Unkenntlichmachen von Personen in Bildern mittels
YOLO11-Segmentierung und wählbarem Verwisch-Effekt (Gaussian Blur oder
Pixelierung). Unterstützt sowohl normale, kaum verzerrte Fotos als auch
Equirectangular-Panoramen der Ricoh Theta Z1. Die bearbeiteten Bilder
werden an einem frei wählbaren Speicherort abgelegt, die Originale bleiben
unverändert.

## Installation

Python 3.9+ empfohlen.

```bash
cd personen_anonymisierer
pip install -r requirements.txt
```

Beim ersten Start wird das YOLO11-Segmentierungsmodell (`yolo11s-seg.pt`)
automatisch von Ultralytics heruntergeladen — dafür ist beim ersten Mal eine
Internetverbindung nötig. Danach liegt die Datei lokal und wird
wiederverwendet.

Falls eine NVIDIA-GPU mit CUDA vorhanden ist, nutzt `ultralytics`
automatisch die GPU (deutlich schneller). Ohne GPU läuft es auch auf der
CPU, nur langsamer.

## Start

```bash
python main.py
```

## Bedienung

1. **Durchsuchen…** (bei "Ordner") klicken und den Ordner mit den
   Quellbildern auswählen. Der Ordner wird (inkl. Unterordner) nach `.jpg`,
   `.jpeg` und `.png` durchsucht.
2. **Durchsuchen…** (bei "Speicherort") klicken und den Ordner auswählen,
   in dem die bearbeiteten Bilder gespeichert werden sollen. Die
   Unterordner-Struktur des Quellordners wird dabei am Speicherort
   nachgebildet.
3. In der Liste die gewünschten Bilder auswählen (Mehrfachauswahl mit
   `Strg`/`Shift`-Klick, oder "Alle auswählen"). Bei Auswahl eines einzelnen
   Bildes erscheint rechts daneben eine kleine Vorschau.
4. Unter **"Verwisch-Einstellungen"** die Methode wählen (**Gaussian Blur**
   oder **Pixelieren/Mosaik**) und die Stärke per Regler einstellen. Die
   Vorschau rechts zeigt den Effekt live an einem kleinen, fiktiven
   Beispielbild.
5. Unter **"Belichtungskorrektur"** die Korrektur aktivieren und die
   beiden unabhängigen Regler **"Schatten aufhellen"** und
   **"Lichter abdunkeln"** (je 0–100) einstellen: 0 = kein Effekt,
   100 = maximale Wirkung. Die Regler wirken unabhängig voneinander.
6. Checkbox **"Theta Z1 (Equirectangular)"** ist standardmässig
   **aktiviert**. Für normale (nicht so stark verzerrte) Fotos die
   Checkbox deaktivieren.
7. **Start** klicken. Es erscheint eine Sicherheitsabfrage mit der Anzahl
   der ausgewählten Bilder ("Es werden X Bild(er) bearbeitet."), da
   gleichnamige Dateien im Speicherort überschrieben werden ("Ja" =
   starten, "Nein"/"Abbrechen" = abbrechen).
8. Fortschritt wird unten angezeigt; am Ende erscheint eine
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
Stärke) an, mit weichem Übergang am Rand.

### Equirectangular-Bilder (Theta Z1)
Wie zuvor: Panorama in überlappende Tiles zerlegen, Segmentierung pro Tile,
Silhouetten zurück ins Panorama-Koordinatensystem projizieren, dann
Verwischen über die resultierende Maske.

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

## Hinweis

Die Originaldateien werden **nicht verändert**. Die bearbeiteten Bilder
werden unter dem gewählten Speicherort abgelegt (mit derselben relativen
Unterordner-Struktur wie im Quellordner). Sind dort bereits Dateien mit
gleichem Namen vorhanden, werden diese überschrieben.
