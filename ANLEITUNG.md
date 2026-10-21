# Anleitung: Personen-Anonymisierung (Theta Z1 & normale Bilder)

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
Form), nicht als Rechteck oder Oval. Es unterstützt zwei Bildarten:

| Bildart | Checkbox "Theta Z1" | Vorgehen |
|---|---|---|
| Normale Fotos (kaum verzerrt) | deaktiviert | Direkte Segmentierung im Bild |
| Equirectangular-Panoramen (Ricoh Theta Z1, 360°) | **aktiviert (Standard)** | Bild wird in Tiles zerlegt, Segmentierung pro Tile, Rückprojektion der Silhouetten ins Panorama |

Zusätzlich kann optional eine automatische Belichtungskorrektur (Schatten
aufhellen, Lichter abdunkeln) angewendet werden, deren beide Effekte über
zwei **unabhängige** Regler ("Schatten aufhellen" / "Lichter abdunkeln")
separat gesteuert werden. Am Ende der Verarbeitung zeigt das Tool eine
Zusammenfassung mit der Anzahl verwischter Bereiche/Bilder sowie der
Anzahl auf- bzw. abgehellter Bilder.

**Wichtig:** Die Originaldateien werden **nicht verändert**. Die bearbeiteten
Bilder werden unter einem separat wählbaren Speicherort abgelegt (mit
derselben relativen Unterordner-Struktur wie im Quellordner). Sind dort
bereits Dateien mit gleichem Namen vorhanden, werden diese überschrieben —
vor dem Start erscheint deshalb eine Sicherheitsabfrage.

---

## Voraussetzungen

- Python 3.9 oder neuer

- Optional: NVIDIA-GPU mit CUDA für deutlich schnellere Verarbeitung
  (funktioniert aber auch ohne GPU, nur langsamer)

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

## Programm starten

```bash
python main.py
```

Es öffnet sich das GUI-Fenster "Personen-Anonymisierung – Theta Z1 / Normale
Bilder".

---

## Bedienung Schritt für Schritt

1. **Ordner mit den Quellbildern auswählen**
   Bei der Zeile **"Ordner"** auf **"Durchsuchen…"** klicken und den Ordner
   mit den Bildern wählen. Das Tool durchsucht den Ordner **inklusive
   Unterordner** nach `.jpg`, `.jpeg` und `.png` (Gross-/Kleinschreibung
   wird ignoriert).

2. **Speicherort auswählen**
   Bei der neuen Zeile **"Speicherort"** direkt darunter auf
   **"Durchsuchen…"** klicken und den Ordner wählen, in dem die bearbeiteten
   Bilder abgelegt werden sollen. Dies muss ein anderer Ordner sein können
   als der Quellordner (z. B. ein separater "Ausgabe"-Ordner) — die
   Originaldateien werden dadurch nicht verändert.

3. **Bilder auswählen**
   Die gefundenen Bilder erscheinen als Liste (Pfad relativ zum gewählten
   Quellordner). Mehrfachauswahl ist möglich:
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

   Rechts daneben zeigt eine **Live-Vorschau an einem kleinen, fiktiven
   Beispielbild** (keinem echten Foto) sofort, wie der gewählte Effekt bei
   der aktuellen Stärke aussieht. So lässt sich die Einstellung testen,
   bevor echte Bilder verarbeitet werden.

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
   Checkbox **"Bilder stammen von der Ricoh Theta Z1 (Equirectangular /
   360°)"** — ist **standardmässig aktiviert**:
   - **Aktiviert (Standard):** Panorama-Verarbeitung mit Tile-Zerlegung
   - **Deaktiviert:** direkte Verarbeitung als normales Foto

   Für normale (nicht so stark verzerrte) Fotos muss die Checkbox also
   aktiv **deaktiviert** werden. Diese Einstellung gilt für **alle** in
   diesem Durchgang ausgewählten Bilder. Sollen sowohl Theta-Z1- als auch
   normale Bilder verarbeitet werden, müssen zwei separate Durchgänge
   gestartet werden (je einmal mit und einmal ohne Häkchen).

7. **Start**
   Unten rechts auf **"Start"** klicken. Wurde noch kein Speicherort
   ausgewählt, erscheint zuerst ein Hinweis, dies nachzuholen. Andernfalls
   erscheint die Sicherheitsabfrage mit der Anzahl der ausgewählten Bilder,
   z. B.:

   > *"Achtung. Falls im Speicherort bereits Dateien mit gleichem Namen
   > vorhanden sind, werden diese überschrieben. Es werden 12 Bild(er)
   > bearbeitet. Soll der Prozess gestartet werden?"*

   - **Ja** → Verarbeitung startet
   - **Nein** → Abbrechen
   - **Abbrechen** → Abbrechen

8. **Fortschritt & Ergebnis**
   Während der Verarbeitung zeigt die Statuszeile das aktuell bearbeitete
   Bild sowie eine Fortschrittsleiste. Am Ende erscheint eine
   Zusammenfassung mit:
   - Anzahl verarbeiteter Bilder
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
| Speicherort für bearbeitete Bilder | Zeile "Speicherort" (unter "Ordner") | beliebiger Ordner (Pflichtfeld vor Start) | leer (muss gewählt werden) |
| Verwisch-Methode | Verwisch-Einstellungen → Methode | Gaussian Blur / Pixelieren (Mosaik) | Gaussian Blur |
| Verwisch-Stärke | Verwisch-Einstellungen → Regler | 1 (schwach) – 30 (sehr stark) | **22** |
| Belichtungskorrektur aktivieren | Belichtungskorrektur → Checkbox | an / aus | aus |
| Schatten aufhellen (Intensität) | Belichtungskorrektur → Regler | 0 (kein Effekt) – 100 (maximal) | **22** |
| Lichter abdunkeln (Intensität) | Belichtungskorrektur → Regler | 0 (kein Effekt) – 100 (maximal) | **22** |
| Bildtyp | Bildtyp → Checkbox "Theta Z1 (Equirectangular)" | Theta Z1 (Equirectangular) / normal | **aktiviert (Theta Z1)** |

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
→ Ohne GPU ist v. a. die Panorama-Verarbeitung (16 Tiles × Detektion pro
Bild) rechenintensiv. Kleineres Modell (`yolo11n.pt`) oder GPU verwenden.

**Personen werden nicht erkannt**
→ `conf_threshold` in `main.py` senken (z. B. auf `0.25`), oder ein
grösseres Segmentierungsmodell verwenden (`yolo11m-seg.pt` oder
`yolo11l-seg.pt`) für höhere Erkennungsgenauigkeit (auf Kosten der
Geschwindigkeit).

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
→ Stärke des Verwisch-Reglers erhöhen, oder auf "Pixelieren (Mosaik)"
wechseln — diese Methode entfernt Details tendenziell zuverlässiger als
Gaussian Blur bei gleicher Stärke.

**Die Bild-Vorschau neben der Liste erscheint nicht**
→ Es ist genau ein Bild in der Liste auszuwählen (bei Mehrfachauswahl wird
nur die Anzahl angezeigt). Bei sehr grossen oder beschädigten Dateien kann
das Laden der Vorschau fehlschlagen; es erscheint dann der Hinweis
"Vorschau nicht verfügbar", die Verarbeitung selbst ist davon nicht
betroffen.

**"Kein Speicherort" beim Klick auf Start**
→ Es muss zuerst über die Zeile "Speicherort" ein Zielordner für die
bearbeiteten Bilder ausgewählt werden (separat von der Zeile "Ordner", die
nur die Quellbilder festlegt). Ohne gewählten Speicherort kann die
Verarbeitung nicht gestartet werden.

**Ich finde meine bearbeiteten Bilder nicht**
→ Sie liegen nicht mehr im Quellordner, sondern im gewählten
**Speicherort**, mit derselben relativen Unterordner-Struktur wie im
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
- Es wird nur eine flache Prüfung auf Namenskollisionen im Speicherort
  vorgenommen: Existiert dort bereits eine Datei mit demselben relativen
  Pfad, wird sie beim Speichern überschrieben (siehe Sicherheitsabfrage
  vor dem Start). Ein Backup der bearbeiteten Bilder wird nicht automatisch
  angelegt.

---

## Projektstruktur

```
personen_anonymisierer/
├── main.py              # GUI, Ablaufsteuerung, Dateiverwaltung
├── detector.py           # YOLO11-Detektion, Equirectangular-Tiling & Rückprojektion
├── processing.py         # Gaussian Blur, Belichtungskorrektur
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

*Dieser Änderungsverlauf wird bei künftigen Anpassungen am Code fortgeführt.*
