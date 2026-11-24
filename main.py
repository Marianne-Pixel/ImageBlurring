"""
main.py
GUI-Tool zur Anonymisierung von Personen in Bildern:
  - normale Fotos
  - Equirectangular-Panoramen der Ricoh Theta Z1
  - Fisheye-Bilder (zirkularer Bildkreis)

Ablauf:
  1. Ordner mit den Quellbildern auswählen -> wird nach JPEG/JPG/PNG durchsucht
  2. Speicherort für die bearbeiteten Bilder auswählen
  3. Gefundene Bilder erscheinen als Liste (Mehrfachauswahl möglich, Anzahl
     gefundener/ausgewählter Bilder wird angezeigt),
     daneben eine kleine Vorschau des in der Liste ausgewählten Bildes
  4. Verwisch-Methode (Gaussian Blur / Pixelieren) und Stärke wählen,
     mit Live-Vorschau an einem kleinen fiktiven Beispielbild
  5. Belichtungskorrektur optional aktivieren, Schatten- und Lichter-
     Korrektur über zwei unabhängige Regler einstellen
  6. Bildtyp wählen (Radio-Buttons: Equirectangular / Fisheye / normale
     Perspektive) – standardmässig ist "Equirectangular" ausgewählt
  7. "Start" klicken -> Bestätigungsdialog (inkl. Anzahl gewählter Bilder)
     -> Verarbeitung
  7b. Optional: Metadaten (EXIF/XMP/ICC) übernehmen, GPS entfernen
  8. Am Ende: Zusammenfassung mit Anzahl verwischter Bereiche/Bilder sowie
     Anzahl auf-/abgehellter Bilder

Standardmässig werden die bearbeiteten Bilder NICHT überschrieben, sondern
unter dem gewählten Speicherort abgelegt (mit derselben relativen
Unterordner-Struktur wie im Quellordner). Existieren dort bereits Dateien
mit gleichem Namen, werden diese überschrieben.
Mit der Checkbox "Bilder im Quellordner ersetzen" werden stattdessen die
Originalbilder im Quellordner ersetzt (kein Speicherort nötig).

Start: python main.py
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

SUPPORTED_EXT = (".jpg", ".jpeg", ".png")

# Kennzeichen für temporäre Dateien beim Speichern (siehe save_image_safely).
# Solche Dateien werden beim Durchsuchen eines Ordners ignoriert.
TEMP_MARKER = ".anonym-tmp"

# Sortiermöglichkeiten der Bildliste (Anzeigetext -> interner Schlüssel)
SORT_OPTIONS = {
    "Name (A–Z)": "name",
    "Bildgrösse (klein → gross)": "size_asc",
    "Bildgrösse (gross → klein)": "size_desc",
}


def read_image_size(path):
    """
    Liest nur den Dateikopf und liefert (Breite, Höhe) in Pixeln, wie das
    Bild angezeigt wird (EXIF-Drehung um 90°/270° berücksichtigt).
    None, wenn die Datei nicht lesbar ist.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            try:
                if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                    w, h = h, w
            except Exception:
                pass
            return w, h
    except Exception:
        return None


def same_folder(a, b):
    """True, wenn a und b auf denselben Ordner zeigen (Gross-/Kleinschreibung
    und Schreibweise unter Windows berücksichtigt)."""
    if not a or not b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def save_image_safely(dest_path, img, meta, strip_gps, orientation=1):
    """
    Speichert ein Bild so, dass eine bereits vorhandene Datei (z.B. das
    Original beim Überschreiben) nie halb geschrieben zurückbleibt:
      1. in eine temporäre Datei im selben Ordner schreiben
         ("<name>.anonym-tmp<endung>", gleiche Endung, damit das richtige
         Format gewählt wird),
      2. erst wenn das fehlerfrei geklappt hat, die Zieldatei per
         os.replace in einem Schritt ersetzen.
    Schlägt etwas fehl, bleibt die Zieldatei unverändert und die temporäre
    Datei wird entfernt. `orientation` = EXIF-Orientation der Pixel (wird
    beibehalten, siehe image_io). Rückgabe: Liste von Metadaten-Warnungen.
    Wirft RuntimeError bei einem Fehler.
    """
    folder, name = os.path.split(dest_path)
    stem, ext = os.path.splitext(name)
    tmp_path = os.path.join(folder, f"{stem}{TEMP_MARKER}{ext}")
    try:
        ok, warnings = write_image_with_meta(tmp_path, img, meta, strip_gps=strip_gps,
                                             orientation=orientation)
        if not ok:
            raise RuntimeError("Bild konnte nicht gespeichert werden.")
        try:
            os.replace(tmp_path, dest_path)
        except OSError as e:
            raise RuntimeError(
                "Bild konnte nicht ersetzt werden (Datei schreibgeschützt oder in "
                f"einem anderen Programm geöffnet?): {e}"
            )
        return warnings
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# Ultralytics-Onlineprüfungen (Updates, Downloads) abschalten
os.environ["YOLO_OFFLINE"] = "True"

from detector import PersonDetector
from processing import build_mask_from_polygons, blur_mask, count_regions, auto_correct_exposure
from icons import projection_icons
from image_io import (
    read_image_with_orientation, read_metadata, write_image_with_meta,
    to_upright, from_upright,
)


def resource_path(name):
    """Pfad zu mitgelieferten Dateien, funktioniert im Skript und in der Exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)

class ScrollableFrame(ttk.Frame):
    """
    Bereich mit Scrollbalken, die nur erscheinen, wenn der Inhalt nicht ins
    Fenster passt (z.B. auf kleinen Bildschirmen oder wenn das Fenster mit
    der Maus verkleinert wird).

    Inhalt in `self.body` platzieren. Ist genug Platz vorhanden, wird
    `body` auf die volle Fenstergrösse gestreckt, sodass Elemente mit
    expand=True (z.B. die Bildliste) wie gewohnt mitwachsen.

    Mausrad: scrollt den Bereich, ausser der Mauszeiger steht über einem
    Element, das selbst scrollt (Bildliste, Auswahlfeld). Shift + Mausrad
    scrollt waagrecht.
    """

    # Elemente, die das Mausrad selbst verwenden
    _OWN_WHEEL_CLASSES = ("Listbox", "TCombobox", "Text", "Treeview")
    _SCROLL_STEP_PX = 20

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        bg = ttk.Style().lookup("TFrame", "background") or None
        self.canvas = tk.Canvas(
            self, highlightthickness=0, borderwidth=0,
            yscrollincrement=self._SCROLL_STEP_PX, xscrollincrement=self._SCROLL_STEP_PX,
            **({"background": bg} if bg else {}),
        )
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.vbar.grid_remove()
        self.hbar.grid_remove()
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self._layout_pending = False
        self.v_visible = False
        self.h_visible = False

        # Neu berechnen, wenn sich die Fenstergrösse oder der Platzbedarf
        # des Inhalts ändert
        self.bind("<Configure>", self._schedule_layout, add="+")
        self.body.bind("<Configure>", self._schedule_layout, add="+")

        for seq in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>",
                    "<Shift-Button-4>", "<Shift-Button-5>"):
            self.bind_all(seq, self._on_mousewheel, add="+")

    def _schedule_layout(self, _event=None):
        # Mehrere Configure-Ereignisse kurz hintereinander zusammenfassen
        if not self._layout_pending:
            self._layout_pending = True
            self.after_idle(self._update_layout)

    def _update_layout(self):
        self._layout_pending = False
        avail_w, avail_h = self.winfo_width(), self.winfo_height()
        if avail_w <= 1 or avail_h <= 1:
            return  # noch nicht angezeigt
        req_w, req_h = self.body.winfo_reqwidth(), self.body.winfo_reqheight()
        bar_w, bar_h = self.vbar.winfo_reqwidth(), self.hbar.winfo_reqheight()

        # Bedarf aus dem gesamten verfügbaren Platz bestimmen (ändert sich
        # nicht, wenn Scrollbalken ein-/ausgeblendet werden -> kein Flackern).
        # Ein Balken nimmt dem jeweils anderen Platz weg, daher zweimal prüfen.
        need_v = need_h = False
        for _ in range(2):
            need_v = req_h > avail_h - (bar_h if need_h else 0)
            need_h = req_w > avail_w - (bar_w if need_v else 0)

        view_w = avail_w - (bar_w if need_v else 0)
        view_h = avail_h - (bar_h if need_h else 0)
        width, height = max(req_w, view_w), max(req_h, view_h)
        self.canvas.itemconfigure(self._window, width=width, height=height)
        self.canvas.configure(scrollregion=(0, 0, width, height))

        if need_v != self.v_visible:
            self.vbar.grid() if need_v else self.vbar.grid_remove()
            self.v_visible = need_v
        if need_h != self.h_visible:
            self.hbar.grid() if need_h else self.hbar.grid_remove()
            self.h_visible = need_h
        if not need_v:
            self.canvas.yview_moveto(0)
        if not need_h:
            self.canvas.xview_moveto(0)

    def _on_mousewheel(self, event):
        widget = event.widget
        if not isinstance(widget, tk.Misc):
            return  # z.B. Aufklappliste eines Auswahlfelds
        try:
            if widget.winfo_toplevel() is not self.winfo_toplevel():
                return  # anderes Fenster (z.B. Bestätigungsdialog)
            if widget.winfo_class() in self._OWN_WHEEL_CLASSES:
                return  # Element scrollt selbst
            # Nur reagieren, wenn die Maus über dem scrollbaren Bereich ist
            # (nicht z.B. über der festen Statuszeile)
            w = widget
            while w is not None and w is not self:
                w = w.master
            if w is None:
                return
        except tk.TclError:
            return

        horizontal = bool(event.state & 0x0001)  # Shift gedrückt
        if event.num == 4:
            steps = -1
        elif event.num == 5:
            steps = 1
        elif sys.platform == "darwin":
            steps = -event.delta
        else:
            steps = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)

        if horizontal:
            if self.h_visible:
                self.canvas.xview_scroll(steps * 3, "units")
        elif self.v_visible:
            self.canvas.yview_scroll(steps * 3, "units")


class ConfirmDialog(tk.Toplevel):
    """Eigener 3-Knopf-Dialog mit den Beschriftungen 'Ja' / 'Nein' / 'Abbrechen'."""

    def __init__(self, parent, title, message):
        super().__init__(parent)
        self.title(title)
        self.result = "cancel"
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        frm = ttk.Frame(self, padding=20)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text=message, wraplength=380, justify="left").pack(pady=(0, 15))

        btn_frame = ttk.Frame(frm)
        btn_frame.pack()

        ttk.Button(btn_frame, text="Ja", width=12, command=self._yes).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Nein", width=12, command=self._no).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Abbrechen", width=12, command=self._cancel).grid(row=0, column=2, padx=5)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update_idletasks()
        self.geometry(f"+{parent.winfo_rootx() + 40}+{parent.winfo_rooty() + 40}")
        self.wait_window(self)

    def _yes(self):
        self.result = "yes"
        self.destroy()

    def _no(self):
        self.result = "no"
        self.destroy()

    def _cancel(self):
        self.result = "cancel"
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Personen-Anonymisierung YOLO11 – Theta Z1 / Fisheye / Normale Bilder")
        # Startgrösse an den Bildschirm anpassen (kleine Bildschirme: Fenster
        # passt hinein, der Inhalt lässt sich dann scrollen)
        width = min(840, self.winfo_screenwidth() - 40)
        height = min(1020, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}")
        self.minsize(420, 300)

        self.folder_path = tk.StringVar(value="")
        self.output_folder_path = tk.StringVar(value="")
        # False (Standard): bearbeitete Bilder in den Speicherort kopieren;
        # True: Originalbilder im Quellordner ersetzen
        self.overwrite = tk.BooleanVar(value=False)
        # Bildtyp: "equirect" (Theta Z1, 360°), "fisheye" oder "normal"
        self.image_type = tk.StringVar(value="equirect")
        self.auto_correct = tk.BooleanVar(value=False)
        self.shadow_intensity = tk.IntVar(value=22)
        self.highlight_intensity = tk.IntVar(value=22)
        self.blur_method = tk.StringVar(value="gaussian")
        self.blur_strength = tk.IntVar(value=22)
        self.relative_strength = tk.BooleanVar(value=True)
        self.keep_metadata = tk.BooleanVar(value=True)
        self.strip_gps = tk.BooleanVar(value=False)
        self.image_paths = []
        self.image_infos = []  # je Bild: {"path", "rel", "size"} (size = (B, H) oder None)
        self.sort_choice = tk.StringVar(value=next(iter(SORT_OPTIONS)))
        self.detector = None  # wird erst beim Start geladen (spart Startzeit)
        self._eta = None       # Zustand der Zeitabschätzung (nur im GUI-Thread benutzt)
        self._eta_job = None   # ID des laufenden after()-Timers
        self._busy = False     # True während der Verarbeitung

        self._thumb_photo = None      # Referenzen halten (sonst Garbage Collection)
        self._preview_photo = None
        self._sample_img, self._sample_polygon = self._create_fictive_sample()

        self._build_ui()
        self.update_preview()

    # ------------------------------------------------------------ UI ----
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Status, Fortschritt und Start bleiben immer unten sichtbar; alles
        # andere liegt in einem Bereich, der bei zu kleinem Fenster scrollt.
        # (Fusszeile zuerst packen, damit sie beim Verkleinern nicht
        # verdrängt wird.)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", **pad)
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x")
        self.scroll_area = ScrollableFrame(self)
        self.scroll_area.pack(side="top", fill="both", expand=True)
        body = self.scroll_area.body

        # --- Ordner auswählen ---
        top = ttk.Frame(body)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Ordner:").pack(side="left")
        ttk.Entry(top, textvariable=self.folder_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(top, text="Durchsuchen…", command=self.select_folder).pack(side="left")

        # --- Speicherort für die bearbeiteten Bilder auswählen ---
        output_row = ttk.Frame(body)
        output_row.pack(fill="x", **pad)

        ttk.Label(output_row, text="Speicherort:").pack(side="left")
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_folder_path, state="readonly")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.output_button = ttk.Button(output_row, text="Durchsuchen…", command=self.select_output_folder)
        self.output_button.pack(side="left")

        # --- Überschreiben statt kopieren ---
        overwrite_row = ttk.Frame(body)
        overwrite_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Checkbutton(
            overwrite_row, text="Bilder im Quellordner ersetzen",
            variable=self.overwrite, command=self._on_overwrite_change,
        ).pack(side="left")
        self.overwrite_hint = ttk.Label(overwrite_row, foreground="#555555")
        self.overwrite_hint.pack(side="left", padx=(10, 0))
        self._on_overwrite_change()

        # --- Bilderliste + kleine Vorschau daneben ---
        content = ttk.Frame(body)
        content.pack(fill="both", expand=True, **pad)

        list_frame = ttk.LabelFrame(content, text="Gefundene Bilder: 0")
        list_frame.pack(side="left", fill="both", expand=True)
        self.list_frame = list_frame  # Titel zeigt die Anzahl gefundener Bilder

        sort_row = ttk.Frame(list_frame)
        sort_row.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(sort_row, text="Sortieren nach:").pack(side="left")
        sort_box = ttk.Combobox(
            sort_row, textvariable=self.sort_choice, values=list(SORT_OPTIONS),
            state="readonly", width=26,
        )
        sort_box.pack(side="left", padx=6)
        sort_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_sort())

        list_inner = ttk.Frame(list_frame)
        list_inner.pack(fill="both", expand=True, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(list_inner, orient="vertical")
        self.listbox = tk.Listbox(
            list_inner, selectmode="extended", yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        sel_btns = ttk.Frame(list_frame)
        sel_btns.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(sel_btns, text="Alle auswählen", command=self.select_all).pack(side="left")
        ttk.Button(sel_btns, text="Auswahl aufheben", command=self.deselect_all).pack(side="left", padx=6)

        thumb_frame = ttk.LabelFrame(content, text="Vorschau")
        thumb_frame.pack(side="left", fill="y", padx=(10, 0))

        self.thumb_label = ttk.Label(
            thumb_frame, text="Keine Auswahl", anchor="center",
            width=26, justify="center"
        )
        self.thumb_label.pack(padx=10, pady=10)

        # --- Verwisch-Einstellungen: Methode, Stärke, Live-Vorschau ---
        blur_frame = ttk.LabelFrame(body, text="Verwisch-Einstellungen")
        blur_frame.pack(fill="x", **pad)

        blur_left = ttk.Frame(blur_frame)
        blur_left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        ttk.Label(blur_left, text="Methode:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Radiobutton(
            blur_left, text="Gaussian Blur", value="gaussian",
            variable=self.blur_method, command=self.update_preview
        ).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Radiobutton(
            blur_left, text="Pixelieren (Mosaik)", value="pixelate",
            variable=self.blur_method, command=self.update_preview
        ).grid(row=0, column=2, sticky="w", padx=6)

        ttk.Label(blur_left, text="Stärke:").grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.strength_scale = ttk.Scale(
            blur_left, from_=1, to=30, orient="horizontal",
            variable=self.blur_strength, command=self._on_strength_change
        )
        self.strength_scale.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=(10, 2))
        blur_left.columnconfigure(1, weight=1)

        self.strength_value_label = ttk.Label(blur_left, text=str(self.blur_strength.get()))
        self.strength_value_label.grid(row=1, column=3, sticky="w", padx=(4, 0))

        ttk.Checkbutton(
            blur_left,
            text="Stärke an Personengrösse anpassen (empfohlen)",
            variable=self.relative_strength,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        blur_right = ttk.Frame(blur_frame)
        blur_right.pack(side="right", padx=8, pady=8)
        self.blur_preview_label = ttk.Label(blur_right)
        self.blur_preview_label.pack()
        ttk.Label(blur_right, text="Vorschau", foreground="#555555").pack(pady=(4, 0))

        # --- Belichtungskorrektur: Aktivieren, Intensität ---
        exposure_frame = ttk.LabelFrame(body, text="Belichtungskorrektur")
        exposure_frame.pack(fill="x", **pad)

        ttk.Checkbutton(
            exposure_frame,
            text="Automatische Belichtungskorrektur aktivieren (Schatten aufhellen / Lichter abdunkeln)",
            variable=self.auto_correct,
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 8))

        ttk.Label(exposure_frame, text="Schatten aufhellen:").grid(row=1, column=0, sticky="w", padx=8)
        self.shadow_scale = ttk.Scale(
            exposure_frame, from_=0, to=100, orient="horizontal",
            variable=self.shadow_intensity, command=self._on_shadow_change
        )
        self.shadow_scale.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6)
        exposure_frame.columnconfigure(1, weight=1)

        self.shadow_value_label = ttk.Label(exposure_frame, text=str(self.shadow_intensity.get()))
        self.shadow_value_label.grid(row=1, column=3, sticky="w", padx=(4, 8))

        ttk.Label(exposure_frame, text="Lichter abdunkeln:").grid(row=2, column=0, sticky="w", padx=8)
        self.highlight_scale = ttk.Scale(
            exposure_frame, from_=0, to=100, orient="horizontal",
            variable=self.highlight_intensity, command=self._on_highlight_change
        )
        self.highlight_scale.grid(row=2, column=1, columnspan=2, sticky="ew", padx=6)

        self.highlight_value_label = ttk.Label(exposure_frame, text=str(self.highlight_intensity.get()))
        self.highlight_value_label.grid(row=2, column=3, sticky="w", padx=(4, 8))

        ttk.Label(
            exposure_frame,
            text="Beide Regler wirken unabhängig voneinander: 0 = keine\n"
                 "Korrektur, 100 = maximale Wirkung.",
            foreground="#555555", justify="left"
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 8))

        # --- Bildtyp ---
        options = ttk.LabelFrame(body, text="Bildtyp")
        options.pack(fill="x", **pad)

        type_row = ttk.Frame(options)
        type_row.pack(fill="x", padx=8, pady=4)
        # Schematische Graustufen-Icons (Würfelraum mit Gitter) je Projektion;
        # Referenzen halten, sonst entfernt der Garbage Collector die Bilder.
        self._type_icons = {
            key: ImageTk.PhotoImage(img) for key, img in projection_icons(48).items()
        }
        for value, label in (
            ("equirect", "Equirectangular"),
            ("fisheye", "Fisheye"),
            ("normal", "Normale Perspektive"),
        ):
            ttk.Radiobutton(
                type_row, text=label, value=value, variable=self.image_type,
                image=self._type_icons[value], compound="left",
            ).pack(side="left", padx=(0, 24))

        # --- Metadaten ---
        meta_frame = ttk.LabelFrame(body, text="Metadaten (nur JPEG)")
        meta_frame.pack(fill="x", **pad)

        ttk.Checkbutton(
            meta_frame,
            text="Metadaten übernehmen (EXIF, XMP inkl. 360°-Angaben, Farbprofil)",
            variable=self.keep_metadata,
            command=self._on_keep_metadata_change,
        ).pack(anchor="w", padx=8, pady=(4, 0))
        self.strip_gps_check = ttk.Checkbutton(
            meta_frame,
            text="GPS-Position entfernen",
            variable=self.strip_gps,
        )
        self.strip_gps_check.pack(anchor="w", padx=28, pady=(2, 4))

        # --- Status, Fortschritt, Start (feste Fusszeile, siehe oben) ---
        bottom.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Bereit.")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=10)

        self.start_btn = ttk.Button(bottom, text="Start", command=self.on_start_clicked)
        self.start_btn.grid(row=0, column=2, sticky="e")

        # Zeitabschätzung direkt unter dem Fortschrittsbalken
        self.eta_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.eta_var, foreground="#555555").grid(
            row=1, column=1, sticky="w", padx=10, pady=(4, 0)
        )

    # ------------------------------------------------ Ordner & Bilder ----
    def select_folder(self):
        folder = filedialog.askdirectory(title="Ordner mit Bildern auswählen")
        if not folder:
            return
        self.folder_path.set(folder)
        self.scan_images(folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Speicherort für bearbeitete Bilder auswählen")
        if not folder:
            return
        self.output_folder_path.set(folder)

    def scan_images(self, folder):
        self.status_var.set("Bilder werden gesucht…")
        self.update_idletasks()
        infos = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(SUPPORTED_EXT) and TEMP_MARKER not in f:
                    full = os.path.join(root, f)
                    infos.append({
                        "path": full,
                        "rel": os.path.relpath(full, folder),
                        "size": read_image_size(full),  # liest nur den Dateikopf
                    })
        self.image_infos = infos
        self.list_frame.config(text=f"Gefundene Bilder: {len(infos)}")
        self.listbox.selection_clear(0, tk.END)
        self.apply_sort()
        self._thumb_photo = None
        self.thumb_label.config(image="", text="Keine Auswahl")

    def apply_sort(self):
        """
        Sortiert die Bildliste gemäss Auswahl ("Name" oder "Bildgrösse" =
        Anzahl Pixel, Breite × Höhe) und baut die Listbox neu auf. Bereits
        ausgewählte Bilder bleiben ausgewählt.
        """
        selected = {self.image_paths[i] for i in self.listbox.curselection()}
        mode = SORT_OPTIONS.get(self.sort_choice.get(), "name")

        def name_key(info):
            return info["rel"].lower()

        if mode == "name":
            infos = sorted(self.image_infos, key=name_key)
        else:
            known = [i for i in self.image_infos if i["size"] is not None]
            unknown = sorted((i for i in self.image_infos if i["size"] is None), key=name_key)
            known.sort(key=name_key)  # bei gleicher Grösse nach Name
            known.sort(key=lambda i: (i["size"][0] * i["size"][1], i["size"][0]),
                       reverse=(mode == "size_desc"))
            infos = known + unknown  # unlesbare Dateien immer am Ende

        self.image_infos = infos
        self.image_paths = [i["path"] for i in infos]
        self.listbox.delete(0, tk.END)
        for idx, info in enumerate(infos):
            size = info["size"]
            size_txt = f"{size[0]} × {size[1]}" if size else "Grösse unbekannt"
            self.listbox.insert(tk.END, f"{info['rel']}   ({size_txt})")
            if info["path"] in selected:
                self.listbox.selection_set(idx)
        self._update_selection_status()

    def select_all(self):
        self.listbox.select_set(0, tk.END)
        self.on_listbox_select()

    def deselect_all(self):
        self.listbox.select_clear(0, tk.END)
        self.on_listbox_select()

    # -------------------------------------------------- Bildvorschau ----
    def _update_selection_status(self):
        """Zeigt unten in der Statuszeile die Anzahl ausgewählter Bilder."""
        if self._busy:
            return  # während der Verarbeitung zeigt die Statuszeile den Fortschritt
        count = len(self.listbox.curselection())
        self.status_var.set(f"{count} Bild ausgewählt" if count == 1 else f"{count} Bilder ausgewählt")

    def on_listbox_select(self, event=None):
        self._update_selection_status()
        sel = self.listbox.curselection()
        if len(sel) == 1:
            self.show_thumbnail(self.image_paths[sel[0]])
        elif len(sel) > 1:
            self._thumb_photo = None
            self.thumb_label.config(image="", text=f"{len(sel)} Bilder\nausgewählt")
        else:
            self._thumb_photo = None
            self.thumb_label.config(image="", text="Keine Auswahl")

    def show_thumbnail(self, path):
        try:
            img = ImageOps.exif_transpose(Image.open(path))  # Hochformat richtig anzeigen
            img.thumbnail((190, 150))
            self._thumb_photo = ImageTk.PhotoImage(img)
            self.thumb_label.config(image=self._thumb_photo, text="")
        except Exception:
            self._thumb_photo = None
            self.thumb_label.config(image="", text="Vorschau nicht\nverfügbar")

    # ------------------------------------------- Verwisch-Live-Vorschau ----
    @staticmethod
    def _create_fictive_sample():
        """
        Erzeugt ein kleines, frei erfundenes Beispielbild mit einer
        schematischen Person, um die Verwisch-Einstellungen zu demonstrieren.
        Das Silhouetten-Polygon bildet bewusst eine unregelmässige Kopf-
        Schulter-Form nach (kein Rechteck, kein Oval).
        """
        # Nur Grauwerte (alle drei Kanäle gleich), passend zu den Bildtyp-Icons
        w, h = 240, 160
        img = np.full((h, w, 3), 228, dtype=np.uint8)  # Hintergrund

        for x in range(0, w, 12):
            cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)
        for y in range(0, h, 12):
            cv2.line(img, (0, y), (w, y), (200, 200, 200), 1)

        cv2.rectangle(img, (88, 96), (152, h), (85, 85, 85), -1)              # Oberkörper
        cv2.ellipse(img, (120, 66), (30, 36), 0, 0, 360, (165, 165, 165), -1)  # Kopf
        cv2.circle(img, (109, 60), 4, (35, 35, 35), -1)                        # Augen
        cv2.circle(img, (131, 60), 4, (35, 35, 35), -1)
        cv2.ellipse(img, (120, 78), (10, 5), 0, 0, 180, (60, 60, 60), 2)       # Mund

        polygon = np.array([
            [120, 28], [138, 34], [147, 50], [148, 66], [146, 80],
            [151, 92], [154, 110], [156, 160], [84, 160], [86, 110],
            [89, 92], [94, 80], [92, 66], [93, 50], [102, 34],
        ], dtype=np.int32)

        return img, polygon

    def _on_strength_change(self, _value):
        self.blur_strength.set(int(round(float(_value))))
        self.strength_value_label.config(text=str(self.blur_strength.get()))
        self.update_preview()

    def update_preview(self, *_):
        method = self.blur_method.get()
        strength = self.blur_strength.get()
        mask = build_mask_from_polygons(self._sample_img.shape, [self._sample_polygon], margin_px=3)
        result = blur_mask(self._sample_img, mask, method=method, strength=strength)
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._preview_photo = ImageTk.PhotoImage(pil_img)
        self.blur_preview_label.config(image=self._preview_photo)

    def _on_overwrite_change(self):
        """Speicherort sperren, solange die Originale überschrieben werden."""
        if self.overwrite.get():
            self.output_entry.config(state="disabled")
            self.output_button.config(state="disabled")
            self.overwrite_hint.config(
                text="Originale werden überschrieben, der Speicherort wird nicht verwendet."
            )
        else:
            self.output_entry.config(state="readonly")
            self.output_button.config(state="normal")
            self.overwrite_hint.config(
                text="Aus: bearbeitete Bilder werden als Kopie im Speicherort abgelegt."
            )

    def _on_keep_metadata_change(self):
        state = "normal" if self.keep_metadata.get() else "disabled"
        self.strip_gps_check.config(state=state)

    def _on_shadow_change(self, _value):
        self.shadow_intensity.set(int(round(float(_value))))
        self.shadow_value_label.config(text=str(self.shadow_intensity.get()))

    def _on_highlight_change(self, _value):
        self.highlight_intensity.set(int(round(float(_value))))
        self.highlight_value_label.config(text=str(self.highlight_intensity.get()))

    # -------------------------------------------------------- Start ----
    def on_start_clicked(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Keine Auswahl", "Bitte mindestens ein Bild auswählen.")
            return

        overwrite = self.overwrite.get()
        count = len(selected_indices)

        if overwrite:
            message = (
                "Achtung. Die ausgewählten Originalbilder im Quellordner werden\n"
                "durch die bearbeiteten Bilder ersetzt. Das kann nicht rückgängig\n"
                "gemacht werden.\n"
                f"Es werden {count} Bild(er) bearbeitet.\n"
                "Soll der Prozess gestartet werden?"
            )
        else:
            if not self.output_folder_path.get():
                messagebox.showinfo(
                    "Kein Speicherort",
                    "Bitte zuerst einen Speicherort für die bearbeiteten Bilder auswählen\n"
                    "oder \"Bilder im Quellordner ersetzen\" aktivieren."
                )
                return
            if same_folder(self.output_folder_path.get(), self.folder_path.get()):
                # Sonst würden die Originale überschrieben, obwohl die
                # Checkbox "Bilder im Quellordner ersetzen" nicht aktiv ist.
                messagebox.showinfo(
                    "Speicherort = Quellordner",
                    "Der Speicherort ist derselbe Ordner wie der Quellordner, die\n"
                    "Originalbilder würden also überschrieben.\n\n"
                    "Bitte einen anderen Speicherort wählen oder\n"
                    "\"Bilder im Quellordner ersetzen\" aktivieren, wenn die Originale\n"
                    "ersetzt werden sollen."
                )
                return
            message = (
                "Achtung. Falls im Speicherort bereits Dateien mit gleichem Namen\n"
                "vorhanden sind, werden diese überschrieben.\n"
                f"Es werden {count} Bild(er) bearbeitet.\n"
                "Soll der Prozess gestartet werden?"
            )
        dlg = ConfirmDialog(self, "Achtung", message)
        if dlg.result != "yes":
            return

        selected_paths = [self.image_paths[i] for i in selected_indices]
        settings = {
            "image_type": self.image_type.get(),
            "do_correct": self.auto_correct.get(),
            "shadow_intensity": self.shadow_intensity.get(),
            "highlight_intensity": self.highlight_intensity.get(),
            "method": self.blur_method.get(),
            "strength": self.blur_strength.get(),
            "relative_strength": self.relative_strength.get(),
            "keep_metadata": self.keep_metadata.get(),
            "strip_gps": self.strip_gps.get(),
            "source_folder": self.folder_path.get(),
            "output_folder": self.output_folder_path.get(),
            "overwrite": overwrite,
        }

        self.start_btn.config(state="disabled")
        self._busy = True
        self.progress.config(maximum=len(selected_paths), value=0)
        self._eta_stop()
        self.eta_var.set("Restzeit: wird berechnet…")

        thread = threading.Thread(
            target=self.process_images, args=(selected_paths, settings), daemon=True
        )
        thread.start()

    # ----------------------------------------------------- Verarbeitung ----
    def process_images(self, paths, settings):
        image_type = settings["image_type"]
        do_correct = settings["do_correct"]
        shadow_intensity = settings["shadow_intensity"]
        highlight_intensity = settings["highlight_intensity"]
        method = settings["method"]
        strength = settings["strength"]
        source_folder = settings["source_folder"]
        output_folder = settings["output_folder"]
        relative_strength = settings["relative_strength"]
        keep_metadata = settings["keep_metadata"]
        strip_gps = settings["strip_gps"]
        overwrite = settings["overwrite"]

        if self.detector is None:
            # Beim allerersten Start ohne GPU wird das Modell einmalig für
            # OpenVINO umgewandelt (einige Sekunden), danach geht es sofort.
            self._set_status("Lade YOLO11-Segmentierungsmodell (beim ersten Start etwas länger)…")
            #self.detector = PersonDetector(model_path="yolo11s-seg.pt", conf_threshold=0.35)
            self.detector = PersonDetector(model_path=resource_path("yolo11s-seg.pt"), conf_threshold=0.35) #greift auf die mitgelieferte Datei im Exe-Ordner zu

        # Zeitmessung erst nach dem Laden des Modells starten, damit die
        # einmalige Ladezeit die Schätzung pro Bild nicht verfälscht.
        run_start = time.monotonic()
        self.after(0, lambda n=len(paths), t=run_start: self._eta_begin(n, t))

        errors = []
        total_regions = 0
        photos_with_blur = 0
        photos_brightened = 0
        photos_darkened = 0
        photos_unchanged = 0  # nur im Überschreib-Modus: Original bleibt unangetastet
        photos_saved = 0      # erfolgreich gespeichert (kopiert bzw. Original ersetzt)

        for idx, path in enumerate(paths, start=1):
            self._set_status(f"Bearbeite {idx}/{len(paths)}: {os.path.basename(path)}")
            try:
                # Pixel so lesen, wie die Kamera sie gespeichert hat (nicht
                # gemäss EXIF-Orientation gedreht). Das Ergebnis wird ebenso
                # gespeichert und behält das Orientation-Tag, damit z.B.
                # RiSCAN dieselbe Geometrie wie beim Original erhält.
                img, orientation = read_image_with_orientation(path)
                if img is None:
                    raise RuntimeError("Bild konnte nicht gelesen werden.")

                # Personenerkennung auf der aufgerichteten Ansicht (aufrechte
                # Personen werden zuverlässiger erkannt), Maske danach in die
                # gespeicherte Ausrichtung zurückdrehen.
                upright = to_upright(img, orientation)
                if image_type == "equirect":
                    mask = self.detector.detect_mask_equirect(upright)
                elif image_type == "fisheye":
                    mask = self.detector.detect_mask_fisheye(upright)
                else:
                    polygons = self.detector.detect_masks_normal(upright)
                    mask = build_mask_from_polygons(upright.shape, polygons)
                mask = from_upright(mask, orientation)

                regions = count_regions(mask)
                if regions > 0:
                    photos_with_blur += 1
                    total_regions += regions
                    img = blur_mask(img, mask, method=method, strength=strength,
                                    relative_strength=relative_strength)

                if do_correct:
                    img, mean_diff = auto_correct_exposure(
                        img, shadow_intensity=shadow_intensity, highlight_intensity=highlight_intensity
                    )
                    if mean_diff > 1.0:
                        photos_brightened += 1
                    elif mean_diff < -1.0:
                        photos_darkened += 1

                if overwrite:
                    # Nichts zu ändern (keine Person, keine Belichtungskorrektur,
                    # Metadaten bleiben wie sie sind): Original nicht neu
                    # speichern, sonst würde es ohne Nutzen erneut als JPEG
                    # komprimiert (Qualitätsverlust).
                    if regions == 0 and not do_correct and keep_metadata and not strip_gps:
                        photos_unchanged += 1
                        continue
                    dest_path = path
                else:
                    # Ziel-Pfad im gewählten Speicherort, unter Beibehaltung der
                    # relativen Unterordner-Struktur des Quellordners.
                    rel_path = os.path.relpath(path, source_folder)
                    dest_path = os.path.join(output_folder, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Metadaten immer aus dem Original lesen (vor dem Ersetzen)
                meta = read_metadata(path) if keep_metadata else None
                meta_warnings = save_image_safely(dest_path, img, meta, strip_gps, orientation)
                photos_saved += 1
                for w in meta_warnings:
                    errors.append(f"{os.path.basename(path)}: {w} (Bild wurde gespeichert)")

            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
            finally:
                # Fortschritt auch bei "continue" (unverändertes Bild) und Fehlern
                self._set_progress(idx)
                done_time = time.monotonic()
                self.after(0, lambda d=idx, t=done_time: self._eta_image_done(d, t))

        self._set_status("Fertig.")
        self.after(0, self._eta_finish)
        self.after(0, self._processing_done)

        summary_lines = [
            f"{len(paths)} Bild(er) verarbeitet.",
            f"In {photos_with_blur} Bild(ern) wurden insgesamt {total_regions} Bereich(e) verwischt.",
            f"Dauer: {self._format_duration(time.monotonic() - run_start)} "
            f"(Rechenwerk: {self.detector.backend_name}).",
        ]
        if do_correct:
            summary_lines.append(
                f"{photos_brightened} Bild(er) wurden aufgehellt, {photos_darkened} Bild(er) abgedunkelt."
            )
        if overwrite:
            summary_lines.append(f"{photos_saved} Originalbild(er) im Quellordner ersetzt.")
            if photos_unchanged:
                summary_lines.append(
                    f"{photos_unchanged} Bild(er) ohne erkannte Person blieben unverändert."
                )
        else:
            summary_lines.append(f"{photos_saved} Bild(er) gespeichert in: {output_folder}")
        summary = "\n".join(summary_lines)

        if errors:
            msg = summary + "\n\nBei folgenden Bildern gab es Probleme:\n" + "\n".join(errors)
            self.after(0, lambda: messagebox.showwarning("Fertig mit Fehlern", msg))
        else:
            self.after(0, lambda: messagebox.showinfo("Fertig", summary))

    def _processing_done(self):
        """Läuft im GUI-Thread nach Abschluss der Verarbeitung."""
        self._busy = False
        self.start_btn.config(state="normal")

    # ------------------------------------------------- Zeitabschätzung ----
    # Alle _eta_*-Methoden laufen im GUI-Thread (Aufruf über self.after).
    # Die Restzeit wird aus der durchschnittlichen Dauer der bereits
    # fertigen Bilder hochgerechnet und jede Sekunde heruntergezählt.

    @staticmethod
    def _format_duration(seconds):
        """Formatiert Sekunden mit Einheiten, z.B. '1 h 02 min 05 sec',
        '2 min 14 sec' oder '45 sec'."""
        seconds = max(0, int(round(seconds)))
        h, rest = divmod(seconds, 3600)
        m, s = divmod(rest, 60)
        if h:
            return f"{h} h {m:02d} min {s:02d} sec"
        if m:
            return f"{m} min {s:02d} sec"
        return f"{s} sec"

    def _eta_begin(self, total, start):
        self._eta = {"start": start, "total": total, "done": 0, "last": None}
        self._eta_tick()

    def _eta_image_done(self, done, finished_at):
        if self._eta is None:
            return
        self._eta["done"] = done
        self._eta["last"] = finished_at
        self._eta_update_text()

    def _eta_update_text(self):
        eta = self._eta
        now = time.monotonic()
        elapsed = now - eta["start"]
        text = f"Verstrichen: {self._format_duration(elapsed)}"

        if eta["done"] == 0:
            text += "  ·  Restzeit: wird nach dem ersten Bild berechnet…"
        elif eta["done"] < eta["total"]:
            per_image = (eta["last"] - eta["start"]) / eta["done"]
            remaining_images = eta["total"] - eta["done"]
            # Seit dem letzten fertigen Bild vergangene Zeit abziehen, damit
            # die Anzeige zwischen zwei Bildern weiter herunterzählt.
            remaining = per_image * remaining_images - (now - eta["last"])
            if remaining > 1:
                finish = datetime.now() + timedelta(seconds=remaining)
                text += (f"  ·  Restzeit: ca. {self._format_duration(remaining)}"
                         f"  ·  fertig um ca. {finish:%H:%M} Uhr")
            else:
                text += "  ·  Restzeit: gleich fertig…"
        self.eta_var.set(text)

    def _eta_tick(self):
        if self._eta is None:
            return
        self._eta_update_text()
        self._eta_job = self.after(1000, self._eta_tick)

    def _eta_stop(self):
        if self._eta_job is not None:
            self.after_cancel(self._eta_job)
            self._eta_job = None

    def _eta_finish(self):
        self._eta_stop()
        if self._eta is not None:
            total_time = time.monotonic() - self._eta["start"]
            count = self._eta["total"]
            per_image = total_time / count if count else 0
            self.eta_var.set(
                f"Gesamtdauer: {self._format_duration(total_time)}"
                f"  (ø {self._format_duration(per_image)} pro Bild)"
            )
        self._eta = None

    # ------------------------------------------------- Thread-sichere UI ----
    def _set_status(self, text):
        self.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, value):
        self.after(0, lambda: self.progress.config(value=value))


if __name__ == "__main__":
    app = App()
    app.mainloop()
