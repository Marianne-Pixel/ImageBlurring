"""
main.py
GUI-Tool zur Anonymisierung von Personen in Bildern:
  - normale Fotos
  - Equirectangular-Panoramen der Ricoh Theta Z1

Ablauf:
  1. Ordner mit den Quellbildern auswählen -> wird nach JPEG/JPG/PNG durchsucht
  2. Speicherort für die bearbeiteten Bilder auswählen
  3. Gefundene Bilder erscheinen als Liste (Mehrfachauswahl möglich),
     daneben eine kleine Vorschau des in der Liste ausgewählten Bildes
  4. Verwisch-Methode (Gaussian Blur / Pixelieren) und Stärke wählen,
     mit Live-Vorschau an einem kleinen fiktiven Beispielbild
  5. Belichtungskorrektur optional aktivieren, Schatten- und Lichter-
     Korrektur über zwei unabhängige Regler einstellen
  6. Bildtyp wählen (Theta Z1 / normale Kamera) – standardmässig ist
     "Theta Z1 (Equirectangular)" bereits aktiviert
  7. "Start" klicken -> Bestätigungsdialog (inkl. Anzahl gewählter Bilder)
     -> Verarbeitung
  8. Am Ende: Zusammenfassung mit Anzahl verwischter Bereiche/Bilder sowie
     Anzahl auf-/abgehellter Bilder

Die bearbeiteten Bilder werden NICHT überschrieben, sondern unter dem
gewählten Speicherort abgelegt (mit derselben relativen Unterordner-
Struktur wie im Quellordner). Existieren dort bereits Dateien mit
gleichem Namen, werden diese überschrieben.

Start: python main.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

SUPPORTED_EXT = (".jpg", ".jpeg", ".png")

# Ultralytics-Onlineprüfungen (Updates, Downloads) abschalten
os.environ["YOLO_OFFLINE"] = "True"

from detector import PersonDetector
from processing import build_mask_from_polygons, blur_mask, count_regions, auto_correct_exposure


def resource_path(name):
    """Pfad zu mitgelieferten Dateien, funktioniert im Skript und in der Exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)

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
        self.title("Personen-Anonymisierung YOLO11 – Theta Z1 / Normale Bilder")
        self.geometry("840x860")
        self.minsize(780, 780)

        self.folder_path = tk.StringVar(value="")
        self.output_folder_path = tk.StringVar(value="")
        self.is_theta = tk.BooleanVar(value=True)
        self.auto_correct = tk.BooleanVar(value=False)
        self.shadow_intensity = tk.IntVar(value=22)
        self.highlight_intensity = tk.IntVar(value=22)
        self.blur_method = tk.StringVar(value="gaussian")
        self.blur_strength = tk.IntVar(value=22)
        self.image_paths = []
        self.detector = None  # wird erst beim Start geladen (spart Startzeit)

        self._thumb_photo = None      # Referenzen halten (sonst Garbage Collection)
        self._preview_photo = None
        self._sample_img, self._sample_polygon = self._create_fictive_sample()

        self._build_ui()
        self.update_preview()

    # ------------------------------------------------------------ UI ----
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # --- Ordner auswählen ---
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Ordner:").pack(side="left")
        ttk.Entry(top, textvariable=self.folder_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(top, text="Durchsuchen…", command=self.select_folder).pack(side="left")

        # --- Speicherort für die bearbeiteten Bilder auswählen ---
        output_row = ttk.Frame(self)
        output_row.pack(fill="x", **pad)

        ttk.Label(output_row, text="Speicherort:").pack(side="left")
        ttk.Entry(output_row, textvariable=self.output_folder_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(output_row, text="Durchsuchen…", command=self.select_output_folder).pack(side="left")

        # --- Bilderliste + kleine Vorschau daneben ---
        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, **pad)

        list_frame = ttk.LabelFrame(content, text="Gefundene Bilder (Mehrfachauswahl möglich)")
        list_frame.pack(side="left", fill="both", expand=True)

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
        blur_frame = ttk.LabelFrame(self, text="Verwisch-Einstellungen")
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

        ttk.Label(
            blur_left,
            text="Die Vorschau rechts zeigt den Effekt an einem kleinen,\n"
                 "fiktiven Beispielbild. Verwischt wird nur die tatsächliche\n"
                 "Personen-Silhouette (unregelmässige Form), nicht ein\n"
                 "Rechteck oder Oval.",
            foreground="#555555", justify="left"
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))

        blur_right = ttk.Frame(blur_frame)
        blur_right.pack(side="right", padx=8, pady=8)
        self.blur_preview_label = ttk.Label(blur_right)
        self.blur_preview_label.pack()

        # --- Belichtungskorrektur: Aktivieren, Intensität ---
        exposure_frame = ttk.LabelFrame(self, text="Belichtungskorrektur")
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
        options = ttk.LabelFrame(self, text="Bildtyp")
        options.pack(fill="x", **pad)

        ttk.Checkbutton(
            options,
            text="Bilder stammen von der Ricoh Theta Z1 (Equirectangular / 360°)",
            variable=self.is_theta,
        ).pack(anchor="w", padx=8, pady=4)

        # --- Status, Fortschritt, Start ---
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)

        self.status_var = tk.StringVar(value="Bereit.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        self.start_btn = ttk.Button(bottom, text="Start", command=self.on_start_clicked)
        self.start_btn.pack(side="right")

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
        self.image_paths = []
        self.listbox.delete(0, tk.END)
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(SUPPORTED_EXT):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, folder)
                    self.image_paths.append(full)
                    self.listbox.insert(tk.END, rel)
        self.status_var.set(f"{len(self.image_paths)} Bild(er) gefunden.")
        self._thumb_photo = None
        self.thumb_label.config(image="", text="Keine Auswahl")

    def select_all(self):
        self.listbox.select_set(0, tk.END)
        self.on_listbox_select()

    def deselect_all(self):
        self.listbox.select_clear(0, tk.END)
        self.on_listbox_select()

    # -------------------------------------------------- Bildvorschau ----
    def on_listbox_select(self, event=None):
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
            img = Image.open(path)
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
        w, h = 240, 160
        img = np.full((h, w, 3), (235, 225, 210), dtype=np.uint8)  # Hintergrund (BGR)

        for x in range(0, w, 12):
            cv2.line(img, (x, 0), (x, h), (210, 200, 185), 1)
        for y in range(0, h, 12):
            cv2.line(img, (0, y), (w, y), (210, 200, 185), 1)

        cv2.rectangle(img, (88, 96), (152, h), (110, 85, 60), -1)
        cv2.ellipse(img, (120, 66), (30, 36), 0, 0, 360, (95, 145, 205), -1)
        cv2.circle(img, (109, 60), 4, (40, 40, 40), -1)
        cv2.circle(img, (131, 60), 4, (40, 40, 40), -1)
        cv2.ellipse(img, (120, 78), (10, 5), 0, 0, 180, (60, 60, 120), 2)

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

        if not self.output_folder_path.get():
            messagebox.showinfo(
                "Kein Speicherort",
                "Bitte zuerst einen Speicherort für die bearbeiteten Bilder auswählen."
            )
            return

        count = len(selected_indices)
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
            "is_theta": self.is_theta.get(),
            "do_correct": self.auto_correct.get(),
            "shadow_intensity": self.shadow_intensity.get(),
            "highlight_intensity": self.highlight_intensity.get(),
            "method": self.blur_method.get(),
            "strength": self.blur_strength.get(),
            "source_folder": self.folder_path.get(),
            "output_folder": self.output_folder_path.get(),
        }

        self.start_btn.config(state="disabled")
        self.progress.config(maximum=len(selected_paths), value=0)

        thread = threading.Thread(
            target=self.process_images, args=(selected_paths, settings), daemon=True
        )
        thread.start()

    # ----------------------------------------------------- Verarbeitung ----
    def process_images(self, paths, settings):
        is_theta = settings["is_theta"]
        do_correct = settings["do_correct"]
        shadow_intensity = settings["shadow_intensity"]
        highlight_intensity = settings["highlight_intensity"]
        method = settings["method"]
        strength = settings["strength"]
        source_folder = settings["source_folder"]
        output_folder = settings["output_folder"]

        if self.detector is None:
            self._set_status("Lade YOLO11-Segmentierungsmodell…")
            #self.detector = PersonDetector(model_path="yolo11s-seg.pt", conf_threshold=0.35)
            self.detector = PersonDetector(model_path=resource_path("yolo11s-seg.pt"), conf_threshold=0.35) #greift auf die mitgelieferte Datei im Exe-Ordner zu

        errors = []
        total_regions = 0
        photos_with_blur = 0
        photos_brightened = 0
        photos_darkened = 0

        for idx, path in enumerate(paths, start=1):
            self._set_status(f"Bearbeite {idx}/{len(paths)}: {os.path.basename(path)}")
            try:
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError("Bild konnte nicht gelesen werden.")

                if is_theta:
                    mask = self.detector.detect_mask_equirect(img)
                else:
                    polygons = self.detector.detect_masks_normal(img)
                    mask = build_mask_from_polygons(img.shape, polygons)

                regions = count_regions(mask)
                if regions > 0:
                    photos_with_blur += 1
                    total_regions += regions
                    img = blur_mask(img, mask, method=method, strength=strength)

                if do_correct:
                    img, mean_diff = auto_correct_exposure(
                        img, shadow_intensity=shadow_intensity, highlight_intensity=highlight_intensity
                    )
                    if mean_diff > 1.0:
                        photos_brightened += 1
                    elif mean_diff < -1.0:
                        photos_darkened += 1

                # Ziel-Pfad im gewählten Speicherort, unter Beibehaltung der
                # relativen Unterordner-Struktur des Quellordners.
                rel_path = os.path.relpath(path, source_folder)
                dest_path = os.path.join(output_folder, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                ok = cv2.imwrite(dest_path, img)
                if not ok:
                    raise RuntimeError("Bild konnte nicht gespeichert werden.")

            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

            self._set_progress(idx)

        self._set_status("Fertig.")
        self.after(0, lambda: self.start_btn.config(state="normal"))

        summary_lines = [
            f"{len(paths)} Bild(er) verarbeitet.",
            f"In {photos_with_blur} Bild(ern) wurden insgesamt {total_regions} Bereich(e) verwischt.",
        ]
        if do_correct:
            summary_lines.append(
                f"{photos_brightened} Bild(er) wurden aufgehellt, {photos_darkened} Bild(er) abgedunkelt."
            )
        summary = "\n".join(summary_lines)

        if errors:
            msg = summary + "\n\nBei folgenden Bildern gab es Probleme:\n" + "\n".join(errors)
            self.after(0, lambda: messagebox.showwarning("Fertig mit Fehlern", msg))
        else:
            self.after(0, lambda: messagebox.showinfo("Fertig", summary))

    # ------------------------------------------------- Thread-sichere UI ----
    def _set_status(self, text):
        self.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, value):
        self.after(0, lambda: self.progress.config(value=value))


if __name__ == "__main__":
    app = App()
    app.mainloop()
