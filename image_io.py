"""
image_io.py
Unicode-sicheres Lesen und Schreiben von Bildern.

cv2.imread / cv2.imwrite scheitern unter Windows still bei Pfaden mit
Umlauten oder anderen Nicht-ASCII-Zeichen (z.B. "Zürich", "Gebäude"):
imread liefert dann None, imwrite gibt False zurück. Deshalb wird die
Datei hier als Byte-Puffer gelesen bzw. geschrieben (np.fromfile / open),
und OpenCV übernimmt nur das De- bzw. Encodieren im Speicher.

Zusätzlich: Übernahme der JPEG-Metadaten (EXIF, XMP inkl. Theta-GPano,
ICC) ins bearbeitete Bild, siehe read_metadata() / write_image_with_meta().

Bildausrichtung (EXIF-Tag "Orientation"):
Kameras speichern Hochformat-Fotos oft quer und vermerken nur im Tag, wie
das Bild beim Anzeigen zu drehen ist. Programme wie RiSCAN arbeiten mit den
Pixeln so, wie die Kamera sie gespeichert hat (Kalibrierung bezieht sich
auf den Sensor). Deshalb gilt:
  - read_image_with_orientation() liest die Pixel UNGEDREHT und liefert
    das Orientation-Tag mit,
  - to_upright()/from_upright() drehen nur intern (z.B. für die
    Personenerkennung) und wieder zurück,
  - write_image_with_meta(..., orientation=o) behält das Tag bei, sodass
    das gespeicherte Bild geometrisch identisch mit dem Original ist.
"""

import os
import re
import struct

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Wie cv2.imread, aber Unicode-sicher. Liefert None bei Fehler."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# ---------------------------------------------------------------------
# Bildausrichtung
# ---------------------------------------------------------------------
def to_upright(img, orientation):
    """Dreht/spiegelt ungedreht gespeicherte Pixel so, wie das Bild gemäss
    EXIF-Orientation (1-8) angezeigt wird."""
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def from_upright(img, orientation):
    """Umkehrung von to_upright(): von der Anzeige-Ausrichtung zurück in die
    gespeicherte Ausrichtung (z.B. für eine Maske aus der Personenerkennung)."""
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # 2, 3, 4, 5, 7 sind jeweils ihre eigene Umkehrung, 1 = unverändert
    return to_upright(img, orientation)


def read_image_with_orientation(path):
    """
    Liest ein Bild Unicode-sicher. Rückgabe: (img, orientation).

    JPEG mit lesbarem EXIF: Pixel UNGEDREHT (so wie gespeichert) und das
    Orientation-Tag (1-8). Zum Anzeigen bzw. für die Personenerkennung mit
    to_upright() drehen, beim Speichern `orientation` an
    write_image_with_meta() übergeben.

    Alle anderen Fälle (PNG, JPEG ohne/mit unlesbarem EXIF): Pixel wie von
    OpenCV angezeigt, orientation = 1 (ein unlesbares EXIF wird beim
    Speichern ohnehin weggelassen, die Drehung steckt dann in den Pixeln).
    img ist None, wenn die Datei nicht gelesen werden kann.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None, 1
    if data.size == 0:
        return None, 1

    orientation = None
    if path.lower().endswith(JPEG_EXT):
        orientation = _read_orientation(data.tobytes())
    if orientation is None:
        return cv2.imdecode(data, cv2.IMREAD_COLOR), 1
    img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return img, orientation


def _read_orientation(jpeg_bytes):
    """Orientation-Tag aus dem EXIF eines JPEG (1-8). 1, wenn kein EXIF oder
    kein Tag vorhanden ist; None, wenn das EXIF nicht lesbar ist."""
    for marker, payload in _iter_jpeg_segments(jpeg_bytes):
        if marker == 0xE1 and payload.startswith(_EXIF_HEADER):
            try:
                return _Tiff(bytearray(payload[len(_EXIF_HEADER):])).get_orientation()
            except Exception:
                return None
    return 1


def write_image(path, img, params=None):
    """Wie cv2.imwrite, aber Unicode-sicher. Liefert True bei Erfolg."""
    ext = os.path.splitext(path)[1]
    if not ext:
        return False
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


# =====================================================================
# Metadaten (nur JPEG)
# =====================================================================
# cv2.imencode schreibt weder EXIF noch XMP. Ohne XMP fehlen die
# GPano-Tags der Theta Z1, und 360°-Viewer erkennen das Bild nicht mehr
# als Panorama. Deshalb werden aus dem Original übernommen:
#   - APP1 EXIF   (Kameradaten, Aufnahmezeit, ...)
#   - APP1 XMP    (u.a. GPano-Tags für die 360°-Darstellung)
#   - APP2 ICC    (Farbprofil)
# Aus Datenschutzgründen IMMER entfernt bzw. nicht übernommen:
#   - das in EXIF eingebettete Vorschaubild (würde die Personen
#     unverwischt zeigen!)
#   - Vorschaubilder in XMP (xmp:Thumbnails) und Extended XMP (kann
#     ganze Bilder enthalten, z.B. GImage:Data)
#   - APP13 (Photoshop/IPTC) und APP2 MPF (Multi-Picture-Format), da
#     beide Vorschaubilder enthalten können
# Optional: GPS-Position aus EXIF und XMP entfernen.
#
# Orientierung: write_image_with_meta(..., orientation=o) schreibt o als
# EXIF-/XMP-Orientation. Standard 1 = die Pixel sind bereits aufgerichtet
# (z.B. von read_image() gelesen, das die Drehung auf die Pixel anwendet).
# Werden die Pixel ungedreht gespeichert (read_image_with_orientation()),
# das gelesene Tag übergeben, damit das Bild unverändert bleibt.

JPEG_EXT = (".jpg", ".jpeg")
JPEG_QUALITY = 95

_EXIF_HEADER = b"Exif\x00\x00"
_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_ICC_HEADER = b"ICC_PROFILE\x00"

_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2,
                    9: 4, 10: 8, 11: 4, 12: 8, 13: 4}

TAG_ORIENTATION = 0x0112
TAG_EXIF_IFD = 0x8769
TAG_GPS_IFD = 0x8825
TAG_THUMB_OFFSET = 0x0201
TAG_THUMB_LENGTH = 0x0202
TAG_STRIP_OFFSETS = 0x0111
TAG_STRIP_BYTECOUNTS = 0x0117
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003


class ImageMeta:
    """Aus dem Original gelesene Metadaten-Segmente (Payload ohne Marker/Länge)."""

    def __init__(self):
        self.exif = None        # bytes, beginnt mit b"Exif\0\0"
        self.xmp = None         # bytes, beginnt mit _XMP_HEADER
        self.icc_chunks = []    # Liste von bytes, je beginnend mit _ICC_HEADER

    def is_empty(self):
        return self.exif is None and self.xmp is None and not self.icc_chunks


def read_metadata(path):
    """Liest EXIF/XMP/ICC eines JPEG (Unicode-sicher). None bei Nicht-JPEG/Fehler."""
    if not path.lower().endswith(JPEG_EXT):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8).tobytes()
    except OSError:
        return None
    meta = ImageMeta()
    for marker, payload in _iter_jpeg_segments(data):
        if marker == 0xE1 and payload.startswith(_EXIF_HEADER) and meta.exif is None:
            meta.exif = payload
        elif marker == 0xE1 and payload.startswith(_XMP_HEADER) and meta.xmp is None:
            meta.xmp = payload
        elif marker == 0xE2 and payload.startswith(_ICC_HEADER):
            meta.icc_chunks.append(payload)
        # Alles andere (APP13, MPF, Extended XMP, ...) wird bewusst verworfen.
    return None if meta.is_empty() else meta


def write_image_with_meta(path, img, meta=None, strip_gps=False, orientation=1):
    """
    Wie write_image(), fügt bei JPEG aber die bereinigten Metadaten des
    Originals ein. Rückgabe: (ok, warnungen).

    orientation: EXIF-Orientation (1-8) der übergebenen Pixel. 1 = Pixel
    sind aufgerichtet (bisheriges Verhalten). Ein anderer Wert wird ins EXIF
    und XMP geschrieben; ohne übernommene Metadaten wird dafür ein
    minimales EXIF erzeugt, das nur die Orientation enthält.
    """
    ext = os.path.splitext(path)[1].lower()
    is_jpeg = ext in JPEG_EXT
    if is_jpeg and orientation != 1 and (meta is None or meta.exif is None):
        meta = _with_minimal_exif(meta, orientation)
    if not is_jpeg or meta is None:
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if is_jpeg else None
        return write_image(path, img, params), []

    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return False, []
    segments, warnings = _build_meta_segments(meta, img.shape, strip_gps, orientation)
    data = _insert_segments(buf.tobytes(), segments) if segments else buf.tobytes()
    try:
        with open(path, "wb") as fh:
            fh.write(data)
    except OSError:
        return False, warnings
    return True, warnings


def _iter_jpeg_segments(data):
    """Liefert (marker, payload) aller Segmente vor dem Bilddatenstrom (SOS)."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return
    pos, n = 2, len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            return
        marker = data[pos + 1]
        if marker == 0xFF:                       # Füllbyte
            pos += 1
            continue
        if 0xD0 <= marker <= 0xD8 or marker == 0x01:
            pos += 2
            continue
        if marker in (0xDA, 0xD9):               # SOS / EOI
            return
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if length < 2 or pos + 2 + length > n:
            return
        yield marker, data[pos + 4:pos + 2 + length]
        pos += 2 + length


def _with_minimal_exif(meta, orientation):
    """Kopie von meta (oder neues ImageMeta) mit einem EXIF, das nur das
    Orientation-Tag enthält (Little Endian, IFD0 mit einem Eintrag)."""
    tiff = (b"II*\x00" + struct.pack("<I", 8)          # Header, IFD0 bei Offset 8
            + struct.pack("<H", 1)                       # 1 Eintrag
            + struct.pack("<HHIHH", TAG_ORIENTATION, 3, 1, int(orientation), 0)
            + struct.pack("<I", 0))                      # kein IFD1
    new = ImageMeta()
    if meta is not None:
        new.xmp, new.icc_chunks = meta.xmp, list(meta.icc_chunks)
    new.exif = _EXIF_HEADER + tiff
    return new


def _build_meta_segments(meta, shape, strip_gps, orientation=1):
    segments, warnings = [], []

    if meta.exif is not None:
        try:
            tiff = _Tiff(bytearray(meta.exif[len(_EXIF_HEADER):]))
            tiff.remove_thumbnail()
            tiff.set_orientation(orientation)
            tiff.set_pixel_dimensions(shape[1], shape[0])
            if strip_gps:
                tiff.remove_gps()
            segments.append(_segment(0xE1, _EXIF_HEADER + bytes(tiff.data)))
        except Exception:
            # Lieber gar kein EXIF als eines mit evtl. unverwischtem Vorschaubild.
            warnings.append("EXIF-Daten nicht lesbar, wurden weggelassen")

    if meta.xmp is not None:
        payload = _XMP_HEADER + _clean_xmp(meta.xmp[len(_XMP_HEADER):], strip_gps, orientation)
        if len(payload) + 2 <= 0xFFFF:
            segments.append(_segment(0xE1, payload))
        else:
            warnings.append("XMP-Daten zu gross, wurden weggelassen")

    for chunk in meta.icc_chunks:
        segments.append(_segment(0xE2, chunk))
    return segments, warnings


def _segment(marker, payload):
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("Segment zu gross")
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def _insert_segments(jpeg, segments):
    """
    Fügt die Segmente direkt nach SOI ein. Das von OpenCV erzeugte
    JFIF-APP0-Segment wird entfernt, damit EXIF gemäss Spezifikation das
    erste Segment ist.
    """
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":
        pos = 4 + struct.unpack(">H", jpeg[4:6])[0]
    return b"\xff\xd8" + b"".join(segments) + jpeg[pos:]


# --- XMP ---------------------------------------------------------------
_XMP_REMOVE_ALWAYS = [
    re.compile(rb"<xmp:Thumbnails\b.*?</xmp:Thumbnails>", re.S),
    re.compile(rb"\s+xmpNote:HasExtendedXMP\s*=\s*\"[^\"]*\""),
    re.compile(rb"<xmpNote:HasExtendedXMP>.*?</xmpNote:HasExtendedXMP>", re.S),
]
_XMP_REMOVE_GPS = [
    re.compile(rb"\s+exif(?:EX)?:GPS\w+\s*=\s*\"[^\"]*\""),
    re.compile(rb"<(exif(?:EX)?:GPS\w+)\b[^>]*/>"),
    re.compile(rb"<(exif(?:EX)?:GPS\w+)\b[^>]*>.*?</\1>", re.S),
]


# Eine evtl. XMP-Orientierung an die geschriebene EXIF-Orientierung angleichen
_XMP_ORIENTATION = [
    re.compile(rb"(tiff:Orientation\s*=\s*\")\d(\")"),
    re.compile(rb"(<tiff:Orientation>)\s*\d\s*(</tiff:Orientation>)"),
]


def _clean_xmp(xmp, strip_gps, orientation=1):
    value = str(int(orientation)).encode()
    for rx in _XMP_ORIENTATION:
        xmp = rx.sub(lambda m: m.group(1) + value + m.group(2), xmp)
    for rx in _XMP_REMOVE_ALWAYS:
        xmp = rx.sub(b"", xmp)
    if strip_gps:
        for rx in _XMP_REMOVE_GPS:
            xmp = rx.sub(b"", xmp)
    return xmp


# --- EXIF ----------------------------------------------------------------
class _Tiff:
    """
    Minimaler TIFF/EXIF-Bearbeiter. Alle Änderungen erfolgen direkt im
    Byte-Puffer ("in place"), damit herstellerspezifische Daten (MakerNote
    mit absoluten Offsets) unverändert gültig bleiben.
    """

    def __init__(self, data):
        self.data = data
        if data[0:2] == b"II":
            self.bo = "<"
        elif data[0:2] == b"MM":
            self.bo = ">"
        else:
            raise ValueError("Ungültiger TIFF-Header")
        if self._u16(2) != 42:
            raise ValueError("Ungültiger TIFF-Header")
        self.ifd0 = self._u32(4)

    def _u16(self, pos):
        return struct.unpack_from(self.bo + "H", self.data, pos)[0]

    def _u32(self, pos):
        return struct.unpack_from(self.bo + "I", self.data, pos)[0]

    def _zero(self, start, length):
        end = min(len(self.data), start + length)
        if 0 <= start < end:
            self.data[start:end] = bytes(end - start)

    def _entries(self, ifd_offset):
        """dict tag -> (typ, count, value_pos, byte_size) sowie Anzahl Einträge."""
        n = self._u16(ifd_offset)
        if ifd_offset + 2 + 12 * n + 4 > len(self.data):
            raise ValueError("IFD ausserhalb der Daten")
        entries = {}
        for i in range(n):
            ep = ifd_offset + 2 + 12 * i
            typ = self._u16(ep + 2)
            count = self._u32(ep + 4)
            size = _TIFF_TYPE_SIZES.get(typ, 1) * count
            value_pos = ep + 8 if size <= 4 else self._u32(ep + 8)
            entries[self._u16(ep)] = (typ, count, value_pos, size)
        return entries, n

    def _read_int(self, entry, index=0):
        typ, _, vp, _ = entry
        return self._u16(vp + 2 * index) if typ == 3 else self._u32(vp + 4 * index)

    def _write_int(self, entry, value):
        typ, _, vp, _ = entry
        if typ == 3:
            struct.pack_into(self.bo + "H", self.data, vp, value)
        elif typ == 4:
            struct.pack_into(self.bo + "I", self.data, vp, value)

    def get_orientation(self):
        """Orientation-Tag (1-8); 1, wenn nicht vorhanden oder ungültig."""
        entries, _ = self._entries(self.ifd0)
        if TAG_ORIENTATION in entries:
            value = self._read_int(entries[TAG_ORIENTATION])
            return value if 1 <= value <= 8 else 1
        return 1

    def set_orientation(self, value):
        entries, _ = self._entries(self.ifd0)
        if TAG_ORIENTATION in entries:
            self._write_int(entries[TAG_ORIENTATION], int(value))

    def set_orientation_normal(self):
        self.set_orientation(1)

    def set_pixel_dimensions(self, width, height):
        entries, _ = self._entries(self.ifd0)
        if TAG_EXIF_IFD not in entries:
            return
        exif_entries, _ = self._entries(self._read_int(entries[TAG_EXIF_IFD]))
        if TAG_PIXEL_X in exif_entries:
            self._write_int(exif_entries[TAG_PIXEL_X], width)
        if TAG_PIXEL_Y in exif_entries:
            self._write_int(exif_entries[TAG_PIXEL_Y], height)

    def remove_thumbnail(self):
        """Entfernt IFD1 (Vorschaubild) und überschreibt dessen Bilddaten mit Nullen."""
        _, n0 = self._entries(self.ifd0)
        next_ptr_pos = self.ifd0 + 2 + 12 * n0
        ifd1 = self._u32(next_ptr_pos)
        if ifd1 == 0:
            return
        entries, n1 = self._entries(ifd1)
        if TAG_THUMB_OFFSET in entries and TAG_THUMB_LENGTH in entries:
            self._zero(self._read_int(entries[TAG_THUMB_OFFSET]),
                       self._read_int(entries[TAG_THUMB_LENGTH]))
        if TAG_STRIP_OFFSETS in entries and TAG_STRIP_BYTECOUNTS in entries:
            so, sc = entries[TAG_STRIP_OFFSETS], entries[TAG_STRIP_BYTECOUNTS]
            for i in range(min(so[1], sc[1])):
                self._zero(self._read_int(so, i), self._read_int(sc, i))
        self._zero(ifd1, 2 + 12 * n1 + 4)
        struct.pack_into(self.bo + "I", self.data, next_ptr_pos, 0)

    def remove_gps(self):
        """Leert die GPS-IFD (0 Einträge) und überschreibt alle GPS-Werte mit Nullen."""
        entries, _ = self._entries(self.ifd0)
        if TAG_GPS_IFD not in entries:
            return
        gps_offset = self._read_int(entries[TAG_GPS_IFD])
        gps_entries, n = self._entries(gps_offset)
        for (_, _, vp, size) in gps_entries.values():
            if size > 4:
                self._zero(vp, size)
        self._zero(gps_offset, 2 + 12 * n + 4)
