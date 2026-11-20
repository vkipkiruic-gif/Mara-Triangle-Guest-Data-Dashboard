import io
import math
import os
import struct
import tempfile
import urllib.request
import zipfile

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


# ==============================================================================
# 1. PURE-PYTHON ESRI SHAPEFILE (.SHP + .DBF + .PRJ) READER (ZERO GDAL NEEDED)
# ==============================================================================
def _utm_to_wgs84(easting: float, northing: float, zone: int = 36, southern: bool = True):
    """Converts UTM coordinates (e.g., Arc 1960 / WGS84 UTM Zone 36S/37S) to WGS84 Lon/Lat."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)

    x = easting - 500000.0
    y = northing - (10000000.0 if southern else 0.0)

    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    J1 = 3 * e1 / 2 - 27 * e1**3 / 32
    J2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    J3 = 151 * e1**3 / 96
    J4 = 1097 * e1**4 / 512

    fp = mu + J1 * math.sin(2 * mu) + J2 * math.sin(4 * mu) + J3 * math.sin(6 * mu) + J4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)

    C1 = ep2 * cos_fp**2
    T1 = tan_fp**2
    R1 = a * (1 - e2) / ((1 - e2 * sin_fp**2) ** 1.5)
    N1 = a / math.sqrt(1 - e2 * sin_fp**2)
    D = x / (N1 * k0)

    lat_rad = fp - (N1 * tan_fp / R1) * (
        D**2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * ep2) * D**4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1**2 - 252 * ep2 - 3 * C1**2) * D**6 / 720
    )
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lon_rad = lon0 + (
        D
        - (1 + 2 * T1 + C1) * D**3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * ep2 + 24 * T1**2) * D**5 / 120
    ) / cos_fp

    return math.degrees(lon_rad), math.degrees(lat_rad)


def _detect_utm_params(prj_text: str):
    """Detects UTM zone & hemisphere from .prj WKT text if shapefile is projected in meters."""
    if not prj_text:
        return None
    up = prj_text.upper()
    if "UTM" not in up and "TRANSVERSE_MERCATOR" not in up:
        return None
    zone = 36
    for z in range(1, 61):
        if f"ZONE_{z}" in up or f"ZONE {z}" in up or f"{z}S" in up or f"{z}N" in up:
            zone = z
            break
    southern = ("SOUTH" in up) or (f"{zone}S" in up) or ("FALSE_NORTHING,10000000" in up.replace(" ", "").replace(chr(34), ""))
    return (zone, southern)


def _read_dbf_records(dbf_path: str) -> list[dict]:
    """Reads attribute records from an ESRI .dbf file in pure Python."""
    if not os.path.exists(dbf_path):
        return []
    with open(dbf_path, "rb") as f:
        header = f.read(32)
        if len(header) < 32:
            return []
        num_records = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]

        fields = []
        while f.tell() < header_len - 1:
            desc = f.read(32)
            if not desc or desc[0] == 0x0D:
                break
            name = desc[:11].split(bytes([0]))[0].decode("utf-8", errors="ignore").strip()
            ftype = chr(desc[11])
            flen = desc[16]
            fields.append((name, ftype, flen))

        f.seek(header_len)
        records = []
        for _ in range(num_records):
            rec_bytes = f.read(record_len)
            if len(rec_bytes) < record_len:
                break
            if rec_bytes[0:1] == b"*":
                continue
            pos = 1
            row = {}
            for name, ftype, flen in fields:
                raw_val = rec_bytes[pos:pos + flen].decode("utf-8", errors="ignore").strip()
                pos += flen
                row[name] = raw_val
            records.append(row)
        return records


def load_shapefile_zip(uploaded_zip, fallback_name: str) -> list[dict]:
    """
    Extracts a zipped shapefile (.shp, .dbf, .prj) and returns a list of feature dicts:
      [{'geom_type': 'Polygon'|'Polyline'|'Point', 'parts': [[(lon, lat), ...]], 'name': str}, ...]
    Works in 100% pure Python with zero external C/GDAL dependencies, and automatically reprojects
    UTM coordinates (e.g., UTM 36S / 37S) to WGS84 Lon/Lat if needed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "archive.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        shp_path = None
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp") and not fname.startswith("._"):
                    shp_path = os.path.join(root, fname)
                    break
            if shp_path:
                break

        if not shp_path:
            raise ValueError(f"No .shp file found inside {uploaded_zip.name}.")

        base_path = os.path.splitext(shp_path)[0]
        dbf_path = base_path + ".dbf"
        prj_path = base_path + ".prj"

        prj_text = ""
        if os.path.exists(prj_path):
            with open(prj_path, "r", encoding="utf-8", errors="ignore") as pf:
                prj_text = pf.read()
        utm_params = _detect_utm_params(prj_text)

        records = _read_dbf_records(dbf_path)
        name_field = None
        if records:
            sample_keys = list(records[0].keys())
            for k in sample_keys:
                if any(tok in k.lower() for tok in [
                    "name", "conserv", "area", "zone", "sector", "patrol",
                    "river", "lugga", "road", "track", "post", "station", "gate", "camp"
                ]):
                    name_field = k
                    break

        def to_lonlat(x, y):
            if utm_params is not None or abs(x) > 180 or abs(y) > 90:
                zone, southern = utm_params if utm_params else (36, True)
                return _utm_to_wgs84(x, y, zone=zone, southern=southern)
            return (x, y)

        features = []
        with open(shp_path, "rb") as sf:
            file_hdr = sf.read(100)
            if len(file_hdr) < 100:
                return []
            file_len_bytes = struct.unpack(">I", file_hdr[24:28])[0] * 2

            rec_idx = 0
            while sf.tell() < file_len_bytes:
                rec_hdr = sf.read(8)
                if len(rec_hdr) < 8:
                    break
                content_len = struct.unpack(">I", rec_hdr[4:8])[0] * 2
                content = sf.read(content_len)
                if len(content) < 4:
                    break

                shape_type = struct.unpack("<I", content[0:4])[0]
                attrs = records[rec_idx] if rec_idx < len(records) else {}
                rec_idx += 1

                feat_name = fallback_name
                if name_field and attrs.get(name_field):
                    cand = str(attrs[name_field]).strip()
                    if cand and cand.lower() not in ("none", "nan", "null"):
                        feat_name = cand

                # Point types (1, 11, 21)
                if shape_type in (1, 11, 21) and len(content) >= 20:
                    x, y = struct.unpack("<dd", content[4:20])
                    lon, lat = to_lonlat(x, y)
                    features.append({"geom_type": "Point", "parts": [[(lon, lat)]], "name": feat_name})

                # MultiPoint types (8, 18, 28)
                elif shape_type in (8, 18, 28) and len(content) >= 40:
                    num_pts = struct.unpack("<I", content[36:40])[0]
                    pts = []
                    for i in range(num_pts):
                        off = 40 + i * 16
                        if off + 16 <= len(content):
                            x, y = struct.unpack("<dd", content[off:off + 16])
                            pts.append(to_lonlat(x, y))
                    if pts:
                        features.append({"geom_type": "Point", "parts": [pts], "name": feat_name})

                # PolyLine (3, 13, 23) or Polygon (5, 15, 25)
                elif shape_type in (3, 13, 23, 5, 15, 25) and len(content) >= 44:
                    gtype = "Polyline" if shape_type in (3, 13, 23) else "Polygon"
                    num_parts, num_pts = struct.unpack("<II", content[36:44])
                    parts_idx = list(struct.unpack(f"<{num_parts}I", content[44:44 + 4 * num_parts]))
                    pts_start = 44 + 4 * num_parts
                    all_pts = []
                    for i in range(num_pts):
                        off = pts_start + i * 16
                        if off + 16 <= len(content):
                            x, y = struct.unpack("<dd", content[off:off + 16])
                            all_pts.append(to_lonlat(x, y))
                    rings = []
                    for p_i in range(num_parts):
                        s_idx = parts_idx[p_i]
                        e_idx = parts_idx[p_i + 1] if p_i + 1 < num_parts else len(all_pts)
                        seg = all_pts[s_idx:e_idx]
                        if seg:
                            rings.append(seg)
                    if rings:
                        features.append({"geom_type": gtype, "parts": rings, "name": feat_name})

        return features


def get_features_bounds(features: list[dict]):
    xs, ys = [], []
    for feat in features:
        for ring in feat["parts"]:
            for lon, lat in ring:
                xs.append(lon)
                ys.append(lat)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def get_feature_centroid(feat: dict):
    xs, ys = [], []
    for ring in feat["parts"]:
        for lon, lat in ring:
            xs.append(lon)
            ys.append(lat)
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ==============================================================================
# 2. GEODESIC DISTANCE & PATROL TRACK SEGMENTATION HELPERS
# ==============================================================================
def _haversine_km_vec(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in kilometers between consecutive GPS fixes."""
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def extract_track_segments(df_tracks: pd.DataFrame, max_gap_km: float = 3.0, max_gap_hours: float = 3.0):
    """
    Splits a patrol unit's ordered GPS track points into continuous polyline segments
    whenever there is a large spatial jump (> max_gap_km) or time break (> max_gap_hours),
    and returns (segments_list, total_distance_km).
    """
    if len(df_tracks) == 0:
        return [], 0.0

    work = df_tracks.copy()
    if "_dt" in work.columns and work["_dt"].notna().any():
        work = work.sort_values(by=["_track_group", "_dt"]).reset_index(drop=True)
    else:
        work = work.sort_values(by=["_track_group"]).reset_index(drop=True)

    segments = []
    total_km = 0.0

    for _, grp in work.groupby("_track_group", sort=False):
        lats = grp["Lat"].to_numpy(dtype=float)
        lons = grp["Lon"].to_numpy(dtype=float)
        if len(lats) == 0:
            continue
        if len(lats) == 1:
            segments.append([(lons[0], lats[0])])
            continue

        dists = _haversine_km_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])

        if "_dt" in grp.columns and grp["_dt"].notna().any():
            dts = grp["_dt"].to_numpy()
            dt_hours = np.abs((dts[1:] - dts[:-1]) / np.timedelta64(1, "h"))
            dt_hours = np.where(np.isnan(dt_hours), 0.0, dt_hours)
        else:
            dt_hours = np.zeros_like(dists)

        cur_seg = [(lons[0], lats[0])]
        for i in range(len(dists)):
            d_km = float(dists[i])
            gap_h = float(dt_hours[i])
            # Break polyline across day/shift gaps or GPS jumps so straight lines don't cut across map
            if d_km <= max_gap_km and gap_h <= max_gap_hours:
                total_km += d_km
                cur_seg.append((lons[i + 1], lats[i + 1]))
            else:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = [(lons[i + 1], lats[i + 1])]
        if cur_seg:
            segments.append(cur_seg)

    return segments, round(total_km, 2)


# ==============================================================================
# 3. OPENTOPOMAP TILE STITCHER (https://tile.opentopomap.org/{z}/{x}/{y}.png)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_opentopomap_basemap(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int = 11):
    def deg2num(lat_deg, lon_deg, z):
        lat_rad = math.radians(max(min(lat_deg, 85.0511), -85.0511))
        n = 2.0 ** z
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    def num2deg(xtile, ytile, z):
        n = 2.0 ** z
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    x_min, y_max = deg2num(min_lat, min_lon, zoom)
    x_max, y_min = deg2num(max_lat, max_lon, zoom)

    while (x_max - x_min + 1) * (y_max - y_min + 1) > 49 and zoom > 7:
        zoom -= 1
        x_min, y_max = deg2num(min_lat, min_lon, zoom)
        x_max, y_min = deg2num(max_lat, max_lon, zoom)

    cols = max(x_max - x_min + 1, 1)
    rows = max(y_max - y_min + 1, 1)
    canvas = Image.new("RGB", (cols * 256, rows * 256), (242, 239, 233))

    headers = {"User-Agent": "MaraTriangleMappingReport/1.0"}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://tile.opentopomap.org/{zoom}/{x}/{y}.png"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    tile_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                    canvas.paste(tile_img, ((x - x_min) * 256, (y - y_min) * 256))
            except Exception:
                pass

    north_lat, west_lon = num2deg(x_min, y_min, zoom)
    south_lat, east_lon = num2deg(x_max + 1, y_max + 1, zoom)
    return canvas, (west_lon, east_lon, south_lat, north_lat)


# ==============================================================================
# 4. CARTOGRAPHIC HELPERS: NORTH ARROW + BEARING ROSE, SCALE BAR, LOGO
# ==============================================================================
def draw_compass_and_bearing(ax, x_frac=0.075, y_frac=0.88):
    """Draws a cartographic North arrow with 360-degree bearing ring (N/000°, E/090°, S/180°, W/270°)."""
    ax_inset = ax.inset_axes([x_frac - 0.055, y_frac - 0.075, 0.11, 0.15], transform=ax.transAxes)
    ax_inset.set_xlim(-1.45, 1.45)
    ax_inset.set_ylim(-1.45, 1.45)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")

    bg = mpatches.Circle((0, 0), 1.32, facecolor="white", edgecolor="#1F4E78", linewidth=1.3, alpha=0.92, zorder=1)
    ax_inset.add_patch(bg)

    for deg in range(0, 360, 30):
        rad = math.radians(90 - deg)
        r_in = 0.88 if deg % 90 == 0 else 0.96
        r_out = 1.08
        ax_inset.plot(
            [r_in * math.cos(rad), r_out * math.cos(rad)],
            [r_in * math.sin(rad), r_out * math.sin(rad)],
            color="#1F4E78",
            linewidth=1.2 if deg % 90 == 0 else 0.7,
            zorder=2
        )

    north_poly = mpatches.Polygon([[0, 0.85], [-0.20, 0], [0, 0.14]], closed=True, facecolor="#1F4E78", edgecolor="#1F4E78", zorder=3)
    north_poly_r = mpatches.Polygon([[0, 0.85], [0.20, 0], [0, 0.14]], closed=True, facecolor="#4A90E2", edgecolor="#1F4E78", zorder=3)
    south_poly = mpatches.Polygon([[0, -0.85], [-0.20, 0], [0.20, 0]], closed=True, facecolor="white", edgecolor="#1F4E78", linewidth=0.9, zorder=3)
    ax_inset.add_patch(north_poly)
    ax_inset.add_patch(north_poly_r)
    ax_inset.add_patch(south_poly)

    ax_inset.text(0, 1.16, "N (000°)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1F4E78", zorder=4)
    ax_inset.text(1.14, 0, "090°", ha="left", va="center", fontsize=6.2, fontweight="bold", color="#333333", zorder=4)
    ax_inset.text(0, -1.16, "180°", ha="center", va="top", fontsize=6.2, fontweight="bold", color="#333333", zorder=4)
    ax_inset.text(-1.14, 0, "270°", ha="right", va="center", fontsize=6.2, fontweight="bold", color="#333333", zorder=4)


def draw_scale_bar(ax, min_lon, max_lon, min_lat, max_lat):
    """Draws an accurate geodesic km scale bar in the bottom-left corner."""
    mid_lat = (min_lat + max_lat) / 2.0
    km_per_deg_lon = 111.32 * math.cos(math.radians(mid_lat))
    map_width_km = (max_lon - min_lon) * km_per_deg_lon

    target_km = map_width_km * 0.22
    nice_steps = [1, 2, 5, 10, 15, 20, 25, 30, 50, 100]
    scale_km = min(nice_steps, key=lambda s: abs(s - target_km))
    bar_deg = scale_km / max(km_per_deg_lon, 1e-6)

    x0 = min_lon + (max_lon - min_lon) * 0.04
    y0 = min_lat + (max_lat - min_lat) * 0.045
    h = (max_lat - min_lat) * 0.014

    pad_x = (max_lon - min_lon) * 0.015
    pad_y = (max_lat - min_lat) * 0.018
    box = mpatches.FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y),
        bar_deg + 2 * pad_x,
        h + 3.2 * pad_y,
        boxstyle="round,pad=0.002",
        facecolor="white",
        edgecolor="#444444",
        linewidth=0.8,
        alpha=0.9,
        zorder=18
    )
    ax.add_patch(box)

    half = bar_deg / 2.0
    ax.add_patch(mpatches.Rectangle((x0, y0), half, h, facecolor="#111111", edgecolor="#111111", linewidth=0.8, zorder=19))
    ax.add_patch(mpatches.Rectangle((x0 + half, y0), half, h, facecolor="white", edgecolor="#111111", linewidth=0.8, zorder=19))

    ax.text(x0, y0 + h * 1.45, "0", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111111", zorder=20)
    ax.text(x0 + half, y0 + h * 1.45, f"{scale_km / 2:g}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111111", zorder=20)
    ax.text(x0 + bar_deg, y0 + h * 1.45, f"{scale_km:g} km", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111111", zorder=20)


def _find_repo_logo_bytes():
    """Searches common repo paths for a Mara Conservancy logo file if none was uploaded in the UI."""
    candidates = [
        "mara_conservancy_logo.png", "mara_conservancy_logo.jpg",
        "logo.png", "logo.jpg",
        "assets/mara_conservancy_logo.png", "assets/logo.png",
        "images/mara_conservancy_logo.png", "images/logo.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return f.read()
            except Exception:
                pass
    return None


def add_top_right_logo(ax, logo_bytes, logo_scale_pct: int = 55):
    """Pins a compact, tightly trimmed Mara Conservancy logo to the top-right corner of `ax`."""
    effective_bytes = logo_bytes if logo_bytes else _find_repo_logo_bytes()
    if not effective_bytes:
        return
    pil_rgba = Image.open(io.BytesIO(effective_bytes)).convert("RGBA")

    alpha_bbox = pil_rgba.getchannel("A").getbbox()
    if alpha_bbox:
        pil_rgba = pil_rgba.crop(alpha_bbox)

    bg = Image.new("RGBA", pil_rgba.size, (255, 255, 255, 255))
    pil_logo = Image.alpha_composite(bg, pil_rgba).convert("RGB")

    gray = pil_logo.convert("L")
    mask = gray.point(lambda p: 255 if p < 232 else 0)
    bbox = mask.getbbox()
    if bbox:
        pad_trim = 4
        x0 = max(bbox[0] - pad_trim, 0)
        y0 = max(bbox[1] - pad_trim, 0)
        x1 = min(bbox[2] + pad_trim, pil_logo.width)
        y1 = min(bbox[3] + pad_trim, pil_logo.height)
        pil_logo = pil_logo.crop((x0, y0, x1, y1))

    factor = max(logo_scale_pct, 20) / 100.0
    target_h = max(int(68 * factor), 22)
    scale = target_h / max(float(pil_logo.height), 1.0)
    target_w = min(max(int(pil_logo.width * scale), 26), 160)
    pil_resized = pil_logo.resize((target_w, target_h), Image.LANCZOS)

    oi = OffsetImage(np.asarray(pil_resized), zoom=0.42)
    ab = AnnotationBbox(
        oi,
        (0.990, 0.990),
        xycoords="axes fraction",
        box_alignment=(1.0, 1.0),
        frameon=True,
        bboxprops=dict(
            facecolor="white",
            edgecolor="#1F4E78",
            linewidth=0.75,
            boxstyle="round,pad=0.08",
            alpha=0.95
        ),
        zorder=99
    )
    ax.add_artist(ab)


# ==============================================================================
# 5. RENDER PATROL TRACKS MAP (PER PATROL UNIT OR COMBINED) WITH ALL GIS LAYERS
# ==============================================================================
def render_patrol_track_map(
    title_subject: str,
    track_series_list: list,
    conservancies: list,
    rivers_features: list,
    rivers_label: str,
    roads_features: list,
    roads_label: str,
    ranger_features: list,
    ranger_posts_label: str,
    label_ranger_posts: bool,
    logo_bytes,
    report_period: str,
    zoom_level: int = 11,
    label_conservancies_on_map: bool = True,
    logo_scale_pct: int = 55,
    area_fill_alpha: float = 0.22,
    track_linewidth: float = 1.9,
    show_track_points: bool = False
) -> bytes:
    """
    `track_series_list` is a list of dicts:
      [{'name': str, 'segments': [[(lon, lat), ...]], 'color': str, 'points_df': DataFrame, 'distance_km': float}, ...]
    """
    bounds_list = []
    for ts in track_series_list:
        pts_df = ts["points_df"]
        if len(pts_df) > 0:
            bounds_list.append((pts_df["Lon"].min(), pts_df["Lat"].min(), pts_df["Lon"].max(), pts_df["Lat"].max()))
    for _, c_feats, _ in conservancies:
        b = get_features_bounds(c_feats)
        if b is not None:
            bounds_list.append(b)

    if bounds_list:
        min_lon = min(b[0] for b in bounds_list)
        min_lat = min(b[1] for b in bounds_list)
        max_lon = max(b[2] for b in bounds_list)
        max_lat = max(b[3] for b in bounds_list)
    else:
        min_lon, min_lat, max_lon, max_lat = 34.75, -1.65, 35.30, -1.24

    # Add 12% padding around the data extent
    lon_span = max(max_lon - min_lon, 0.05)
    lat_span = max(max_lat - min_lat, 0.05)
    min_lon -= lon_span * 0.12
    max_lon += lon_span * 0.12
    min_lat -= lat_span * 0.12
    max_lat += lat_span * 0.12

    # Enforce true 1:1 geographic aspect ratio (cos(lat) corrected) matching figure proportions
    fig_w, fig_h = 11.5, 8.4
    target_aspect = fig_w / fig_h
    mid_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    cos_lat = max(math.cos(mid_lat_rad), 0.1)

    cur_lon_span = max_lon - min_lon
    cur_lat_span = max_lat - min_lat
    ground_width = cur_lon_span * cos_lat
    ground_height = cur_lat_span

    if (ground_width / ground_height) < target_aspect:
        desired_ground_w = ground_height * target_aspect
        desired_lon_span = desired_ground_w / cos_lat
        lon_center = (min_lon + max_lon) / 2.0
        min_lon = lon_center - desired_lon_span / 2.0
        max_lon = lon_center + desired_lon_span / 2.0
    else:
        desired_lat_span = ground_width / target_aspect
        lat_center = (min_lat + max_lat) / 2.0
        min_lat = lat_center - desired_lat_span / 2.0
        max_lat = lat_center + desired_lat_span / 2.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)

    # 1. OpenTopoMap Basemap
    basemap_img, extent = fetch_opentopomap_basemap(min_lon, min_lat, max_lon, max_lat, zoom=zoom_level)
    ax.imshow(basemap_img, extent=extent, aspect="auto", zorder=1)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect(1.0 / cos_lat, adjustable="box")

    legend_handles = []

    # 2. Conservancy / Patrol Area Boundaries (with distinct translucent fill)
    for idx, (layer_label, c_feats, edge_color) in enumerate(conservancies):
        if not c_feats:
            continue
        lstyle = "-" if idx == 0 else "--"
        unique_names = []
        for feat in c_feats:
            if feat["name"] not in unique_names:
                unique_names.append(feat["name"])
            for ring in feat["parts"]:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                if feat["geom_type"] == "Polygon" and area_fill_alpha > 0:
                    poly = mpatches.Polygon(
                        ring,
                        closed=True,
                        facecolor=edge_color,
                        edgecolor="none",
                        alpha=area_fill_alpha,
                        zorder=3
                    )
                    ax.add_patch(poly)
                ax.plot(xs, ys, color=edge_color, linewidth=2.3, linestyle=lstyle, zorder=5 + idx)

            if label_conservancies_on_map:
                cen = get_feature_centroid(feat)
                if cen is not None and min_lon <= cen[0] <= max_lon and min_lat <= cen[1] <= max_lat:
                    ax.text(
                        cen[0],
                        cen[1],
                        str(feat["name"]).upper(),
                        fontsize=8.0,
                        fontweight="bold",
                        color=edge_color,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=edge_color, alpha=0.78, lw=0.8),
                        zorder=9
                    )

        disp_legend_name = ", ".join(unique_names[:3]) if len(unique_names) > 1 else (unique_names[0] if unique_names else layer_label)
        legend_handles.append(
            mpatches.Patch(
                facecolor=edge_color if area_fill_alpha > 0 else "none",
                edgecolor=edge_color,
                alpha=min(area_fill_alpha + 0.12, 0.65) if area_fill_alpha > 0 else 1.0,
                linewidth=2.2,
                linestyle=lstyle,
                label=disp_legend_name
            )
        )

    # 3. Roads & Tracks Overlay
    if roads_features:
        for feat in roads_features:
            for ring in feat["parts"]:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.plot(xs, ys, color="#5C4033", linewidth=1.15, linestyle="-", alpha=0.84, zorder=6)
        legend_handles.append(
            mlines.Line2D([], [], color="#5C4033", linewidth=1.6, linestyle="-", label=roads_label)
        )

    # 4. Rivers & Luggas Overlay
    if rivers_features:
        for feat in rivers_features:
            for ring in feat["parts"]:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.plot(xs, ys, color="#0066CC", linewidth=1.35, linestyle="-", alpha=0.88, zorder=7)
        legend_handles.append(
            mlines.Line2D([], [], color="#0066CC", linewidth=1.8, label=rivers_label)
        )

    # 5. Ranger Posts / Stations Overlay
    if ranger_features:
        pts_x, pts_y = [], []
        for feat in ranger_features:
            cen = get_feature_centroid(feat)
            if cen is not None:
                pts_x.append(cen[0])
                pts_y.append(cen[1])
                if label_ranger_posts and min_lon <= cen[0] <= max_lon and min_lat <= cen[1] <= max_lat:
                    ax.annotate(
                        str(feat["name"]),
                        xy=(cen[0], cen[1]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7.2,
                        fontweight="bold",
                        color="#145A32",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#1E8449", alpha=0.82, lw=0.6),
                        zorder=13
                    )
        if pts_x:
            ax.scatter(
                pts_x,
                pts_y,
                s=85,
                marker="^",
                c="#1E8449",
                edgecolors="white",
                linewidths=1.0,
                zorder=11
            )
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color="white",
                marker="^",
                markerfacecolor="#1E8449",
                markeredgecolor="white",
                markersize=9.5,
                label=ranger_posts_label
            )
        )

    # 6. Patrol Tracks Overlay (Polylines + Optional GPS Waypoints)
    total_points = 0
    total_dist_km = 0.0
    track_legend_handles = []

    for ts in track_series_list:
        t_name = ts["name"]
        t_color = ts["color"]
        t_segs = ts["segments"]
        pts_df = ts["points_df"]
        dist_km = ts["distance_km"]
        total_points += len(pts_df)
        total_dist_km += dist_km

        for seg in t_segs:
            if len(seg) >= 2:
                xs = [p[0] for p in seg]
                ys = [p[1] for p in seg]
                # White halo underneath for high contrast over OpenTopoMap terrain
                ax.plot(xs, ys, color="white", linewidth=track_linewidth + 1.4, linestyle="-", alpha=0.75, zorder=13.5)
                ax.plot(xs, ys, color=t_color, linewidth=track_linewidth, linestyle="-", alpha=0.92, zorder=14)
            elif len(seg) == 1:
                ax.scatter([seg[0][0]], [seg[0][1]], s=24, c=t_color, edgecolors="white", linewidths=0.7, zorder=14)

        if show_track_points and len(pts_df) > 0:
            ax.scatter(
                pts_df["Lon"],
                pts_df["Lat"],
                s=18,
                c=t_color,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.85,
                zorder=14.5
            )

        track_legend_handles.append(
            mlines.Line2D(
                [], [],
                color=t_color,
                linewidth=2.4,
                linestyle="-",
                label=f"{t_name} ({dist_km:,.1f} km | {len(pts_df):,} pts)"
            )
        )

    # Insert track legend entries at the top of the legend (cap at 10 individual items to keep legend compact)
    if len(track_legend_handles) <= 10:
        legend_handles = track_legend_handles + legend_handles
    else:
        legend_handles = track_legend_handles[:8] + [
            mlines.Line2D([], [], color="#D92323", linewidth=2.4, label=f"+ {len(track_legend_handles) - 8} more patrol tracks")
        ] + legend_handles

    # 7. Cartographic North Arrow & Bearing Rose + Scale Bar
    draw_compass_and_bearing(ax, x_frac=0.075, y_frac=0.88)
    draw_scale_bar(ax, min_lon, max_lon, min_lat, max_lat)

    # 8. Legend (Bottom-Right)
    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        title="MAP LEGEND",
        frameon=True,
        facecolor="white",
        edgecolor="#1F4E78",
        framealpha=0.94,
        fontsize=8.0,
        title_fontsize=9.0
    )
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(22)

    # 9. Title & Coordinate Frame
    title_line_1 = f"{title_subject.upper()} PATROL TRACKS MAP — {report_period.upper()}"
    title_line_2 = f"Patrol Distance: {total_dist_km:,.1f} km   |   GPS Fixes: {total_points:,}   |   Basemap: OpenTopoMap"
    ax.set_title(
        title_line_1 + chr(10) + title_line_2,
        fontsize=12.5,
        fontweight="bold",
        color="#1F4E78",
        pad=12
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Latitude (°S)", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#555555", alpha=0.45, zorder=2)

    ax.text(
        0.5, 0.008,
        "Map Data: © OpenStreetMap contributors, SRTM | Map Style: © OpenTopoMap (CC-BY-SA)",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#333333",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
        zorder=15
    )

    add_top_right_logo(ax, logo_bytes, logo_scale_pct=logo_scale_pct)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# STREAMLIT PAGE UI
# ==============================================================================
st.title("🛰️ Patrol Tracks Distribution Maps (CSV / Excel)")
st.markdown(
    "Generate cartographic patrol track maps for each ranger team, vehicle, or patrol unit over **OpenTopoMap** (`tile.opentopomap.org`), "
    "complete with **shaded conservancy/patrol boundaries**, **roads**, **rivers & luggas**, **ranger posts**, **company logo**, **North/bearing rose**, and **scale bar**."
)

col1, col2 = st.columns(2)
with col1:
    tracks_file = st.file_uploader(
        "1️⃣ Upload EarthRanger Patrol Tracks CSV / Excel (with Latitude & Longitude)",
        type=["csv", "xlsx", "xls"],
        key="patrol_tracks_csv_uploader"
    )
    logo_file = st.file_uploader(
        "2️⃣ Upload Mara Conservancy / Company Logo (PNG / JPG — placed in Top-Right)",
        type=["png", "jpg", "jpeg"],
        key="tracks_company_logo_uploader"
    )
    if logo_file is not None:
        st.image(logo_file, caption="Logo loaded (will appear in top-right corner of every map)", width=110)
    conservancy_zips = st.file_uploader(
        "3️⃣ Upload Conservancy / Patrol Area Boundary Shapefiles (`.zip` — one or multiple)",
        type=["zip"],
        accept_multiple_files=True,
        key="tracks_conservancy_zip_uploader"
    )

with col2:
    roads_zip = st.file_uploader(
        "4️⃣ Upload Roads & Tracks Shapefile (`.zip`)",
        type=["zip"],
        key="tracks_roads_zip_uploader"
    )
    rivers_zips = st.file_uploader(
        "5️⃣ Upload Rivers & Luggas Shapefiles (`.zip` — upload Rivers & Luggas together)",
        type=["zip"],
        accept_multiple_files=True,
        key="tracks_rivers_zip_uploader"
    )
    ranger_posts_zip = st.file_uploader(
        "6️⃣ Upload Ranger Posts / Stations Shapefile (`.zip`)",
        type=["zip"],
        key="tracks_ranger_posts_zip_uploader"
    )

with st.sidebar:
    st.subheader("Map Styling & Legend Labels")
    report_period = st.text_input("Report Period Label", value="November 2026", key="tracks_map_period_label")
    zoom_level = st.slider("OpenTopoMap Tile Zoom Level", min_value=9, max_value=13, value=11, key="tracks_topo_zoom")
    logo_scale_pct = st.slider("Top-Right Logo Size (%)", min_value=25, max_value=100, value=55, step=5, key="tracks_logo_size_pct")
    track_linewidth = st.slider("Patrol Track Line Width", min_value=1.0, max_value=4.0, value=1.9, step=0.1, key="tracks_line_w")
    show_track_points = st.checkbox("Show Individual GPS Waypoint Dots on Tracks", value=False, key="tracks_show_dots")
    max_gap_km = st.number_input("Split Track Segment if Jump > (km)", min_value=0.5, max_value=25.0, value=3.0, step=0.5, key="tracks_max_gap_km")
    roads_legend_name = st.text_input("Roads Legend Label", value="Roads & Tracks", key="tracks_roads_legend_txt")
    rivers_legend_name = st.text_input("Rivers & Luggas Legend Label", value="Rivers & Luggas", key="tracks_rivers_legend_txt")
    ranger_legend_name = st.text_input("Ranger Posts Legend Label", value="Ranger Posts", key="tracks_ranger_legend_txt")
    area_fill_opacity = st.slider("Patrol Area Shading Opacity (%)", min_value=0, max_value=55, value=22, step=2, key="tracks_area_shade_alpha")
    show_poly_labels = st.checkbox("Label Conservancy / Patrol Areas on Map", value=True, key="tracks_show_poly_names")
    show_post_labels = st.checkbox("Label Ranger Posts on Map", value=True, key="tracks_show_post_names")

if tracks_file is not None:
    if tracks_file.name.lower().endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(tracks_file)
    else:
        raw_df = pd.read_csv(tracks_file)

    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    lat_col = next((c for c in raw_df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in raw_df.columns if "lon" in c.lower()), None)

    if not lat_col or not lon_col:
        st.error("Uploaded patrol tracks file must contain 'Latitude' and 'Longitude' columns.")
        st.stop()

    # Auto-detect grouping column (Patrol / Subject / Ranger / Team / Vehicle / Leader / Device / Track ID)
    candidate_group_cols = [
        c for c in raw_df.columns
        if c not in (lat_col, lon_col)
        and any(tok in c.lower() for tok in [
            "subject", "patrol", "ranger", "team", "unit", "vehicle",
            "callsign", "leader", "device", "source", "track", "name", "title", "user"
        ])
    ]
    if not candidate_group_cols:
        candidate_group_cols = [c for c in raw_df.columns if c not in (lat_col, lon_col)]

    group_col = st.selectbox(
        "Group Patrol Tracks By Column (each group gets its own track color & individual map):",
        options=candidate_group_cols,
        index=0,
        key="tracks_group_col_select"
    )

    # Detect timestamp column if present so GPS points are connected chronologically
    time_col = next(
        (c for c in raw_df.columns if any(tok in c.lower() for tok in ["recorded", "time", "date", "timestamp", "created", "start"])),
        None
    )

    raw_df["Lat"] = pd.to_numeric(raw_df[lat_col], errors="coerce")
    raw_df["Lon"] = pd.to_numeric(raw_df[lon_col], errors="coerce")
    raw_df["_track_group"] = raw_df[group_col].fillna("Unassigned Patrol").astype(str).str.strip()
    if time_col:
        raw_df["_dt"] = pd.to_datetime(raw_df[time_col], errors="coerce")
    else:
        raw_df["_dt"] = pd.NaT

    valid_geo_df = raw_df.dropna(subset=["Lat", "Lon"]).copy()

    # Load Conservancy ZIP shapefiles
    conservancies_loaded = []
    boundary_palette = ["#1B4F72", "#D35400", "#7D3C98", "#117864", "#B7950B", "#922B21"]
    if conservancy_zips:
        st.subheader("Uploaded Conservancy / Patrol Area Boundaries")
        name_cols = st.columns(min(len(conservancy_zips), 3))
        for idx, zfile in enumerate(conservancy_zips):
            default_label = os.path.splitext(zfile.name)[0].replace("_", " ").title()
            default_col = boundary_palette[idx % len(boundary_palette)]
            with name_cols[idx % len(name_cols)]:
                custom_name = st.text_input(
                    f"Area {idx + 1} Name ({zfile.name})",
                    value=default_label,
                    key=f"tracks_cons_name_{idx}_{zfile.name}"
                )
                chosen_col = st.color_picker(
                    f"Area {idx + 1} Shade & Border Color",
                    value=default_col,
                    key=f"tracks_cons_col_{idx}_{zfile.name}"
                )
            try:
                feats_c = load_shapefile_zip(zfile, fallback_name=custom_name)
                unique_fnames = set(f["name"] for f in feats_c)
                if len(unique_fnames) == 1 and default_label in unique_fnames:
                    for f in feats_c:
                        f["name"] = custom_name
                conservancies_loaded.append((custom_name, feats_c, chosen_col))
            except Exception as e:
                st.warning(f"Could not read {zfile.name}: {e}")

    # Load Roads, Rivers & Luggas, and Ranger Posts ZIP shapefiles
    roads_features = []
    if roads_zip is not None:
        try:
            roads_features = load_shapefile_zip(roads_zip, fallback_name=roads_legend_name)
        except Exception as e:
            st.warning(f"Could not read Roads shapefile: {e}")

    rivers_features = []
    if rivers_zips:
        for rzip in rivers_zips:
            try:
                rivers_features.extend(load_shapefile_zip(rzip, fallback_name=rivers_legend_name))
            except Exception as e:
                st.warning(f"Could not read Rivers/Luggas shapefile ({rzip.name}): {e}")

    ranger_features = []
    if ranger_posts_zip is not None:
        try:
            ranger_features = load_shapefile_zip(ranger_posts_zip, fallback_name=ranger_legend_name)
        except Exception as e:
            st.warning(f"Could not read Ranger Posts shapefile: {e}")

    logo_bytes = None
    if logo_file is not None:
        try:
            raw_logo = logo_file.getvalue()
            Image.open(io.BytesIO(raw_logo)).verify()
            logo_bytes = raw_logo
        except Exception as e:
            st.warning(f"Could not read uploaded logo image ({logo_file.name}): {e}")

    # --- COMPUTE TRACK SEGMENTS & DISTANCE SUMMARY PER PATROL GROUP ---
    summary_rows = []
    group_segments_cache = {}
    for grp_name, grp_df in valid_geo_df.groupby("_track_group"):
        segs, dist_km = extract_track_segments(grp_df, max_gap_km=max_gap_km, max_gap_hours=3.0)
        group_segments_cache[grp_name] = (segs, dist_km, grp_df)
        summary_rows.append({
            "Patrol Unit / Track Group": grp_name,
            "GPS Fixes Recorded": len(grp_df),
            "Continuous Track Segments": len(segs),
            "Estimated Patrol Distance (km)": dist_km,
        })

    tracks_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(by=["Estimated Patrol Distance (km)", "GPS Fixes Recorded"], ascending=False)
        .reset_index(drop=True)
    )
    total_all_km = tracks_summary["Estimated Patrol Distance (km)"].sum() if len(tracks_summary) > 0 else 0.0
    tracks_summary["Share of Distance (%)"] = (
        (tracks_summary["Estimated Patrol Distance (km)"] / total_all_km * 100).round(2)
        if total_all_km > 0 else 0.0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total GPS Track Fixes", f"{len(valid_geo_df):,}")
    m2.metric("Patrol Units / Tracks", f"{len(tracks_summary)}")
    m3.metric("Total Patrol Distance", f"{total_all_km:,.1f} km")
    active_layers = len(conservancies_loaded) + int(len(roads_features) > 0) + int(len(rivers_features) > 0) + int(len(ranger_features) > 0)
    m4.metric("Active GIS Map Layers", f"{active_layers} layers")

    st.subheader(f"1. Patrol Effort & Distance Summary ({report_period})")
    st.dataframe(tracks_summary, use_container_width=True, hide_index=True)

    st.divider()

    # --- GENERATE PATROL TRACK MAPS ---
    st.subheader(f"2. Patrol Tracks Distribution Maps ({report_period})")
    all_groups_list = tracks_summary["Patrol Unit / Track Group"].tolist()
    default_selected = all_groups_list[:6]

    include_combined_map = st.checkbox(
        "Also Generate a Combined All-Selected Patrol Tracks Overview Map",
        value=True,
        key="include_combined_tracks_map"
    )

    selected_groups = st.multiselect(
        "Select Patrol Units / Tracks to Generate Individual Maps For:",
        options=all_groups_list,
        default=default_selected,
        key="selected_track_groups"
    )

    track_palette = [
        "#D92323", "#E67E22", "#8E44AD", "#16A085", "#C0392B",
        "#2980B9", "#D35400", "#27AE60", "#884EA0", "#B03A2E"
    ]

    if selected_groups:
        if logo_bytes or _find_repo_logo_bytes():
            st.success("✅ Mara Conservancy Logo active — it will appear in the top-right corner of every generated map.")
        else:
            st.warning("⚠️ No logo uploaded yet in box **2️⃣** above (Streamlit clears uploaded files whenever the app restarts). Upload your Mara Conservancy logo PNG/JPG in box **2️⃣** if you want it in the top-right corner.")

        if st.button("🗺️ Generate All Selected Patrol Track Maps & ZIP Bundle", type="primary", key="btn_gen_track_maps"):
            zip_buffer = io.BytesIO()
            generated_maps = []

            total_steps = len(selected_groups) + (1 if include_combined_map else 0)
            progress = st.progress(0)
            step_done = 0

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # 1. Optional Combined Overview Map of all selected patrol tracks
                if include_combined_map:
                    combined_series = []
                    for idx, grp_name in enumerate(selected_groups):
                        segs, dist_km, grp_df = group_segments_cache[grp_name]
                        combined_series.append({
                            "name": grp_name,
                            "segments": segs,
                            "color": track_palette[idx % len(track_palette)],
                            "points_df": grp_df,
                            "distance_km": dist_km,
                        })
                    combined_png = render_patrol_track_map(
                        title_subject="ALL SELECTED UNITS",
                        track_series_list=combined_series,
                        conservancies=conservancies_loaded,
                        rivers_features=rivers_features,
                        rivers_label=rivers_legend_name,
                        roads_features=roads_features,
                        roads_label=roads_legend_name,
                        ranger_features=ranger_features,
                        ranger_posts_label=ranger_legend_name,
                        label_ranger_posts=show_post_labels,
                        logo_bytes=logo_bytes,
                        report_period=report_period,
                        zoom_level=zoom_level,
                        label_conservancies_on_map=show_poly_labels,
                        logo_scale_pct=logo_scale_pct,
                        area_fill_alpha=area_fill_opacity / 100.0,
                        track_linewidth=track_linewidth,
                        show_track_points=show_track_points,
                    )
                    generated_maps.append(("Combined_All_Patrol_Tracks", "All Selected Patrol Tracks (Combined Overview)", combined_png))
                    zf.writestr(f"All_Patrol_Tracks_Overview_{report_period.replace(' ', '_')}.png", combined_png)
                    step_done += 1
                    progress.progress(step_done / total_steps)

                # 2. Individual Map per Patrol Unit / Track Group
                for idx, grp_name in enumerate(selected_groups):
                    segs, dist_km, grp_df = group_segments_cache[grp_name]
                    color = track_palette[idx % len(track_palette)]
                    single_series = [{
                        "name": grp_name,
                        "segments": segs,
                        "color": color,
                        "points_df": grp_df,
                        "distance_km": dist_km,
                    }]
                    png_bytes = render_patrol_track_map(
                        title_subject=grp_name,
                        track_series_list=single_series,
                        conservancies=conservancies_loaded,
                        rivers_features=rivers_features,
                        rivers_label=rivers_legend_name,
                        roads_features=roads_features,
                        roads_label=roads_legend_name,
                        ranger_features=ranger_features,
                        ranger_posts_label=ranger_legend_name,
                        label_ranger_posts=show_post_labels,
                        logo_bytes=logo_bytes,
                        report_period=report_period,
                        zoom_level=zoom_level,
                        label_conservancies_on_map=show_poly_labels,
                        logo_scale_pct=logo_scale_pct,
                        area_fill_alpha=area_fill_opacity / 100.0,
                        track_linewidth=track_linewidth,
                        show_track_points=show_track_points,
                    )
                    safe_grp = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in grp_name).strip().replace(" ", "_")
                    generated_maps.append((safe_grp, f"{grp_name} — Distance: {dist_km:,.1f} km ({len(grp_df):,} GPS fixes)", png_bytes))
                    zf.writestr(f"{safe_grp}_Patrol_Track_Map_{report_period.replace(' ', '_')}.png", png_bytes)
                    step_done += 1
                    progress.progress(step_done / total_steps)

            zip_buffer.seek(0)
            st.download_button(
                label=f"📦 Download All {len(generated_maps)} Patrol Track Maps (.ZIP)",
                data=zip_buffer,
                file_name=f"Mara_Triangle_Patrol_Tracks_Maps_{report_period.replace(' ', '_')}.zip",
                mime="application/zip"
            )

            for idx, (safe_key, caption_txt, img_bytes) in enumerate(generated_maps):
                st.image(img_bytes, caption=caption_txt, use_container_width=True)
                st.download_button(
                    label=f"📥 Download {safe_key} Map (.png)",
                    data=img_bytes,
                    file_name=f"{safe_key}_Map.png",
                    mime="image/png",
                    key=f"dl_track_map_{idx}_{safe_key}"
                )
                st.divider()
else:
    st.info("👆 Upload your **EarthRanger Patrol Tracks CSV**, **Conservancy Boundary ZIP Shapefiles**, **Roads ZIP**, **Rivers & Luggas ZIP**, **Ranger Posts ZIP**, and **Company Logo** above to generate patrol track maps.")
