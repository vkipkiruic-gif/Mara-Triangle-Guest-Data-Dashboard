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
from matplotlib.collections import LineCollection
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
# 2. PARSE EARTHRANGER PATROL TRACKS (WKT LINESTRING OR LAT/LON POINTS)
# ==============================================================================
def _haversine_km_vec(lat1, lon1, lat2, lon2):
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def derive_patrol_sector(subject_val: str) -> str:
    """
    Maps ranger callsigns like 'Ranger2 Iseiya', 'Ranger4 Iseiya' -> 'Iseiya'
    and 'Ranger3 Ngiro-are', 'Ranger1 Ngiro-are' -> 'Ngiro-are', while preserving
    other patrol group names cleanly when many teams/groups are uploaded.
    """
    s = str(subject_val).strip()
    low = s.lower()
    if "iseya" in low or "iseiya" in low:
        return "Iseiya"
    if "ngiro" in low:
        return "Ngiro-are"
    return s


def prepare_patrol_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Normalizes either:
      - Ecoscope / EarthRanger trajectory CSV (`geometry` with `LINESTRING (lon1 lat1, lon2 lat2)`,
        `dist_meters`, `patrol_subject`, `junk_status`, `patrol_id`, etc.)
      - Standard point-based CSV (`Latitude`, `Longitude`, timestamp)
    Returns `(normalized_df, mode)` where `mode` is `'wkt'` or `'points'`.
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Filter out junk GPS fixes if column exists
    if "junk_status" in df.columns:
        df = df[df["junk_status"].astype(str).str.strip().str.lower() != "true"].copy()

    geom_col = next((c for c in df.columns if c.lower() == "geometry"), None)
    if geom_col is not None and df[geom_col].astype(str).str.contains("LINESTRING", case=False, na=False).any():
        coords = df[geom_col].astype(str).str.extract(
            r"LINESTRING\s*\(\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*\)"
        )
        df["lon1"] = pd.to_numeric(coords[0], errors="coerce")
        df["lat1"] = pd.to_numeric(coords[1], errors="coerce")
        df["lon2"] = pd.to_numeric(coords[2], errors="coerce")
        df["lat2"] = pd.to_numeric(coords[3], errors="coerce")
        df = df.dropna(subset=["lon1", "lat1", "lon2", "lat2"]).copy()

        if "dist_meters" in df.columns:
            df["_seg_km"] = pd.to_numeric(df["dist_meters"], errors="coerce").fillna(0.0) / 1000.0
        else:
            df["_seg_km"] = _haversine_km_vec(
                df["lat1"].to_numpy(), df["lon1"].to_numpy(),
                df["lat2"].to_numpy(), df["lon2"].to_numpy()
            )
        df["Lat"] = df["lat1"]
        df["Lon"] = df["lon1"]
        mode = "wkt"
    else:
        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
        lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
        if not lat_col or not lon_col:
            raise ValueError("Uploaded file must contain either a WKT 'geometry' (LINESTRING) column or 'Latitude' & 'Longitude' columns.")
        df["Lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["Lon"] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=["Lat", "Lon"]).copy()
        df["_seg_km"] = 0.0
        mode = "points"

    # Add smart Patrol Base / Sector column if `patrol_subject` or `subject` exists
    subj_col = next((c for c in df.columns if c.lower() in ("patrol_subject", "subject", "ranger", "team", "unit")), None)
    if subj_col:
        df["Patrol Base / Sector"] = df[subj_col].fillna("Unassigned").apply(derive_patrol_sector)

    return df, mode


def build_group_line_segments(grp_df: pd.DataFrame, mode: str, max_gap_km: float = 3.0):
    """
    Returns `(line_segments_array, bounds_tuple, total_distance_km, unique_patrols_count)`.
    `line_segments_array` is a list/array of `[(lon1, lat1), (lon2, lat2)]` pairs ready for `LineCollection`.
    """
    if len(grp_df) == 0:
        return [], None, 0.0, 0

    unique_patrols = int(grp_df["patrol_id"].nunique()) if "patrol_id" in grp_df.columns else 1

    if mode == "wkt":
        valid_segs = grp_df[grp_df["_seg_km"] <= max_gap_km]
        dist_km = float(valid_segs["_seg_km"].sum())
        if len(valid_segs) == 0:
            return [], None, 0.0, unique_patrols

        p1 = np.column_stack((valid_segs["lon1"].to_numpy(), valid_segs["lat1"].to_numpy()))
        p2 = np.column_stack((valid_segs["lon2"].to_numpy(), valid_segs["lat2"].to_numpy()))
        segs = np.stack((p1, p2), axis=1)  # shape: (N, 2, 2)

        min_lon = float(min(valid_segs["lon1"].min(), valid_segs["lon2"].min()))
        max_lon = float(max(valid_segs["lon1"].max(), valid_segs["lon2"].max()))
        min_lat = float(min(valid_segs["lat1"].min(), valid_segs["lat2"].min()))
        max_lat = float(max(valid_segs["lat1"].max(), valid_segs["lat2"].max()))
        return segs, (min_lon, min_lat, max_lon, max_lat), round(dist_km, 2), unique_patrols

    # Point-based fallback mode
    work = grp_df.copy()
    time_col = next((c for c in work.columns if any(t in c.lower() for t in ["recorded", "time", "date", "timestamp"])), None)
    if time_col:
        work["_dt"] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values(by=["_dt"])

    lats = work["Lat"].to_numpy(dtype=float)
    lons = work["Lon"].to_numpy(dtype=float)
    if len(lats) < 2:
        b = (float(lons[0]), float(lats[0]), float(lons[0]), float(lats[0])) if len(lats) == 1 else None
        return [], b, 0.0, unique_patrols

    dists = _haversine_km_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
    keep = dists <= max_gap_km
    dist_km = float(dists[keep].sum())

    p1 = np.column_stack((lons[:-1][keep], lats[:-1][keep]))
    p2 = np.column_stack((lons[1:][keep], lats[1:][keep]))
    segs = np.stack((p1, p2), axis=1)
    bounds = (float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()))
    return segs, bounds, round(dist_km, 2), unique_patrols


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
# 5. RENDER PATROL TRACKS MAP WITH FAST LINECOLLECTION & ALL GIS LAYERS
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
    track_linewidth: float = 1.6,
) -> bytes:
    """
    `track_series_list` is a list of dicts:
      [{'name': str, 'segments': ndarray(N,2,2), 'bounds': tuple, 'color': str, 'distance_km': float, 'patrol_count': int, 'seg_count': int}, ...]
    """
    bounds_list = []
    for ts in track_series_list:
        if ts["bounds"] is not None:
            bounds_list.append(ts["bounds"])
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

    # 6. Patrol Tracks Overlay via LineCollection (ultra-fast even for 100k+ segments)
    total_dist_km = 0.0
    total_patrols = 0
    track_legend_handles = []

    for ts in track_series_list:
        t_name = ts["name"]
        t_color = ts["color"]
        segs = ts["segments"]
        dist_km = ts["distance_km"]
        p_count = ts["patrol_count"]
        total_dist_km += dist_km
        total_patrols += p_count

        if len(segs) > 0:
            lc_halo = LineCollection(
                segs,
                colors="white",
                linewidths=track_linewidth + 1.2,
                alpha=0.65,
                capstyle="round",
                joinstyle="round",
                zorder=13.5,
            )
            lc = LineCollection(
                segs,
                colors=t_color,
                linewidths=track_linewidth,
                alpha=0.88,
                capstyle="round",
                joinstyle="round",
                zorder=14,
            )
            ax.add_collection(lc_halo)
            ax.add_collection(lc)

        track_legend_handles.append(
            mlines.Line2D(
                [], [],
                color=t_color,
                linewidth=2.4,
                linestyle="-",
                label=f"{t_name} ({dist_km:,.1f} km | {p_count} patrols)"
            )
        )

    if len(track_legend_handles) <= 10:
        legend_handles = track_legend_handles + legend_handles
    else:
        legend_handles = track_legend_handles[:8] + [
            mlines.Line2D([], [], color="#D92323", linewidth=2.4, label=f"+ {len(track_legend_handles) - 8} more patrol groups")
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
    title_line_2 = f"Total Distance Covered: {total_dist_km:,.1f} km   |   Patrols Logged: {total_patrols}   |   Basemap: OpenTopoMap"
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
    "Generate cartographic patrol track maps for each patrol base/sector (e.g. **Ngiro-are**, **Iseiya**) or ranger unit over **OpenTopoMap** (`tile.opentopomap.org`), "
    "complete with **shaded conservancy/patrol boundaries**, **roads**, **rivers & luggas**, **ranger posts**, **company logo**, **North/bearing rose**, and **scale bar**."
)

col1, col2 = st.columns(2)
with col1:
    tracks_file = st.file_uploader(
        "1️⃣ Upload EarthRanger Patrol Tracks CSV / Excel (WKT LINESTRING or Lat/Lon)",
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
    track_linewidth = st.slider("Patrol Track Line Width", min_value=0.8, max_value=3.5, value=1.5, step=0.1, key="tracks_line_w")
    max_gap_km = st.number_input("Ignore Single Segment Jump > (km)", min_value=0.5, max_value=25.0, value=3.0, step=0.5, key="tracks_max_gap_km")
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

    try:
        clean_df, track_mode = prepare_patrol_dataframe(raw_df)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    # Build ordered list of meaningful grouping columns (putting `Patrol Base / Sector` first, followed by `patrol_subject`, `patrol_type__display`, `patrol_title`, etc.)
    preferred_order = [
        "Patrol Base / Sector",
        "patrol_subject",
        "patrol_type__display",
        "patrol_title",
        "subject",
        "patrol_type",
    ]
    exclude_cols = {
        "Unnamed: 0", "groupby_col", "segment_start", "segment_end",
        "timespan_seconds", "dist_meters", "speed_kmhr", "heading",
        "junk_status", "nsd", "patrol_end_time", "patrol_id",
        "patrol_serial_number", "patrol_start_time", "patrol_status",
        "source", "subject_id", "patrol_traj_colormap", "geometry",
        "lon1", "lat1", "lon2", "lat2", "_seg_km", "Lat", "Lon",
    }
    group_options = [c for c in preferred_order if c in clean_df.columns]
    for c in clean_df.columns:
        if c not in group_options and c not in exclude_cols:
            group_options.append(c)

    group_col = st.selectbox(
        "Group Patrol Tracks By (choose 'Patrol Base / Sector' for Ngiro-are & Iseiya, or 'patrol_subject' for all individual teams):",
        options=group_options,
        index=0,
        key="tracks_group_col_select"
    )

    clean_df["_track_group"] = clean_df[group_col].fillna("Unassigned").astype(str).str.strip()

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
    group_cache = {}
    for grp_name, grp_df in clean_df.groupby("_track_group"):
        segs, bounds, dist_km, p_cnt = build_group_line_segments(grp_df, mode=track_mode, max_gap_km=max_gap_km)
        group_cache[grp_name] = {
            "segments": segs,
            "bounds": bounds,
            "distance_km": dist_km,
            "patrol_count": p_cnt,
            "seg_count": len(segs),
        }
        summary_rows.append({
            "Patrol Group / Sector": grp_name,
            "Patrols Logged": p_cnt,
            "Track Segments": len(segs),
            "Total Distance Covered (km)": dist_km,
        })

    tracks_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(by=["Total Distance Covered (km)", "Track Segments"], ascending=False)
        .reset_index(drop=True)
    )
    total_all_km = tracks_summary["Total Distance Covered (km)"].sum() if len(tracks_summary) > 0 else 0.0
    total_all_patrols = int(clean_df["patrol_id"].nunique()) if "patrol_id" in clean_df.columns else int(tracks_summary["Patrols Logged"].sum())

    tracks_summary["Share of Distance (%)"] = (
        (tracks_summary["Total Distance Covered (km)"] / total_all_km * 100).round(2)
        if total_all_km > 0 else 0.0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Track Segments", f"{len(clean_df):,}")
    m2.metric("Patrols Logged", f"{total_all_patrols:,}")
    m3.metric("Total Patrol Distance", f"{total_all_km:,.1f} km")
    active_layers = len(conservancies_loaded) + int(len(roads_features) > 0) + int(len(rivers_features) > 0) + int(len(ranger_features) > 0)
    m4.metric("Active GIS Map Layers", f"{active_layers} layers")

    st.subheader(f"1. Patrol Effort & Distance Summary ({report_period})")
    st.dataframe(tracks_summary, use_container_width=True, hide_index=True)

    st.divider()

    # --- GENERATE PATROL TRACK MAPS ---
    st.subheader(f"2. Patrol Tracks Distribution Maps ({report_period})")
    all_groups_list = tracks_summary["Patrol Group / Sector"].tolist()

    # Auto-select meaningful groups (>= 1 km distance, e.g., Iseiya & Ngiro-are, or top 8 when many groups exist)
    meaningful_groups = tracks_summary[tracks_summary["Total Distance Covered (km)"] >= 1.0]["Patrol Group / Sector"].tolist()
    default_selected = meaningful_groups[:8] if meaningful_groups else all_groups_list[:6]

    include_combined_map = st.checkbox(
        "Also Generate a Combined All-Selected Patrol Tracks Overview Map",
        value=True,
        key="include_combined_tracks_map"
    )

    selected_groups = st.multiselect(
        "Select Patrol Groups / Sectors to Generate Individual Maps For:",
        options=all_groups_list,
        default=default_selected,
        key="selected_track_groups"
    )

    track_palette = [
        "#D92323", "#1F618D", "#8E44AD", "#16A085", "#D35400",
        "#C0392B", "#2980B9", "#27AE60", "#884EA0", "#B03A2E"
    ]

    if selected_groups:
        if logo_bytes or _find_repo_logo_bytes():
            st.success("✅ Mara Conservancy Logo active — it will appear in the top-right corner of every generated map.")
        else:
            st.warning("⚠️ No logo uploaded yet in box **2️⃣** above. Upload your Mara Conservancy logo PNG/JPG in box **2️⃣** (or place `mara_conservancy_logo.png` in the repo root) if you want it in the top-right corner.")

        if st.button("🗺️ Generate All Selected Patrol Track Maps & ZIP Bundle", type="primary", key="btn_gen_track_maps"):
            zip_buffer = io.BytesIO()
            generated_maps = []

            total_steps = len(selected_groups) + (1 if include_combined_map else 0)
            progress = st.progress(0)
            step_done = 0

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # 1. Combined Overview Map of all selected patrol groups
                if include_combined_map:
                    combined_series = []
                    for idx, grp_name in enumerate(selected_groups):
                        info = group_cache[grp_name]
                        combined_series.append({
                            "name": grp_name,
                            "segments": info["segments"],
                            "bounds": info["bounds"],
                            "color": track_palette[idx % len(track_palette)],
                            "distance_km": info["distance_km"],
                            "patrol_count": info["patrol_count"],
                            "seg_count": info["seg_count"],
                        })
                    combined_png = render_patrol_track_map(
                        title_subject="ALL SELECTED SECTORS",
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
                    )
                    generated_maps.append(("Combined_All_Patrol_Tracks", "All Selected Patrol Tracks (Combined Overview)", combined_png))
                    zf.writestr(f"All_Patrol_Tracks_Overview_{report_period.replace(' ', '_')}.png", combined_png)
                    step_done += 1
                    progress.progress(step_done / total_steps)

                # 2. Individual Map per Patrol Group / Sector
                for idx, grp_name in enumerate(selected_groups):
                    info = group_cache[grp_name]
                    color = track_palette[idx % len(track_palette)]
                    single_series = [{
                        "name": grp_name,
                        "segments": info["segments"],
                        "bounds": info["bounds"],
                        "color": color,
                        "distance_km": info["distance_km"],
                        "patrol_count": info["patrol_count"],
                        "seg_count": info["seg_count"],
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
                    )
                    safe_grp = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in grp_name).strip().replace(" ", "_")
                    generated_maps.append((
                        safe_grp,
                        f"{grp_name} — Distance Covered: {info['distance_km']:,.1f} km ({info['patrol_count']} patrols | {info['seg_count']:,} segments)",
                        png_bytes
                    ))
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
