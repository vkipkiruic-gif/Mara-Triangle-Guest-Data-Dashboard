import ast
import io
import math
import os
import re
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
    southern = ("SOUTH" in up) or (f"{zone}S" in up) or ("FALSE_NORTHING,10000000" in up.replace(" ", "").replace('"', ""))
    return (zone, southern)


def _read_dbf_records(dbf_path: str) -> list[dict]:
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
    """Reads a zipped ESRI shapefile (.shp, .dbf, .prj) in pure Python."""
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

                if shape_type in (1, 11, 21) and len(content) >= 20:
                    x, y = struct.unpack("<dd", content[4:20])
                    lon, lat = to_lonlat(x, y)
                    features.append({"geom_type": "Point", "parts": [[(lon, lat)]], "name": feat_name})

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
# 2. PARSE EARTHRANGER BLACK RHINO SIGHTINGS & MATCH INDIVIDUAL KIF IDs -> NAMES
# ==============================================================================
def build_rhino_id_lookup(ids_df: pd.DataFrame) -> dict[str, str]:
    """
    Parses the Rhino IDs subject CSV (e.g., '1622 - NAMUNYAK', '1631 - PEENGO')
    into a clean dictionary mapping KIF ID string -> formatted display label ('1622 - Namunyak').
    """
    lookup = {}
    if ids_df is None or len(ids_df) == 0:
        return lookup

    cols = [str(c).strip() for c in ids_df.columns]
    ids_df = ids_df.copy()
    ids_df.columns = cols

    id_col = next((c for c in cols if c.lower() in ("id", "kif_id", "kifid", "rhino_id", "number")), None)
    name_col = next((c for c in cols if "name" in c.lower() or "rhino" in c.lower() or "subject" in c.lower()), cols[0])

    for _, row in ids_df.iterrows():
        raw_name = row.get(name_col)
        if pd.isna(raw_name):
            continue
        s = str(raw_name).strip()
        if not s:
            continue

        if id_col and pd.notna(row.get(id_col)):
            kif = str(row.get(id_col)).strip().split(".")[0]
            clean_nm = re.sub(rf"^{re.escape(kif)}\s*-\s*", "", s).strip().title()
            lookup[kif] = f"{kif} - {clean_nm}" if clean_nm else f"ID {kif}"
        else:
            m = re.match(r"^(\d+)\s*-\s*(.+)\$", s)
            if m:
                kif = m.group(1).strip()
                clean_nm = m.group(2).strip().title()
                lookup[kif] = f"{kif} - {clean_nm}"
            else:
                lookup[s] = s.title()
    return lookup


def parse_rhino_sightings_dataframe(events_df: pd.DataFrame, id_lookup: dict[str, str]) -> pd.DataFrame:
    """
    Expands EarthRanger Rhino Details Event Exports (`Sighting_Details_-_Bulls`,
    `Sighting_Details_-_Cows`, `Sighting_Details_-_Unsexed_Individuals`) into one row per
    individual rhino sighting, matched to `id_lookup` by KIF ID.
    """
    df = events_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lat_col = next((c for c in df.columns if c.lower() == "latitude" or "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if c.lower() == "longitude" or "lon" in c.lower()), None)
    if not lat_col or not lon_col:
        raise ValueError("Events file must contain 'Latitude' and 'Longitude' columns.")

    df["Lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["Lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["Lat", "Lon"]).copy()

    date_col = next((c for c in df.columns if "reported_at" in c.lower() or "time" in c.lower() or "date" in c.lower()), None)

    detail_specs = [
        ("Sighting_Details_-_Bulls", "Bull"),
        ("Sighting_Details_-_Cows", "Cow"),
        ("Sighting_Details_-_Unsexed_Individuals", "Unsexed"),
    ]
    has_detail_cols = any(col in df.columns for col, _ in detail_specs)

    records = []
    for _, row in df.iterrows():
        event_id = row.get("Report_Id", "")
        dt_val = row.get(date_col, "") if date_col else ""
        sector_val = str(row.get("Sector", "")).strip() if pd.notna(row.get("Sector")) else ""
        station_val = str(row.get("Station", "")).strip() if pd.notna(row.get("Station")) else ""
        loc_val = str(row.get("Location", "")).strip() if pd.notna(row.get("Location")) else ""
        area_val = str(row.get("Area", "")).strip() if pd.notna(row.get("Area")) else ""

        matched_any = False
        if has_detail_cols:
            for col_name, sex_label in detail_specs:
                if col_name not in df.columns:
                    continue
                raw_cell = row.get(col_name)
                if pd.isna(raw_cell) or not str(raw_cell).strip():
                    continue

                for chunk in str(raw_cell).split(";"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    matched_any = True
                    parsed_dict = {}
                    try:
                        parsed_dict = ast.literal_eval(chunk)
                    except Exception:
                        parsed_dict = {}

                    kif_id = ""
                    for k, v in parsed_dict.items():
                        if "kifid" in str(k).lower() and v is not None and str(v).strip():
                            kif_id = str(v).strip().split(".")[0]
                            break

                    if kif_id:
                        rhino_display = id_lookup.get(kif_id, f"ID {kif_id}")
                    else:
                        rhino_display = f"Unidentified ({sex_label})"

                    age_code = (
                        parsed_dict.get("known_age_bull")
                        or parsed_dict.get("known_age_cow")
                        or parsed_dict.get("known_age_unsexed")
                        or ""
                    )
                    records.append({
                        "Report_Id": event_id,
                        "Reported_At": dt_val,
                        "Lat": float(row["Lat"]),
                        "Lon": float(row["Lon"]),
                        "Area": area_val,
                        "Sector": sector_val,
                        "Station": station_val,
                        "Location": loc_val,
                        "Sex": sex_label,
                        "KIF_ID": kif_id if kif_id else "Unidentified",
                        "Rhino_Individual": rhino_display,
                        "Age_Class": str(age_code).upper() if age_code else "",
                        "Behaviour": str(parsed_dict.get("behaviour", "")).replace("_", " ").title(),
                        "Condition": str(parsed_dict.get("condition", "")).title(),
                    })

        if not matched_any:
            grp_sz = int(pd.to_numeric(row.get("Group_Size", 1), errors="coerce") or 1)
            for _ in range(max(grp_sz, 1)):
                records.append({
                    "Report_Id": event_id,
                    "Reported_At": dt_val,
                    "Lat": float(row["Lat"]),
                    "Lon": float(row["Lon"]),
                    "Area": area_val,
                    "Sector": sector_val,
                    "Station": station_val,
                    "Location": loc_val,
                    "Sex": "Unknown",
                    "KIF_ID": "Unidentified",
                    "Rhino_Individual": "Unidentified Rhino",
                    "Age_Class": "",
                    "Behaviour": "",
                    "Condition": "",
                })

    return pd.DataFrame(records)


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

    headers = {"User-Agent": "MaraTriangleRhinoDetailsReport/1.0"}
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
    scale = target_h /
