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
    southern = ("SOUTH" in up) or (f"{zone}S" in up) or ("FALSE_NORTHING,10000000" in up.replace(" ", "").replace(chr(34), ""))
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

    # Find best name/id column(s)
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
            m = re.match(r"^(\d+)\s*-\s*(.+)$", s)
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

        # Fallback if a row had no structured Bull/Cow dictionary
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

    headers = {"User-Agent": "MaraTriangleRhinoSightingsReport/1.0"}
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
# 5. RENDER INDIVIDUAL-CODED BLACK RHINO SIGHTINGS DISTRIBUTION MAP
# ==============================================================================
def render_rhino_sightings_map(
    title_subject: str,
    sightings_df: pd.DataFrame,
    rhino_color_map: dict[str, str],
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
    label_rhinos_on_map: bool = False,
    logo_scale_pct: int = 55,
    area_fill_alpha: float = 0.22,
    marker_size: int = 85,
) -> bytes:
    """
    Renders a cartographic Black Rhino sightings map where every individual rhino
    (e.g. '1622 - Namunyak', '1631 - Peengo') is plotted in its own distinct color
    matching the monthly breakdown bar chart, alongside all GIS layers.
    """
    bounds_list = []
    if len(sightings_df) > 0:
        bounds_list.append((
            float(sightings_df["Lon"].min()),
            float(sightings_df["Lat"].min()),
            float(sightings_df["Lon"].max()),
            float(sightings_df["Lat"].max()),
        ))
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
        min_lon, min_lat, max_lon, max_lat = 34.80, -1.62, 35.38, -1.22

    lon_span = max(max_lon - min_lon, 0.05)
    lat_span = max(max_lat - min_lat, 0.05)
    min_lon -= lon_span * 0.13
    max_lon += lon_span * 0.13
    min_lat -= lat_span * 0.13
    max_lat += lat_span * 0.13

    # True 1:1 geographic aspect ratio (cos(lat) corrected)
    fig_w, fig_h = 11.8, 8.6
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

    gis_legend_handles = []

    # 2. Conservancy / Sector Boundaries
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
                ax.plot(xs, ys, color=edge_color, linewidth=2.2, linestyle=lstyle, zorder=5 + idx)

            if label_conservancies_on_map:
                cen = get_feature_centroid(feat)
                if cen is not None and min_lon <= cen[0] <= max_lon and min_lat <= cen[1] <= max_lat:
                    ax.text(
                        cen[0],
                        cen[1],
                        str(feat["name"]).upper(),
                        fontsize=7.8,
                        fontweight="bold",
                        color=edge_color,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=edge_color, alpha=0.78, lw=0.8),
                        zorder=9
                    )

        disp_legend_name = ", ".join(unique_names[:3]) if len(unique_names) > 1 else (unique_names[0] if unique_names else layer_label)
        gis_legend_handles.append(
            mpatches.Patch(
                facecolor=edge_color if area_fill_alpha > 0 else "none",
                edgecolor=edge_color,
                alpha=min(area_fill_alpha + 0.12, 0.65) if area_fill_alpha > 0 else 1.0,
                linewidth=2.0,
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
                ax.plot(xs, ys, color="#5C4033", linewidth=1.1, linestyle="-", alpha=0.82, zorder=6)
        gis_legend_handles.append(
            mlines.Line2D([], [], color="#5C4033", linewidth=1.5, linestyle="-", label=roads_label)
        )

    # 4. Rivers & Luggas Overlay
    if rivers_features:
        for feat in rivers_features:
            for ring in feat["parts"]:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.plot(xs, ys, color="#0066CC", linewidth=1.3, linestyle="-", alpha=0.86, zorder=7)
        gis_legend_handles.append(
            mlines.Line2D([], [], color="#0066CC", linewidth=1.7, label=rivers_label)
        )

    # 5. Ranger Posts Overlay
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
                        fontsize=7.0,
                        fontweight="bold",
                        color="#145A32",
                        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#1E8449", alpha=0.82, lw=0.6),
                        zorder=13
                    )
        if pts_x:
            ax.scatter(
                pts_x,
                pts_y,
                s=80,
                marker="^",
                c="#1E8449",
                edgecolors="white",
                linewidths=1.0,
                zorder=11
            )
        gis_legend_handles.append(
            mlines.Line2D(
                [], [],
                color="white",
                marker="^",
                markerfacecolor="#1E8449",
                markeredgecolor="white",
                markersize=9.0,
                label=ranger_posts_label
            )
        )

    # 6. Plot Each Rhino Sighting Colored by Individual ID/Name
    # When multiple rhinos are seen at the exact same GPS coordinate (e.g., mother + calf or group),
    # apply a tiny circular radial jitter (~0.0014 deg) so every individual marker remains clearly visible!
    rhino_legend_handles = []
    if len(sightings_df) > 0:
        work_df = sightings_df.copy().reset_index(drop=True)
        coord_counts = {}
        plot_lons = []
        plot_lats = []
        for _, r in work_df.iterrows():
            key = (round(float(r["Lon"]), 5), round(float(r["Lat"]), 5))
            idx_at_pt = coord_counts.get(key, 0)
            coord_counts[key] = idx_at_pt + 1
            if idx_at_pt == 0:
                plot_lons.append(float(r["Lon"]))
                plot_lats.append(float(r["Lat"]))
            else:
                angle = (idx_at_pt * 2.0 * math.pi) / 6.0
                radius = 0.0014 * (1 + (idx_at_pt - 1) // 6)
                plot_lons.append(float(r["Lon"]) + radius * math.cos(angle))
                plot_lats.append(float(r["Lat"]) + radius * math.sin(angle))

        work_df["_plot_lon"] = plot_lons
        work_df["_plot_lat"] = plot_lats

        counts_by_rhino = work_df["Rhino_Individual"].value_counts()
        for rhino_name, cnt in counts_by_rhino.items():
            sub = work_df[work_df["Rhino_Individual"] == rhino_name]
            col_hex = rhino_color_map.get(rhino_name, "#D92323")
            ax.scatter(
                sub["_plot_lon"],
                sub["_plot_lat"],
                s=marker_size,
                c=col_hex,
                edgecolors="white",
                linewidths=1.15,
                alpha=0.95,
                zorder=15
            )
            if label_rhinos_on_map:
                for _, sr in sub.iterrows():
                    if min_lon <= sr["_plot_lon"] <= max_lon and min_lat <= sr["_plot_lat"] <= max_lat:
                        short_lbl = rhino_name.split(" - ")[-1] if " - " in rhino_name else rhino_name
                        ax.annotate(
                            short_lbl,
                            xy=(sr["_plot_lon"], sr["_plot_lat"]),
                            xytext=(4, 3),
                            textcoords="offset points",
                            fontsize=6.6,
                            fontweight="bold",
                            color="#111111",
                            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=col_hex, alpha=0.82, lw=0.7),
                            zorder=16
                        )

            rhino_legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color="white",
                    marker="o",
                    markerfacecolor=col_hex,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    markersize=8.5,
                    label=f"{rhino_name} ({cnt})"
                )
            )

    # 7. North Arrow & Scale Bar
    draw_compass_and_bearing(ax, x_frac=0.075, y_frac=0.88)
    draw_scale_bar(ax, min_lon, max_lon, min_lat, max_lat)

    # 8. Legend (Bottom-Right, 2 columns if many individuals so nothing overflows)
    all_handles = rhino_legend_handles + gis_legend_handles
    ncol_leg = 2 if len(all_handles) > 12 else 1
    leg = ax.legend(
        handles=all_handles,
        loc="lower right",
        ncol=ncol_leg,
        title="INDIVIDUAL RHINOS & MAP LAYERS",
        frameon=True,
        facecolor="white",
        edgecolor="#1F4E78",
        framealpha=0.94,
        fontsize=7.3 if ncol_leg == 2 else 7.8,
        title_fontsize=8.5,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(22)

    # 9. Title & Coordinate Frame
    total_ind_sightings = len(sightings_df)
    unique_rhinos_cnt = int(sightings_df["Rhino_Individual"].nunique()) if len(sightings_df) > 0 else 0
    unique_events_cnt = int(sightings_df["Report_Id"].nunique()) if len(sightings_df) > 0 else 0

    title_line_1 = f"{title_subject.upper()} — {report_period.upper()}"
    title_line_2 = (
        f"Individual Detections: {total_ind_sightings}   |   "
        f"Unique Rhinos Identified: {unique_rhinos_cnt}   |   "
        f"Sighting Events: {unique_events_cnt}"
    )
    ax.set_title(
        title_line_1 + chr(10) + title_line_2,
        fontsize=12.0,
        fontweight="bold",
        color="#1F4E78",
        pad=12
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Latitude (°S)", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#555555", alpha=0.45, zorder=2)


    add_top_right_logo(ax, logo_bytes, logo_scale_pct=logo_scale_pct)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# 6. RENDER COLOR-MATCHED MONTHLY SIGHTINGS BREAKDOWN BAR CHART
# ==============================================================================
def render_rhino_breakdown_chart(
    summary_df: pd.DataFrame,
    rhino_color_map: dict[str, str],
    report_period: str,
    logo_bytes=None,
    orientation: str = "Horizontal (Most Sighted at Top)",
) -> bytes:
    """
    Renders a bar graph ordered strictly from Most Sighted Individual to Least Sighted Individual,
    using the EXACT same color per individual as the distribution map.
    """
    # Strictly sorted from MOST sighted to LEAST sighted
    ranked_df = summary_df.sort_values(by=["Sightings", "Rhino Individual"], ascending=[False, True]).reset_index(drop=True)
    n_bars = max(len(ranked_df), 1)
    total_det = int(ranked_df["Sightings"].sum())

    if orientation.startswith("Vertical"):
        fig_w = max(10.5, 0.58 * n_bars + 2.2)
        fig, ax = plt.subplots(figsize=(fig_w, 6.2), dpi=220)
        colors = [rhino_color_map.get(name, "#D92323") for name in ranked_df["Rhino Individual"]]
        bars = ax.bar(
            ranked_df["Rhino Individual"],
            ranked_df["Sightings"],
            color=colors,
            edgecolor="#222222",
            linewidth=0.65,
            width=0.64,
            zorder=3
        )
        max_val = int(ranked_df["Sightings"].max()) if len(ranked_df) > 0 else 1
        ax.set_ylim(0, max_val * 1.22 + 0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, (_, r) in zip(bars, ranked_df.iterrows()):
            h = bar.get_height()
            pct = r["Share (%)"]
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max(max_val * 0.02, 0.12),
                f"{int(h)}\n({pct:.1f}%)",
                va="bottom",
                ha="center",
                fontsize=8.0,
                fontweight="bold",
                color="#111111"
            )

        ax.set_xlabel("Individual Rhino (Most Sighted → Least Sighted)", fontsize=9.8, fontweight="bold", labelpad=8)
        ax.set_ylabel("Number of Sightings", fontsize=9.8, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=38, ha="right", rotation_mode="anchor", fontsize=8.5, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.8)
    else:
        # Horizontal bar chart with #1 Most Sighted at the very TOP and Least Sighted at the BOTTOM
        plot_df = ranked_df.iloc[::-1].reset_index(drop=True)
        fig_h = max(4.8, 0.40 * n_bars + 1.9)
        fig, ax = plt.subplots(figsize=(10.8, fig_h), dpi=220)

        colors = [rhino_color_map.get(name, "#D92323") for name in plot_df["Rhino Individual"]]
        bars = ax.barh(
            plot_df["Rhino Individual"],
            plot_df["Sightings"],
            color=colors,
            edgecolor="#222222",
            linewidth=0.65,
            height=0.68,
            zorder=3
        )

        max_val = int(plot_df["Sightings"].max()) if len(plot_df) > 0 else 1
        ax.set_xlim(0, max_val * 1.20 + 0.5)
        ax.grid(axis="x", linestyle="--", alpha=0.45, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, (_, r) in zip(bars, plot_df.iterrows()):
            w = bar.get_width()
            pct = r["Share (%)"]
            sex_tag = f" [{r['Sex']}]" if r.get("Sex") and r["Sex"] not in ("", "Unknown") else ""
            ax.text(
                w + max(max_val * 0.02, 0.15),
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(w)} ({pct:.1f}%){sex_tag}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#111111"
            )

        ax.set_xlabel("Number of Sightings Logged", fontsize=9.8, fontweight="bold")
        ax.set_ylabel("Individual Rhino (Most Sighted at Top → Least Sighted)", fontsize=9.5, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8.8)

    ax.set_title(
        f"RHINO SIGHTINGS RANKING (MOST SIGHTED TO LEAST SIGHTED) — {report_period.upper()}"
        + chr(10)
        + f"Total Individual Sightings: {total_det} across {len(ranked_df)} Identified Individuals (Color-Matched to Map)",
        fontsize=11.5,
        fontweight="bold",
        color="#1F4E78",
        pad=14
    )

    add_top_right_logo(ax, logo_bytes, logo_scale_pct=45)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# STREAMLIT PAGE UI
# ==============================================================================
st.title("🦏 Rhino Details")
st.markdown(
    "Generate a color-coded **Black Rhino Sightings Distribution Map** and matching **Monthly Individual Sightings Breakdown Chart** "
    "by linking your **EarthRanger Rhino Details Export** with your **Rhino IDs Subject List** — processed 100% locally in memory with zero external data sharing."
)

col1, col2 = st.columns(2)
with col1:
    rhino_events_file = st.file_uploader(
        "1️⃣ Upload EarthRanger Rhino Details Event Export (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="rhino_events_uploader"
    )
    rhino_ids_file = st.file_uploader(
        "2️⃣ Upload Rhino IDs Subject List CSV / Excel (e.g., '1622 - NAMUNYAK')",
        type=["csv", "xlsx", "xls"],
        key="rhino_ids_subject_uploader"
    )
    logo_file = st.file_uploader(
        "3️⃣ Upload Mara Conservancy / Company Logo (PNG / JPG — placed in Top-Right)",
        type=["png", "jpg", "jpeg"],
        key="rhino_company_logo_uploader"
    )
    if logo_file is not None:
        st.image(logo_file, caption="Logo loaded (will appear in top-right corner)", width=110)

with col2:
    conservancy_zips = st.file_uploader(
        "4️⃣ Upload Conservancy / Patrol Area Boundary Shapefiles (`.zip` — one or multiple)",
        type=["zip"],
        accept_multiple_files=True,
        key="rhino_conservancy_zip_uploader"
    )
    roads_zip = st.file_uploader(
        "5️⃣ Upload Roads & Tracks Shapefile (`.zip`)",
        type=["zip"],
        key="rhino_roads_zip_uploader"
    )
    rivers_zips = st.file_uploader(
        "6️⃣ Upload Rivers & Luggas Shapefiles (`.zip` — upload Rivers & Luggas together)",
        type=["zip"],
        accept_multiple_files=True,
        key="rhino_rivers_zip_uploader"
    )
    ranger_posts_zip = st.file_uploader(
        "7️⃣ Upload Ranger Posts / Stations Shapefile (`.zip`)",
        type=["zip"],
        key="rhino_ranger_posts_zip_uploader"
    )

with st.sidebar:
    st.subheader("Rhino Map & Chart Settings")
    report_period = st.text_input("Report Period Label", value="November 2026", key="rhino_map_period_label")
    zoom_level = st.slider("OpenTopoMap Tile Zoom Level", min_value=9, max_value=13, value=11, key="rhino_topo_zoom")
    logo_scale_pct = st.slider("Top-Right Logo Size (%)", min_value=25, max_value=100, value=55, step=5, key="rhino_logo_size_pct")
    marker_size = st.slider("Rhino Sighting Dot Size", min_value=45, max_value=160, value=90, step=5, key="rhino_dot_size")
    roads_legend_name = st.text_input("Roads Legend Label", value="Roads & Tracks", key="rhino_roads_legend_txt")
    rivers_legend_name = st.text_input("Rivers & Luggas Legend Label", value="Rivers & Luggas", key="rhino_rivers_legend_txt")
    ranger_legend_name = st.text_input("Ranger Posts Legend Label", value="Ranger Posts", key="rhino_ranger_legend_txt")
    area_fill_opacity = st.slider("Patrol Area Shading Opacity (%)", min_value=0, max_value=55, value=22, step=2, key="rhino_area_shade_alpha")
    show_poly_labels = st.checkbox("Label Conservancy / Patrol Areas on Map", value=True, key="rhino_show_poly_names")
    show_post_labels = st.checkbox("Label Ranger Posts on Map", value=True, key="rhino_show_post_names")
    show_rhino_labels = st.checkbox("Label Individual Rhino Names Next to Dots", value=False, key="rhino_show_dot_names")

if rhino_events_file is not None:
    if rhino_events_file.name.lower().endswith((".xlsx", ".xls")):
        raw_events_df = pd.read_excel(rhino_events_file)
    else:
        raw_events_df = pd.read_csv(rhino_events_file)

    id_lookup = {}
    if rhino_ids_file is not None:
        if rhino_ids_file.name.lower().endswith((".xlsx", ".xls")):
            raw_ids_df = pd.read_excel(rhino_ids_file)
        else:
            raw_ids_df = pd.read_csv(rhino_ids_file)
        id_lookup = build_rhino_id_lookup(raw_ids_df)
    else:
        st.info("ℹ️ Tip: Upload your **Rhino IDs Subject CSV** in box **2️⃣** to map KIF IDs (e.g. `1622`) to individual rhino names (`1622 - Namunyak`).")

    try:
        sightings_df = parse_rhino_sightings_dataframe(raw_events_df, id_lookup)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    if len(sightings_df) == 0:
        st.warning("No valid rhino sighting coordinates were found in the uploaded file.")
        st.stop()

    # Optional sector filter
    available_sectors = sorted([s for s in sightings_df["Sector"].dropna().unique() if str(s).strip()])
    if available_sectors:
        selected_sectors = st.multiselect(
            "Filter by Sector (leave all selected for full monthly map):",
            options=available_sectors,
            default=available_sectors,
            key="rhino_sector_filter"
        )
        if selected_sectors:
            sightings_df = sightings_df[sightings_df["Sector"].isin(selected_sectors)].reset_index(drop=True)

    # Build monthly individual summary table
    total_detections = len(sightings_df)
    summary_rows = []
    for rhino_name, grp in sightings_df.groupby("Rhino_Individual"):
        sex_vals = sorted([s for s in grp["Sex"].dropna().unique() if s])
        sectors_seen = ", ".join(sorted([s for s in grp["Sector"].dropna().unique() if s]))
        stations_seen = ", ".join(sorted([s for s in grp["Station"].dropna().unique() if s]))
        summary_rows.append({
            "Rhino Individual": rhino_name,
            "KIF ID": grp["KIF_ID"].iloc[0],
            "Sex": "/".join(sex_vals),
            "Sightings": len(grp),
            "Share (%)": round(len(grp) / max(total_detections, 1) * 100.0, 2),
            "Sectors Spotted": sectors_seen,
            "Stations": stations_seen,
        })

    rhino_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(by=["Sightings", "Rhino Individual"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Distinct, high-contrast non-river color palette so each rhino stands out clearly on OpenTopoMap
    distinct_rhino_palette = [
        "#D92323", "#E67E22", "#8E44AD", "#C2185B", "#D4AC0D",
        "#16A085", "#D35400", "#6C3483", "#2E7D32", "#AD1457",
        "#B7950B", "#A04000", "#880E4F", "#5D4037", "#00838F",
        "#C0392B", "#F39C12", "#7D3C98", "#1E8449", "#922B21",
        "#6D4C41", "#00695C", "#9C27B0", "#EF6C00", "#4527A0",
    ]
    ordered_rhinos = rhino_summary["Rhino Individual"].tolist()
    rhino_color_map = {
        r_name: distinct_rhino_palette[idx % len(distinct_rhino_palette)]
        for idx, r_name in enumerate(ordered_rhinos)
    }

    # KPI metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Individual Detections", f"{len(sightings_df):,}")
    m2.metric("Unique Rhinos Identified", f"{len(rhino_summary):,}")
    m3.metric("Sighting Events Logged", f"{sightings_df['Report_Id'].nunique():,}")
    top_rhino = rhino_summary.iloc[0]["Rhino Individual"] if len(rhino_summary) > 0 else "—"
    top_cnt = int(rhino_summary.iloc[0]["Sightings"]) if len(rhino_summary) > 0 else 0
    m4.metric("Most Sighted Individual", f"{top_rhino} ({top_cnt})")

    st.subheader(f"1. Monthly Black Rhino Individual Sightings Breakdown ({report_period})")
    st.dataframe(rhino_summary, use_container_width=True, hide_index=True)

    chart_orientation = st.radio(
        "Bar Graph Layout (Most Sighted → Least Sighted):",
        options=["Horizontal (Most Sighted at Top)", "Vertical (Most Sighted Left → Right)"],
        horizontal=True,
        key="rhino_bar_orientation"
    )

    # Load GIS boundary / infrastructure shapefiles
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
                    key=f"rhino_cons_name_{idx}_{zfile.name}"
                )
                chosen_col = st.color_picker(
                    f"Area {idx + 1} Shade & Border Color",
                    value=default_col,
                    key=f"rhino_cons_col_{idx}_{zfile.name}"
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

    # Optional individual color customization expander
    with st.expander("🎨 Customize Individual Rhino Colors (Synced Across Both Map & Chart)", expanded=False):
        p_cols = st.columns(4)
        for idx, r_name in enumerate(ordered_rhinos):
            with p_cols[idx % 4]:
                rhino_color_map[r_name] = st.color_picker(
                    r_name,
                    value=rhino_color_map[r_name],
                    key=f"rhino_ind_color_{idx}_{r_name}"
                )

    # Render & display the Most Sighted -> Least Sighted Bar Graph immediately
    live_chart_png = render_rhino_breakdown_chart(
        summary_df=rhino_summary,
        rhino_color_map=rhino_color_map,
        report_period=report_period,
        logo_bytes=logo_bytes,
        orientation=chart_orientation,
    )
    st.subheader(f"2. Sightings Bar Graph: Most Sighted to Least Sighted ({report_period})")
    st.image(live_chart_png, caption=f"Rhino Sightings Ranked Most Sighted to Least Sighted — {report_period}", use_container_width=True)
    safe_period_live = report_period.replace(" ", "_")
    st.download_button(
        label="📥 Download Sightings Ranking Bar Graph (.png)",
        data=live_chart_png,
        file_name=f"Rhino_Sightings_Ranking_Chart_{safe_period_live}.png",
        mime="image/png",
        key="dl_rhino_live_chart_png"
    )

    st.divider()
    st.subheader(f"3. Color-Matched Rhino Details Distribution ({report_period})")

    if st.button("🦏 Generate Rhino Details Distribution & Full Report Bundle", type="primary", key="btn_gen_rhino_outputs"):
        with st.spinner("Rendering color-matched Black Rhino Sightings Map & Monthly Breakdown Chart..."):
            map_png = render_rhino_sightings_map(
                title_subject="RHINO DETAILS — INDIVIDUAL SIGHTINGS",
                sightings_df=sightings_df,
                rhino_color_map=rhino_color_map,
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
                label_rhinos_on_map=show_rhino_labels,
                logo_scale_pct=logo_scale_pct,
                area_fill_alpha=area_fill_opacity / 100.0,
                marker_size=marker_size,
            )

            chart_png = render_rhino_breakdown_chart(
                summary_df=rhino_summary,
                rhino_color_map=rhino_color_map,
                report_period=report_period,
                logo_bytes=logo_bytes,
                orientation=chart_orientation,
            )

        # Bundle Map + Chart + Summary Table CSV into a single ZIP download
        zip_buf = io.BytesIO()
        safe_period = report_period.replace(" ", "_")
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"Black_Rhino_Sightings_Map_{safe_period}.png", map_png)
            zf.writestr(f"Black_Rhino_Individual_Chart_{safe_period}.png", chart_png)
            zf.writestr(f"Black_Rhino_Sightings_Summary_{safe_period}.csv", rhino_summary.to_csv(index=False))
        zip_buf.seek(0)

        st.download_button(
            label="📦 Download Rhino Map + Breakdown Chart + Summary (.ZIP)",
            data=zip_buf,
            file_name=f"Mara_Triangle_Rhino_Details_{safe_period}.zip",
            mime="application/zip"
        )

        st.image(map_png, caption=f"Rhino Details — Individual Sightings Distribution ({report_period})", use_container_width=True)
        st.download_button(
            label="📥 Download Rhino Details Distribution (.png)",
            data=map_png,
            file_name=f"Rhino_Details_Distribution_{safe_period}.png",
            mime="image/png",
            key="dl_rhino_map_png"
        )

        st.divider()

        st.image(chart_png, caption=f"Black Rhino Individual Sightings Monthly Breakdown — {report_period}", use_container_width=True)
        st.download_button(
            label="📥 Download Individual Breakdown Chart (.png)",
            data=chart_png,
            file_name=f"Black_Rhino_Individual_Chart_{safe_period}.png",
            mime="image/png",
            key="dl_rhino_chart_png"
        )
else:
    st.info("👆 Upload your **EarthRanger Rhino Details Export CSV**, **Rhino IDs Subject CSV**, **GIS Shapefiles (.zip)**, and **Company Logo** above to generate the color-matched Rhino Sightings Map & Monthly Breakdown Chart.")
