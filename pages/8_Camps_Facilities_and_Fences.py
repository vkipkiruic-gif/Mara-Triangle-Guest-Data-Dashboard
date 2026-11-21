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
    """Reads a zipped ESRI shapefile (.shp, .dbf, .prj) in pure Python, keeping all DBF attributes."""
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
                    features.append({"geom_type": "Point", "parts": [[(lon, lat)]], "name": feat_name, "attrs": attrs})

                elif shape_type in (8, 18, 28) and len(content) >= 40:
                    num_pts = struct.unpack("<I", content[36:40])[0]
                    pts = []
                    for i in range(num_pts):
                        off = 40 + i * 16
                        if off + 16 <= len(content):
                            x, y = struct.unpack("<dd", content[off:off + 16])
                            pts.append(to_lonlat(x, y))
                    if pts:
                        features.append({"geom_type": "Point", "parts": [pts], "name": feat_name, "attrs": attrs})

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
                        features.append({"geom_type": gtype, "parts": rings, "name": feat_name, "attrs": attrs})

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
# 2. PARSE CAMPS & FACILITIES CSV + MARA ELEPHANT PROJECT (MEP) FENCES DATA
# ==============================================================================
def parse_camps_facilities_dataframe(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """
    Parses EarthRanger / Park Camps & Facilities CSV/Excel into a standardized DataFrame:
    Columns: Name, Facility_Type, Lat, Lon, Reported_By, Report_Id
    """
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    lat_col = next((c for c in work.columns if c.lower() == "latitude" or "lat" in c.lower()), None)
    lon_col = next((c for c in work.columns if c.lower() == "longitude" or "lon" in c.lower()), None)
    if not lat_col or not lon_col:
        raise ValueError("Camps & Facilities file must contain 'Latitude' and 'Longitude' columns.")

    name_col = next(
        (c for c in work.columns if any(k in c.lower() for k in ["name_of_the_facility", "facility_name", "camp_name", "name"])),
        None
    )
    type_col = next(
        (c for c in work.columns if any(k in c.lower() for k in ["type_/_description", "facility_type", "type", "description", "category"])),
        None
    )

    work["Lat"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["Lon"] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=["Lat", "Lon"]).copy()

    if name_col:
        work["Name"] = work[name_col].fillna("Unnamed Facility").astype(str).str.strip()
        work["Name"] = work["Name"].str.rstrip(".").str.strip()
    else:
        work["Name"] = [f"Facility {i + 1}" for i in range(len(work))]

    if type_col:
        work["Facility_Type"] = work[type_col].fillna("Camp / Facility").astype(str).str.strip()
        work["Facility_Type"] = work["Facility_Type"].replace({
            "Eco Lodge": "Eco-Lodge",
            "Eco camp": "Eco Camp",
            "Tented camp": "Tented Camp",
        })
    else:
        work["Facility_Type"] = "Camp / Facility"

    work["Reported_By"] = work["Reported_By"].fillna("").astype(str).str.strip() if "Reported_By" in work.columns else ""
    work["Report_Id"] = work["Report_Id"].fillna("").astype(str).str.strip() if "Report_Id" in work.columns else ""

    if deduplicate:
        work["_lat_r"] = work["Lat"].round(5)
        work["_lon_r"] = work["Lon"].round(5)
        work["_name_norm"] = work["Name"].str.lower()
        work = work.drop_duplicates(subset=["_name_norm", "_lat_r", "_lon_r"], keep="first").drop(
            columns=["_lat_r", "_lon_r", "_name_norm"]
        )

    return work.reset_index(drop=True)


def classify_mep_fence_type(raw_type: str) -> tuple[str, str, str]:
    """
    Classifies Mara Elephant Project (MEP) fence codes into clean legend labels:
      (Broad_Category, Detailed_Subtype_Label, Strand_Detail)
    """
    s = str(raw_type or "").strip()
    low = s.lower()

    strand_match = re.search(r"(\d+)\s*_?\s*strand", low)
    strand_label = f"{strand_match.group(1)}-Strand" if strand_match else ("Unspecified Strands" if "unknown" in low else "")

    if "electric" in low:
        broad = "Electric Fence"
        if strand_match:
            detail = f"Electric Fence ({strand_match.group(1)}-Strand)"
        elif "unknown" in low:
            detail = "Electric Fence (Unspecified Strands)"
        else:
            detail = "Electric Fence"
    elif "wire" in low:
        broad = "Wire Fence"
        if strand_match:
            detail = f"Wire Fence ({strand_match.group(1)}-Strand)"
        else:
            detail = "Wire Fence"
    elif "cactus" in low or "live" in low or "hedge" in low:
        broad = "Cactus / Live Fence"
        detail = "Cactus / Live Fence"
    elif "defenc" in low or "defens" in low:
        broad = "Defenced Fence"
        detail = "Defenced Fence"
    elif "stone" in low or "wall" in low:
        broad = "Stone Wall / Fence"
        detail = "Stone Wall / Fence"
    elif "wood" in low or "post" in low or "timber" in low:
        broad = "Woodpost Fence"
        detail = "Woodpost Fence"
    elif not s or low in ("nan", "none", "null", "other", "unknown"):
        broad = "Other Fence"
        detail = "Other Fence"
    else:
        clean_lbl = re.sub(r"^fence[_\s-]*", "", s, flags=re.IGNORECASE).replace("_", " ").strip().title()
        if not clean_lbl or clean_lbl.lower() == "other":
            broad = "Other Fence"
            detail = "Other Fence"
        else:
            broad = f"{clean_lbl} Fence" if not clean_lbl.lower().endswith("fence") else clean_lbl
            detail = broad

    return broad, detail, strand_label


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def compute_polyline_length_km(parts: list[list[tuple[float, float]]]) -> float:
    total_km = 0.0
    for ring in parts:
        for i in range(len(ring) - 1):
            total_km += _haversine_km(ring[i][0], ring[i][1], ring[i + 1][0], ring[i + 1][1])
    return total_km


def process_mep_fence_shapefile_features(raw_features: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    """
    Enriches shapefile features from the Mara Elephant Project (MEP) fencing dataset
    with standardized Electric vs Wire classification, active status, and segment length in km.
    """
    enriched = []
    rows = []
    for feat in raw_features:
        attrs = feat.get("attrs", {}) or {}
        attr_lower = {str(k).lower(): v for k, v in attrs.items()}

        raw_type = (
            attr_lower.get("type")
            or attr_lower.get("fence_type")
            or attr_lower.get("fencetype")
            or attr_lower.get("category")
            or feat.get("name", "")
        )
        broad_cat, detail_lbl, _ = classify_mep_fence_type(raw_type)

        raw_active = str(attr_lower.get("active", "")).strip().lower()
        if raw_active in ("1", "true", "yes", "active"):
            active_status = "Active (1)"
        elif raw_active in ("0", "false", "no", "inactive"):
            active_status = "Inactive / Historical (0)"
        else:
            active_status = "Unspecified"

        length_km = compute_polyline_length_km(feat.get("parts", []))
        if length_km == 0.0 and attr_lower.get("shape__len"):
            try:
                length_km = float(attr_lower["shape__len"]) * 111.1
            except Exception:
                length_km = 0.0

        source_org = str(attr_lower.get("spatial_da") or "MEP").strip() or "MEP"
        collect_method = str(attr_lower.get("collect_me") or "").strip()
        collect_date = str(attr_lower.get("collect_da") or "").strip()

        feat_copy = dict(feat)
        feat_copy["fence_raw_type"] = str(raw_type)
        feat_copy["fence_broad"] = broad_cat
        feat_copy["fence_detail"] = detail_lbl
        feat_copy["fence_active"] = active_status
        feat_copy["length_km"] = length_km
        enriched.append(feat_copy)

        rows.append({
            "Broad Category": broad_cat,
            "Detailed Fence Type": detail_lbl,
            "Raw MEP Code": str(raw_type),
            "Status": active_status,
            "Length (km)": round(length_km, 3),
            "Collection Method": collect_method,
            "Collection Date": collect_date,
            "Data Source": "Mara Elephant Project (MEP)" if source_org.upper() == "MEP" else source_org,
        })

    return enriched, pd.DataFrame(rows)


def parse_mep_fence_attribute_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the Mara Elephant Project (MEP) fence attribute table CSV (e.g., `fenced Mt.csv`).
    """
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    col_map = {c.lower(): c for c in work.columns}

    type_col = col_map.get("type") or col_map.get("fence_type") or work.columns[0]
    active_col = col_map.get("active")
    len_col = col_map.get("shape__len") or col_map.get("shape_leng") or col_map.get("length")
    method_col = col_map.get("collect_me")
    date_col = col_map.get("collect_da")

    rows = []
    for _, r in work.iterrows():
        raw_t = str(r.get(type_col, "")).strip()
        broad_cat, detail_lbl, _ = classify_mep_fence_type(raw_t)

        if active_col and pd.notna(r.get(active_col)):
            act_v = str(r.get(active_col)).strip().lower()
            status = "Active (1)" if act_v in ("1", "1.0", "true", "yes") else "Inactive / Historical (0)"
        else:
            status = "Unspecified"

        est_km = 0.0
        if len_col and pd.notna(r.get(len_col)):
            try:
                val = float(r.get(len_col))
                est_km = val * 111.1 if val < 20 else val / 1000.0
            except Exception:
                est_km = 0.0

        rows.append({
            "Broad Category": broad_cat,
            "Detailed Fence Type": detail_lbl,
            "Raw MEP Code": raw_t,
            "Status": status,
            "Est. Length (km)": round(est_km, 3),
            "Collection Method": str(r.get(method_col, "")).strip() if method_col and pd.notna(r.get(method_col)) else "",
            "Collection Date": str(r.get(date_col, "")).strip() if date_col and pd.notna(r.get(date_col)) else "",
            "Data Source": "Mara Elephant Project (MEP)",
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 3. OPENTOPOMAP TILE STITCHER ([https://tile.opentopomap.org/](https://tile.opentopomap.org/){z}/{x}/{y}.png)
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

    while (x_max - x_min + 1) * (y_max - y_min + 1) > 64 and zoom > 7:
        zoom -= 1
        x_min, y_max = deg2num(min_lat, min_lon, zoom)
        x_max, y_min = deg2num(max_lat, max_lon, zoom)

    cols = max(x_max - x_min + 1, 1)
    rows = max(y_max - y_min + 1, 1)
    canvas = Image.new("RGB", (cols * 256, rows * 256), (242, 239, 233))

    headers = {"User-Agent": "MaraTriangleCampsFencesMap/1.0"}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"[https://tile.opentopomap.org/](https://tile.opentopomap.org/){zoom}/{x}/{y}.png"
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
def draw_compass_and_bearing(ax, x_frac=0.06, y_frac=0.89):
    ax_inset = ax.inset_axes([x_frac - 0.045, y_frac - 0.065, 0.09, 0.13], transform=ax.transAxes)
    ax_inset.set_xlim(-1.45, 1.45)
    ax_inset.set_ylim(-1.45, 1.45)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")

    bg = mpatches.Circle((0, 0), 1.32, facecolor="white", edgecolor="#1F4E78", linewidth=1.3, alpha=0.92, zorder=1)
    ax_
