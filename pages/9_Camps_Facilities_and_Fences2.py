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
import matplotlib.path as mpath
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


def _web_mercator_to_wgs84(x: float, y: float):
    """Converts EPSG:3857 (Web Mercator) meters to WGS84 Lon/Lat."""
    lon = (x / 20037508.342789244) * 180.0
    lat = math.degrees(2.0 * math.atan(math.exp((y / 20037508.342789244) * math.pi)) - math.pi / 2.0)
    return lon, lat


def _detect_crs_info(prj_text: str):
    """Detects CRS from .prj text: ('WGS84', None), ('WEBMERCATOR', None), or ('UTM', (zone, southern))."""
    if not prj_text:
        return None
    up = prj_text.upper()
    if any(k in up for k in ["3857", "900913", "PSEUDO_MERCATOR", "AUXILIARY_SPHERE", "POPULAR_VISUALISATION"]):
        return ("WEBMERCATOR", None)
    if "UTM" in up or "TRANSVERSE_MERCATOR" in up:
        zone = 36
        for z in range(1, 61):
            if f"ZONE_{z}" in up or f"ZONE {z}" in up or f"{z}S" in up or f"{z}N" in up:
                zone = z
                break
        southern = (
            ("SOUTH" in up)
            or (f"{zone}S" in up)
            or ("FALSE_NORTHING,10000000" in up.replace(" ", "").replace(chr(34), ""))
        )
        return ("UTM", (zone, southern))
    if "GEOGCS" in up and "PROJCS" not in up:
        return ("WGS84", None)
    return None


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
    """Reads a zipped ESRI shapefile (.shp, .dbf, .prj) in pure Python, supporting WGS84, UTM & Web Mercator."""
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
        crs_info = _detect_crs_info(prj_text)

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
            if abs(x) <= 180.0 and abs(y) <= 90.0:
                return (x, y)
            if crs_info is not None and crs_info[0] == "WEBMERCATOR":
                return _web_mercator_to_wgs84(x, y)
            if crs_info is not None and crs_info[0] == "UTM":
                zone, southern = crs_info[1]
                return _utm_to_wgs84(x, y, zone=zone, southern=southern)
            if 3_200_000 <= abs(x) <= 5_000_000 and abs(y) <= 1_200_000:
                return _web_mercator_to_wgs84(x, y)
            return _utm_to_wgs84(x, y, zone=36, southern=True)

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


def get_features_bounds(features: list[dict], clip_near_box=None):
    """
    Computes (min_lon, min_lat, max_lon, max_lat) for features.
    If `clip_near_box` = (ref_min_lon, ref_min_lat, ref_max_lon, ref_max_lat) is passed,
    only features intersecting within 0.45° of that reference box are included so far-away
    polygons never zoom the map out.
    """
    xs, ys = [], []
    for feat in features:
        f_xs, f_ys = [], []
        for ring in feat["parts"]:
            for lon, lat in ring:
                if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                    f_xs.append(lon)
                    f_ys.append(lat)
        if not f_xs:
            continue
        f_min_x, f_max_x = min(f_xs), max(f_xs)
        f_min_y, f_max_y = min(f_ys), max(f_ys)

        if clip_near_box is not None:
            pad = 0.45
            if (
                f_max_x < clip_near_box[0] - pad
                or f_min_x > clip_near_box[2] + pad
                or f_max_y < clip_near_box[1] - pad
                or f_min_y > clip_near_box[3] + pad
            ):
                continue

        xs.extend([f_min_x, f_max_x])
        ys.extend([f_min_y, f_max_y])

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
# 2. PARSE CAMPS, AIRSTRIPS CSV & MARA ELEPHANT PROJECT (MEP) FENCES DATA
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


def parse_airstrips_dataframe(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """
    Parses Airstrips CSV/Excel into a standardized DataFrame:
    Columns: Name, Code, Surface, Lat, Lon
    """
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    lat_col = next((c for c in work.columns if c.lower() == "latitude" or "lat" in c.lower()), None)
    lon_col = next((c for c in work.columns if c.lower() == "longitude" or "lon" in c.lower()), None)
    if not lat_col or not lon_col:
        raise ValueError("Airstrips file must contain 'Latitude' and 'Longitude' columns.")

    name_col = next(
        (c for c in work.columns if any(k in c.lower() for k in ["airstrip_name", "name_of_airstrip", "facility_name", "airstrip", "name"])),
        None
    )
    code_col = next(
        (c for c in work.columns if any(k in c.lower() for k in ["icao", "iata", "code", "id"])),
        None
    )
    surf_col = next(
        (c for c in work.columns if any(k in c.lower() for k in ["surface", "type", "runway", "status", "description"])),
        None
    )

    work["Lat"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["Lon"] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=["Lat", "Lon"]).copy()

    if name_col:
        work["Name"] = work[name_col].fillna("Airstrip").astype(str).str.strip().str.rstrip(".").str.strip()
    else:
        work["Name"] = [f"Airstrip {i + 1}" for i in range(len(work))]

    work["Code"] = work[code_col].fillna("").astype(str).str.strip() if code_col else ""
    work["Surface"] = work[surf_col].fillna("Airstrip").astype(str).str.strip() if surf_col else "Airstrip"

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
# 3. RELIABLE BRIGHT BASEMAP TILE STITCHER (OPENTOPOMAP / CARTO / ESRI TOPO)
# ==============================================================================
BASEMAP_PROVIDERS = {
    "OpenTopoMap (Bright Contour & Terrain — Recommended)": [
        "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    ],
    "CartoDB Voyager (Clean Light Pastel Terrain)": [
        "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    ],
    "Esri World Topographic": [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "https://tile.opentopomap.org/{z}/{x}/{y}.png",
    ],
    "Esri World Imagery (Satellite)": [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "https://tile.opentopomap.org/{z}/{x}/{y}.png",
    ],
}


@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_single_tile_bytes(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MaraConservationGIS/1.0)",
        "Accept": "image/png,image/jpeg,image/*,*/*;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.read()


def fetch_basemap_tiles(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    zoom: int = 11,
    provider_label: str = "OpenTopoMap (Bright Contour & Terrain — Recommended)",
):
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
    canvas = Image.new("RGB", (cols * 256, rows * 256), (247, 246, 242))

    url_templates = BASEMAP_PROVIDERS.get(
        provider_label,
        BASEMAP_PROVIDERS["OpenTopoMap (Bright Contour & Terrain — Recommended)"]
    )

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            for tmpl in url_templates:
                url = tmpl.format(z=zoom, x=x, y=y)
                try:
                    raw_bytes = _fetch_single_tile_bytes(url)
                    tile_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    canvas.paste(tile_img, ((x - x_min) * 256, (y - y_min) * 256))
                    break
                except Exception:
                    continue

    north_lat, west_lon = num2deg(x_min, y_min, zoom)
    south_lat, east_lon = num2deg(x_max + 1, y_max + 1, zoom)
    return canvas, (west_lon, east_lon, south_lat, north_lat)


# ==============================================================================
# 4. CARTOGRAPHIC HELPERS: NORTH ARROW + BEARING ROSE, SCALE BAR, LOGO
# ==============================================================================
def draw_compass_and_bearing(ax, x_frac=0.055, y_frac=0.895):
    ax_inset = ax.inset_axes([x_frac - 0.045, y_frac - 0.065, 0.09, 0.13], transform=ax.transAxes)
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

    target_km = map_width_km * 0.20
    nice_steps = [1, 2, 5, 10, 15, 20, 25, 30, 50, 100]
    scale_km = min(nice_steps, key=lambda s: abs(s - target_km))
    bar_deg = scale_km / max(km_per_deg_lon, 1e-6)

    x0 = min_lon + (max_lon - min_lon) * 0.03
    y0 = min_lat + (max_lat - min_lat) * 0.055
    h = (max_lat - min_lat) * 0.011

    pad_x = (max_lon - min_lon) * 0.012
    pad_y = (max_lat - min_lat) * 0.014
    box = mpatches.FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y),
        bar_deg + 2 * pad_x,
        h + 2.8 * pad_y,
        boxstyle="round,pad=0.002",
        facecolor="white",
        edgecolor="#444444",
        linewidth=0.8,
        alpha=0.92,
        zorder=18
    )
    ax.add_patch(box)

    half = bar_deg / 2.0
    ax.add_patch(mpatches.Rectangle((x0, y0), half, h, facecolor="#111111", edgecolor="#111111", linewidth=0.8, zorder=19))
    ax.add_patch(mpatches.Rectangle((x0 + half, y0), half, h, facecolor="white", edgecolor="#111111", linewidth=0.8, zorder=19))

    ax.text(x0, y0 + h * 1.45, "0", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#111111", zorder=20)
    ax.text(x0 + half, y0 + h * 1.45, f"{scale_km / 2:g}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#111111", zorder=20)
    ax.text(x0 + bar_deg, y0 + h * 1.45, f"{scale_km:g} km", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#111111", zorder=20)


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
    target_h = max(int(76 * factor), 24)
    scale = target_h / max(float(pil_logo.height), 1.0)
    target_w = min(max(int(pil_logo.width * scale), 28), 180)
    pil_resized = pil_logo.resize((target_w, target_h), Image.LANCZOS)

    oi = OffsetImage(np.asarray(pil_resized), zoom=0.45)
    ab = AnnotationBbox(
        oi,
        (0.992, 0.992),
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


def compute_adaptive_marker_size(num_points: int, user_scale_pct: int = 100) -> float:
    """
    Automatically scales point marker area (`s` in points^2) so tent icons remain bold and
    distinct on large-format paper while preventing crowding up to 1,000+ facilities.
    """
    if num_points <= 40:
        base_s = 135.0
    elif num_points <= 120:
        base_s = 105.0
    elif num_points <= 300:
        base_s = 78.0
    elif num_points <= 600:
        base_s = 52.0
    elif num_points <= 950:
        base_s = 36.0
    else:
        base_s = 26.0
    return max(base_s * (max(user_scale_pct, 30) / 100.0), 12.0)


# ==============================================================================
# 5. CUSTOM A-FRAME TENT & AIRSTRIP ICONS & STYLES
# ==============================================================================
def _build_safari_tent_marker() -> mpath.Path:
    """Creates a crisp A-frame safari tent triangle icon with a central doorway flap."""
    verts = [
        (0.0, 1.08),      # Tent peak
        (1.02, -0.86),    # Bottom right corner
        (0.28, -0.86),    # Right of tent doorway
        (0.0, -0.14),     # Doorway apex
        (-0.28, -0.86),   # Left of tent doorway
        (-1.02, -0.86),   # Bottom left corner
        (0.0, 1.08),      # Close tent outline
    ]
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.CLOSEPOLY,
    ]
    return mpath.Path(verts, codes)


def _build_airstrip_marker() -> mpath.Path:
    """Creates a crisp airplane silhouette marker for airstrips."""
    verts = [
        (0.0, 1.05),      # Nose
        (0.18, 0.35),
        (1.05, -0.10),    # Right wingtip
        (1.05, -0.32),
        (0.18, -0.15),
        (0.16, -0.72),
        (0.50, -0.95),    # Right tailplane
        (0.50, -1.08),
        (0.0, -0.92),
        (-0.50, -1.08),   # Left tailplane
        (-0.50, -0.95),
        (-0.16, -0.72),
        (-0.18, -0.15),
        (-1.05, -0.32),   # Left wingtip
        (-1.05, -0.10),
        (-0.18, 0.35),
        (0.0, 1.05),
    ]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)


TENT_CAMP_MARKER = _build_safari_tent_marker()
AIRSTRIP_MARKER = _build_airstrip_marker()
AIRSTRIP_COLOR = "#111827"             # High-contrast charcoal/black airplane icon with white outline
AIRSTRIP_LEGEND_LABEL = "Airstrips"

UNIFIED_FACILITY_COLOR = "#0057FF"     # Vivid cobalt blue — stands out sharply against red/orange fences & terrain
UNIFIED_FACILITY_MARKER = TENT_CAMP_MARKER
UNIFIED_FACILITY_LABEL = "Tented Camps & Lodges"

FACILITY_STYLE_PALETTE = {
    "Tented Camp": ("#0057FF", TENT_CAMP_MARKER),
    "Lodge": ("#7D3C98", "s"),
    "Eco-Lodge": ("#117864", TENT_CAMP_MARKER),
    "Eco Camp": ("#008B8B", TENT_CAMP_MARKER),
    "Ranger Post": ("#1E8449", "^"),
    "Gate": ("#1F4E78", "v"),
    "Cottage": ("#8E44AD", "v"),
    "Villa": ("#2E4053", "h"),
    "Airstrip": (AIRSTRIP_COLOR, AIRSTRIP_MARKER),
}
FALLBACK_FACILITY_STYLES = [
    ("#0057FF", TENT_CAMP_MARKER), ("#8E44AD", "s"), ("#16A085", "D"),
    ("#2980B9", "P"), ("#1F618D", "^"), ("#6C3483", "v"),
]

# Distinct, vivid non-opaque colors for Conservancy / Park boundary polygons
CONSERVANCY_DISTINCT_PALETTE = [
    "#1E8449",  # Emerald Forest Green (e.g., Maasai Mara / Mara Triangle)
    "#1B4F72",  # Deep Royal Navy Blue (e.g., Mara North / Olare Motorogi)
    "#8E44AD",  # Rich Plum Violet (e.g., Naboisho / Ol Kinyei)
    "#B9770E",  # Warm Ochre Gold (e.g., Lemek / Ol Chorro)
    "#0E6251",  # Deep Teal (e.g., Pardamat / Siana)
    "#922B21",  # Crimson Terracotta (e.g., Olarro / Isaaten)
    "#2874A6",  # Cerulean Blue
    "#6C3483",  # Indigo Purple
]

FENCE_BROAD_STYLES = {
    "Electric Fence": ("#E60039", "-", 2.1),
    "Wire Fence": ("#FF8C00", "--", 1.9),
    "Cactus / Live Fence": ("#2E8B57", "-.", 1.8),
    "Defenced Fence": ("#6C3483", ":", 1.8),
    "Stone Wall / Fence": ("#5D6D7E", "-", 1.8),
    "Woodpost Fence": ("#8E5B32", "--", 1.7),
    "Other Fence": ("#7F8C8D", ":", 1.5),
}

FENCE_DETAILED_STYLES = {
    "Electric Fence (9-Strand)": ("#8E0038", "-", 2.4),
    "Electric Fence (7-Strand)": ("#D90429", "-", 2.2),
    "Electric Fence (6-Strand)": ("#FF3366", "-", 2.0),
    "Electric Fence (Unspecified Strands)": ("#E60073", (0, (5, 2)), 1.9),
    "Wire Fence (5-Strand)": ("#B9770E", "--", 2.0),
    "Wire Fence (4-Strand)": ("#D68910", "--", 1.9),
    "Wire Fence (3-Strand)": ("#F39C12", ":", 1.9),
    "Wire Fence": ("#E67E22", "--", 1.9),
    "Cactus / Live Fence": ("#2E8B57", "-.", 1.8),
    "Defenced Fence": ("#6C3483", ":", 1.8),
    "Stone Wall / Fence": ("#5D6D7E", "-", 1.8),
    "Woodpost Fence": ("#8E5B32", "--", 1.7),
    "Other Fence": ("#7F8C8D", ":", 1.5),
}

PAPER_SIZES_INCHES = {
    "Large Poster / A2 Landscape (23.4 × 16.5 in) — Best Visibility": (23.4, 16.5),
    "Executive Wall Map / A3 Landscape (16.5 × 11.7 in)": (16.5, 11.7),
    "Standard Report / A4 Landscape (12.4 × 8.8 in)": (12.4, 8.8),
}


def render_camps_and_fences_map(
    map_title: str,
    camps_df: pd.DataFrame,
    airstrips_df: pd.DataFrame,
    fence_features: list[dict],
    fence_breakdown_mode: str,
    focus_on_facilities_extent: bool = True,
    group_all_facilities_as_one: bool = True,
    show_legend_counts: bool = False,
    paper_size_label: str = "Large Poster / A2 Landscape (23.4 × 16.5 in) — Best Visibility",
    basemap_provider: str = "OpenTopoMap (Bright Contour & Terrain — Recommended)",
    conservancies: list = None,
    rivers_features: list = None,
    rivers_label: str = "Rivers & Luggas",
    roads_features: list = None,
    roads_label: str = "Roads & Tracks",
    ranger_features: list = None,
    ranger_posts_label: str = "Ranger Posts",
    label_ranger_posts: bool = False,
    logo_bytes=None,
    report_period: str = "November 2026",
    zoom_level: int = 11,
    include_camp_names_in_legend: bool = False,
    label_camps_on_map: bool = False,
    label_airstrips_on_map: bool = True,
    label_conservancies_on_map: bool = True,
    logo_scale_pct: int = 60,
    area_fill_alpha: float = 0.18,
    marker_scale_pct: int = 125,
) -> bytes:
    """
    Renders a bright, large-format cartographic map of Tented Camps & Lodges (tent triangle icons),
    Airstrips (airplane icons), translucent distinct-color Conservancy Polygons, and Fencing Data
    (sourced from Mara Elephant Project), shifted flush-left so the lower-right legend sits in open space.
    """
    conservancies = conservancies or []
    rivers_features = rivers_features or []
    roads_features = roads_features or []
    ranger_features = ranger_features or []

    # Anchor reference box on Camps + Airstrips so far-away polygons never zoom the map out
    ref_lons, ref_lats = [], []
    if len(camps_df) > 0:
        ref_lons.extend([float(camps_df["Lon"].min()), float(camps_df["Lon"].max())])
        ref_lats.extend([float(camps_df["Lat"].min()), float(camps_df["Lat"].max())])
    if len(airstrips_df) > 0:
        ref_lons.extend([float(airstrips_df["Lon"].min()), float(airstrips_df["Lon"].max())])
        ref_lats.extend([float(airstrips_df["Lat"].min()), float(airstrips_df["Lat"].max())])

    camps_ref_box = (min(ref_lons), min(ref_lats), max(ref_lons), max(ref_lats)) if ref_lons else None

    bounds_list = []
    if camps_ref_box is not None:
        bounds_list.append(camps_ref_box)

    for _, c_feats, _ in conservancies:
        b = get_features_bounds(c_feats, clip_near_box=camps_ref_box)
        if b is not None:
            bounds_list.append(b)

    if fence_features:
        fb = get_features_bounds(fence_features, clip_near_box=camps_ref_box)
        if fb is not None:
            bounds_list.append(fb)

    if bounds_list:
        min_lon = min(b[0] for b in bounds_list)
        min_lat = min(b[1] for b in bounds_list)
        max_lon = max(b[2] for b in bounds_list)
        max_lat = max(b[3] for b in bounds_list)
    else:
        min_lon, min_lat, max_lon, max_lat = 34.82, -1.70, 35.42, -1.20

    lon_span = max(max_lon - min_lon, 0.05)
    lat_span = max(max_lat - min_lat, 0.05)

    # Tight left margin (3%) so the map shifts left to fill empty space,
    # and generous right/bottom margin (15% right, 11% bottom, 7% top) so the lower-right legend
    # sits in open space without covering southeastern camps or fence segments.
    min_lon = min_lon - lon_span * 0.03
    max_lon = max_lon + lon_span * 0.15
    min_lat = min_lat - lat_span * 0.11
    max_lat = max_lat + lat_span * 0.07

    fig_w, fig_h = PAPER_SIZES_INCHES.get(paper_size_label, (23.4, 16.5))
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
        extra_lon = desired_lon_span - cur_lon_span
        min_lon -= extra_lon * 0.10
        max_lon += extra_lon * 0.90
    else:
        desired_lat_span = ground_width / target_aspect
        lat_center = (min_lat + max_lat) / 2.0
        min_lat = lat_center - desired_lat_span / 2.0
        max_lat = lat_center + desired_lat_span / 2.0

    size_mult = fig_w / 14.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 1. Bright Basemap Layer (OpenTopoMap default + slight white blend for high clarity)
    basemap_img, extent = fetch_basemap_tiles(
        min_lon, min_lat, max_lon, max_lat, zoom=zoom_level, provider_label=basemap_provider
    )
    ax.imshow(basemap_img, extent=extent, aspect="auto", alpha=0.88, zorder=1)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect(1.0 / cos_lat, adjustable="box")

    legend_handles = []

    # 2. Conservancy / Park Boundaries (Distinct Color + Non-Opaque Translucent Polygon Fill)
    fill_alpha = min(max(area_fill_alpha, 0.08), 0.45) if area_fill_alpha > 0 else 0.0
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
                if feat["geom_type"] == "Polygon" and fill_alpha > 0:
                    poly = mpatches.Polygon(
                        ring,
                        closed=True,
                        facecolor=edge_color,
                        edgecolor="none",
                        alpha=fill_alpha,  # Non-opaque translucent tint so terrain, rivers & fences shine through
                        zorder=3
                    )
                    ax.add_patch(poly)
                ax.plot(xs, ys, color=edge_color, linewidth=2.3 * size_mult * 0.85, linestyle=lstyle, alpha=0.95, zorder=5 + idx)

            if label_conservancies_on_map:
                cen = get_feature_centroid(feat)
                if cen is not None and min_lon <= cen[0] <= max_lon and min_lat <= cen[1] <= max_lat:
                    ax.text(
                        cen[0],
                        cen[1],
                        str(feat["name"]).upper(),
                        fontsize=8.5 * size_mult * 0.85,
                        fontweight="bold",
                        color=edge_color,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=edge_color, alpha=0.85, lw=0.9),
                        zorder=9
                    )

        disp_legend_name = ", ".join(unique_names[:3]) if len(unique_names) > 1 else (unique_names[0] if unique_names else layer_label)
        legend_handles.append(
            mpatches.Patch(
                facecolor=edge_color if fill_alpha > 0 else "none",
                edgecolor=edge_color,
                alpha=min(fill_alpha + 0.14, 0.55) if fill_alpha > 0 else 1.0,
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
                ax.plot(xs, ys, color="#5C4033", linewidth=1.1 * size_mult * 0.8, linestyle="-", alpha=0.72, zorder=6)
        legend_handles.append(
            mlines.Line2D([], [], color="#5C4033", linewidth=1.8, linestyle="-", label=roads_label)
        )

    # 4. Rivers & Luggas Overlay
    if rivers_features:
        for feat in rivers_features:
            for ring in feat["parts"]:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.plot(xs, ys, color="#0066CC", linewidth=1.3 * size_mult * 0.8, linestyle="-", alpha=0.82, zorder=7)
        legend_handles.append(
            mlines.Line2D([], [], color="#0066CC", linewidth=1.9, label=rivers_label)
        )

    # 5. Fencing Layer
    if fence_features:
        use_detailed = fence_breakdown_mode.startswith("Detailed")
        key_field = "fence_detail" if use_detailed else "fence_broad"
        style_map = FENCE_DETAILED_STYLES if use_detailed else FENCE_BROAD_STYLES

        grouped_fences = {}
        for feat in fence_features:
            cat = feat.get(key_field, "Other Fence")
            grouped_fences.setdefault(cat, []).append(feat)

        ordered_fence_cats = sorted(
            grouped_fences.keys(),
            key=lambda x: (
                0 if "electric" in x.lower() else (1 if "wire" in x.lower() else (2 if "other" not in x.lower() else 3)),
                x
            )
        )

        for f_cat in ordered_fence_cats:
            f_list = grouped_fences[f_cat]
            col, lstyle, lw = style_map.get(
                f_cat,
                ("#D90429" if "electric" in f_cat.lower() else "#E67E22", "-" if "electric" in f_cat.lower() else "--", 1.8)
            )
            scaled_lw = lw * max(size_mult * 0.82, 1.0)
            total_km = sum(f.get("length_km", 0.0) for f in f_list)

            for feat in f_list:
                for ring in feat["parts"]:
                    xs = [p[0] for p in ring]
                    ys = [p[1] for p in ring]
                    ax.plot(xs, ys, color=col, linewidth=scaled_lw, linestyle=lstyle, alpha=0.88, zorder=10)

            leg_lbl = f"{f_cat} ({total_km:,.1f} km)" if (show_legend_counts and total_km > 0.05) else f_cat
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color=col,
                    linewidth=2.6,
                    linestyle=lstyle,
                    label=leg_lbl
                )
            )

    # 6. Ranger Posts Overlay
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
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7.2,
                        fontweight="bold",
                        color="#145A32",
                        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#1E8449", alpha=0.82, lw=0.6),
                        zorder=13
                    )
        if pts_x:
            ax.scatter(pts_x, pts_y, s=65 * size_mult, marker="^", c="#1E8449", edgecolors="white", linewidths=1.0, zorder=12)
        legend_handles.append(
            mlines.Line2D([], [], color="white", marker="^", markerfacecolor="#1E8449", markeredgecolor="white", markersize=8.5, label=ranger_posts_label)
        )

    # 7. Plot Tented Camps & Lodges (Triangle Camp Tent Icon)
    if len(camps_df) > 0:
        work_df = camps_df.copy().reset_index(drop=True)
        n_total_pts = len(work_df)
        adaptive_s = compute_adaptive_marker_size(n_total_pts, user_scale_pct=marker_scale_pct) * (size_mult ** 1.25)
        edge_lw = 0.75 if n_total_pts >= 400 else (1.05 if n_total_pts >= 150 else 1.35)
        jitter_rad = 0.00035 if n_total_pts >= 250 else 0.00080

        coord_counts = {}
        plot_lons, plot_lats = [], []
        for _, r in work_df.iterrows():
            key = (round(float(r["Lon"]), 5), round(float(r["Lat"]), 5))
            idx_at_pt = coord_counts.get(key, 0)
            coord_counts[key] = idx_at_pt + 1
            if idx_at_pt == 0:
                plot_lons.append(float(r["Lon"]))
                plot_lats.append(float(r["Lat"]))
            else:
                angle = (idx_at_pt * 2.0 * math.pi) / 6.0
                radius = jitter_rad * (1 + (idx_at_pt - 1) // 6)
                plot_lons.append(float(r["Lon"]) + radius * math.cos(angle))
                plot_lats.append(float(r["Lat"]) + radius * math.sin(angle))

        work_df["_plot_lon"] = plot_lons
        work_df["_plot_lat"] = plot_lats

        if group_all_facilities_as_one:
            ax.scatter(
                work_df["_plot_lon"],
                work_df["_plot_lat"],
                s=adaptive_s,
                marker=UNIFIED_FACILITY_MARKER,
                c=UNIFIED_FACILITY_COLOR,
                edgecolors="white",
                linewidths=edge_lw,
                alpha=0.96,
                zorder=16
            )
            fac_leg_lbl = f"{UNIFIED_FACILITY_LABEL} ({len(work_df):,})" if show_legend_counts else UNIFIED_FACILITY_LABEL
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color="white",
                    marker=UNIFIED_FACILITY_MARKER,
                    markerfacecolor=UNIFIED_FACILITY_COLOR,
                    markeredgecolor="#002B80",
                    markeredgewidth=0.9,
                    markersize=11.5,
                    label=fac_leg_lbl
                )
            )
            if include_camp_names_in_legend:
                for _, sr in work_df.iterrows():
                    legend_handles.append(
                        mlines.Line2D(
                            [], [],
                            color="white",
                            marker=UNIFIED_FACILITY_MARKER,
                            markerfacecolor=UNIFIED_FACILITY_COLOR,
                            markeredgecolor="#002B80",
                            markersize=7.5,
                            label=f"  • {sr['Name']}"
                        )
                    )
            if label_camps_on_map:
                for _, sr in work_df.iterrows():
                    if min_lon <= sr["_plot_lon"] <= max_lon and min_lat <= sr["_plot_lat"] <= max_lat:
                        ax.annotate(
                            str(sr["Name"]),
                            xy=(sr["_plot_lon"], sr["_plot_lat"]),
                            xytext=(3, 2),
                            textcoords="offset points",
                            fontsize=6.2,
                            fontweight="bold",
                            color="#111111",
                            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor=UNIFIED_FACILITY_COLOR, alpha=0.80, lw=0.5),
                            zorder=17
                        )
        else:
            fac_types = work_df["Facility_Type"].value_counts().index.tolist()
            for t_idx, f_type in enumerate(fac_types):
                sub = work_df[work_df["Facility_Type"] == f_type]
                col, marker_sym = FACILITY_STYLE_PALETTE.get(
                    f_type, FALLBACK_FACILITY_STYLES[t_idx % len(FALLBACK_FACILITY_STYLES)]
                )
                ax.scatter(
                    sub["_plot_lon"],
                    sub["_plot_lat"],
                    s=adaptive_s,
                    marker=marker_sym,
                    c=col,
                    edgecolors="white",
                    linewidths=edge_lw,
                    alpha=0.96,
                    zorder=16
                )
                fac_leg_lbl = f"{f_type} ({len(sub):,})" if show_legend_counts else f_type
                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        color="white",
                        marker=marker_sym,
                        markerfacecolor=col,
                        markeredgecolor="white",
                        markeredgewidth=0.9,
                        markersize=10.5,
                        label=fac_leg_lbl
                    )
                )
                if include_camp_names_in_legend:
                    for _, sr in sub.iterrows():
                        legend_handles.append(
                            mlines.Line2D(
                                [], [],
                                color="white",
                                marker=marker_sym,
                                markerfacecolor=col,
                                markeredgecolor="white",
                                markersize=7.0,
                                label=f"  • {sr['Name']}"
                            )
                        )
                if label_camps_on_map:
                    for _, sr in sub.iterrows():
                        if min_lon <= sr["_plot_lon"] <= max_lon and min_lat <= sr["_plot_lat"] <= max_lat:
                            ax.annotate(
                                str(sr["Name"]),
                                xy=(sr["_plot_lon"], sr["_plot_lat"]),
                                xytext=(3, 2),
                                textcoords="offset points",
                                fontsize=6.2,
                                fontweight="bold",
                                color="#111111",
                                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor=col, alpha=0.80, lw=0.5),
                                zorder=17
                            )

    # 8. Plot Airstrips (Airplane Silhouette Icon)
    if len(airstrips_df) > 0:
        air_s = max(145.0 * (max(marker_scale_pct, 40) / 100.0) * (size_mult ** 1.25), 40.0)
        ax.scatter(
            airstrips_df["Lon"],
            airstrips_df["Lat"],
            s=air_s,
            marker=AIRSTRIP_MARKER,
            c=AIRSTRIP_COLOR,
            edgecolors="white",
            linewidths=1.2,
            alpha=0.98,
            zorder=18
        )
        air_leg_lbl = f"{AIRSTRIP_LEGEND_LABEL} ({len(airstrips_df):,})" if show_legend_counts else AIRSTRIP_LEGEND_LABEL
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color="white",
                marker=AIRSTRIP_MARKER,
                markerfacecolor=AIRSTRIP_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=12.0,
                label=air_leg_lbl
            )
        )
        if label_airstrips_on_map:
            for _, ar in airstrips_df.iterrows():
                alon, alat = float(ar["Lon"]), float(ar["Lat"])
                if min_lon <= alon <= max_lon and min_lat <= alat <= max_lat:
                    disp_air = f"{ar['Name']} ({ar['Code']})" if str(ar.get("Code", "")).strip() else str(ar["Name"])
                    ax.annotate(
                        disp_air,
                        xy=(alon, alat),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7.2 * max(size_mult * 0.78, 0.90),
                        fontweight="bold",
                        color="#111827",
                        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#111827", alpha=0.88, lw=0.7),
                        zorder=19
                    )

    # 9. North Arrow & Scale Bar
    draw_compass_and_bearing(ax, x_frac=0.055, y_frac=0.895)
    draw_scale_bar(ax, min_lon, max_lon, min_lat, max_lat)

    # 10. Clean, Unclipped Legend (Lower-Right Open Space or Outside if >18 items)
    n_leg = len(legend_handles)
    leg_fs = 8.5 * max(size_mult * 0.78, 0.95)
    title_fs = 9.2 * max(size_mult * 0.78, 0.95)

    if n_leg > 18:
        leg = ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1 if n_leg <= 36 else 2,
            title="MAP LEGEND",
            frameon=True,
            facecolor="white",
            edgecolor="#1F4E78",
            framealpha=0.96,
            fontsize=leg_fs * 0.9,
            title_fontsize=title_fs,
        )
    else:
        leg = ax.legend(
            handles=legend_handles,
            loc="lower right",
            ncol=1 if n_leg <= 12 else 2,
            title="MAP LEGEND",
            frameon=True,
            facecolor="white",
            edgecolor="#1F4E78",
            framealpha=0.95,
            fontsize=leg_fs,
            title_fontsize=title_fs,
            borderpad=0.65,
            labelspacing=0.45,
            handletextpad=0.55,
        )
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(22)

    # 11. Title & Mara Elephant Project Source Recognition
    total_camps = len(camps_df)
    total_airstrips = len(airstrips_df)
    total_fence_segs = len(fence_features)
    total_fence_km = sum(f.get("length_km", 0.0) for f in fence_features)

    title_line_1 = f"{map_title.upper()} — {report_period.upper()}"
    subtitle_parts = [f"Tented Camps & Lodges: {total_camps:,}"]
    if total_airstrips > 0:
        subtitle_parts.append(f"Airstrips: {total_airstrips:,}")
    if fence_features:
        subtitle_parts.append(f"Fence Segments: {total_fence_segs:,} ({total_fence_km:,.1f} km)")
        subtitle_parts.append("Fencing Data Source: Mara Elephant Project (MEP)")
    title_line_2 = "   |   ".join(subtitle_parts)

    ax.set_title(
        title_line_1 + "\n" + title_line_2,
        fontsize=12.5 * max(size_mult * 0.82, 0.95),
        fontweight="bold",
        color="#1F4E78",
        pad=14
    )
    ax.set_xlabel("Longitude (°E)", fontsize=10.5 * max(size_mult * 0.80, 0.95), fontweight="bold")
    ax.set_ylabel("Latitude (°S)", fontsize=10.5 * max(size_mult * 0.80, 0.95), fontweight="bold")
    ax.tick_params(axis="both", labelsize=9.2 * max(size_mult * 0.80, 0.95))
    ax.grid(True, linestyle=":", linewidth=0.55, color="#555555", alpha=0.45, zorder=2)

    if fence_features:
        ax.text(
            0.010, 0.012,
            "Fencing Data Source: Mara Elephant Project (MEP)",
            transform=ax.transAxes,
            fontsize=8.2 * max(size_mult * 0.78, 0.95),
            fontweight="bold",
            color="#1F4E78",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.90, edgecolor="#1F4E78", linewidth=0.7, pad=2.2),
            zorder=20
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
st.title("🏕️ Tented Camps, Lodges, Airstrips & Fencing Map")
st.markdown(
    "Large-format cartographic map of **Tented Camps & Lodges**, **Airstrips**, and **Conservancy Boundaries** "
    "overlaid with **Fencing Shapefiles** (sourced from the **Mara Elephant Project**)."
)

col1, col2 = st.columns(2)
with col1:
    camps_file = st.file_uploader(
        "1️⃣ Upload Camps & Facilities Locations (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="camps_facilities_uploader"
    )
    airstrips_file = st.file_uploader(
        "2️⃣ Upload Airstrips Locations (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="camps_airstrips_uploader"
    )
    fences_shp_zips = st.file_uploader(
        "3️⃣ Upload Fences Shapefile (`.zip` — Source: Mara Elephant Project)",
        type=["zip"],
        accept_multiple_files=True,
        key="mep_fences_shp_zip_uploader"
    )
    fences_attr_csv = st.file_uploader(
        "4️⃣ Optional: Upload Fences Attribute Table CSV (`fenced Mt.csv`)",
        type=["csv", "xlsx", "xls"],
        key="mep_fences_csv_uploader"
    )
    logo_file = st.file_uploader(
        "5️⃣ Upload Mara Conservancy / Company Logo (PNG / JPG — Top-Right)",
        type=["png", "jpg", "jpeg"],
        key="camps_company_logo_uploader"
    )

with col2:
    conservancy_zips = st.file_uploader(
        "6️⃣ Upload Conservancy / Park Boundary Shapefiles (`.zip` — one or multiple)",
        type=["zip"],
        accept_multiple_files=True,
        key="camps_conservancy_zip_uploader"
    )
    roads_zip = st.file_uploader(
        "7️⃣ Upload Roads & Tracks Shapefile (`.zip`)",
        type=["zip"],
        key="camps_roads_zip_uploader"
    )
    rivers_zips = st.file_uploader(
        "8️⃣ Upload Rivers & Luggas Shapefiles (`.zip`)",
        type=["zip"],
        accept_multiple_files=True,
        key="camps_rivers_zip_uploader"
    )
    ranger_posts_zip = st.file_uploader(
        "9️⃣ Upload Ranger Posts / Gates Shapefile (`.zip`)",
        type=["zip"],
        key="camps_ranger_posts_zip_uploader"
    )

with st.sidebar:
    st.subheader("Paper Size & Bright Basemap Layer")
    paper_size_label = st.selectbox(
        "Map Sheet / Paper Size",
        options=list(PAPER_SIZES_INCHES.keys()),
        index=0,
        key="camps_paper_size_sel"
    )
    basemap_provider = st.selectbox(
        "Background Map Layer",
        options=list(BASEMAP_PROVIDERS.keys()),
        index=0,
        key="camps_basemap_provider_sel"
    )
    map_title = st.text_input("Map Title", value="TENTED CAMPS, LODGES & FENCING DISTRIBUTION", key="camps_map_title_txt")
    report_period = st.text_input("Report Period Label", value="November 2026", key="camps_map_period_txt")
    zoom_level = st.slider("Map Tile Detail (Zoom)", min_value=9, max_value=13, value=11, key="camps_topo_zoom")
    logo_scale_pct = st.slider("Top-Right Logo Size (%)", min_value=25, max_value=100, value=60, step=5, key="camps_logo_size_pct")
    marker_scale_pct = st.slider(
        "Tented Camps & Airstrips Icon Size (%)",
        min_value=50,
        max_value=200,
        value=125,
        step=10,
        key="camps_marker_scale_pct"
    )

    st.divider()
    st.subheader("Legend & Grouping Options")
    group_all_facilities_as_one = st.checkbox(
        "Group All Facilities as 'Tented Camps & Lodges' (Single Standout Color)",
        value=True,
        key="camps_group_unified_chk"
    )
    show_legend_counts = st.checkbox(
        "Show Numbers / Kilometers in Legend",
        value=False,
        key="camps_show_leg_counts_chk"
    )
    focus_on_facilities_extent = st.checkbox(
        "Frame Map Around Camps & Boundaries (Recommended)",
        value=True,
        key="camps_focus_extent_chk"
    )
    label_airstrips_on_map = st.checkbox(
        "Show Airstrip Name Labels on Map",
        value=True,
        key="camps_label_airstrips_chk"
    )
    label_camps_on_map = st.checkbox(
        "Show Camp / Facility Name Labels on Map",
        value=False,
        key="camps_labels_on_map_chk"
    )
    include_camp_names_in_legend = st.checkbox(
        "Include Individual Camp / Facility Names in Legend",
        value=False,
        key="camps_names_in_legend_chk"
    )
    dedup_camps = st.checkbox(
        "Remove Exact Duplicate Coordinates",
        value=True,
        key="camps_dedup_chk"
    )

    st.divider()
    st.subheader("Fencing Breakdown (Source: MEP)")
    fence_breakdown_mode = st.radio(
        "Classify Fences By:",
        options=[
            "Broad Type (Electric Fence vs. Wire Fence)",
            "Detailed Strand Specification (e.g., Electric Fence 7-Strand, Wire Fence 5-Strand)"
        ],
        key="mep_fence_breakdown_radio"
    )
    only_active_fences = st.checkbox(
        "Show Only Active Fences (active = 1) when status attribute exists",
        value=False,
        key="mep_only_active_fences_chk"
    )

    st.divider()
    st.subheader("Conservancies & Overlays")
    area_fill_opacity = st.slider(
        "Conservancy Polygon Fill Opacity (% — Non-Opaque)",
        min_value=5,
        max_value=45,
        value=18,
        step=2,
        key="camps_area_alpha"
    )
    show_poly_labels = st.checkbox("Label Conservancy / Park Areas", value=True, key="camps_poly_lbl")
    roads_legend_name = st.text_input("Roads Legend Label", value="Roads & Tracks", key="camps_roads_leg")
    rivers_legend_name = st.text_input("Rivers & Luggas Legend Label", value="Rivers & Luggas", key="camps_rivers_leg")
    ranger_legend_name = st.text_input("Ranger Posts Legend Label", value="Ranger Posts", key="camps_ranger_leg")
    show_post_labels = st.checkbox("Label Ranger Posts", value=False, key="camps_post_lbl")


# ------------------------------------------------------------------------------
# Parse Camps & Facilities Data
# ------------------------------------------------------------------------------
camps_df = pd.DataFrame()
if camps_file is not None:
    if camps_file.name.lower().endswith((".xlsx", ".xls")):
        raw_camps = pd.read_excel(camps_file)
    else:
        raw_camps = pd.read_csv(camps_file)
    try:
        camps_df = parse_camps_facilities_dataframe(raw_camps, deduplicate=dedup_camps)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    avail_fac_types = sorted(camps_df["Facility_Type"].unique().tolist())
    selected_fac_types = st.multiselect(
        "Filter Source Facility Categories to Include:",
        options=avail_fac_types,
        default=avail_fac_types,
        key="camps_fac_type_filter"
    )
    if selected_fac_types:
        camps_df = camps_df[camps_df["Facility_Type"].isin(selected_fac_types)].reset_index(drop=True)

# ------------------------------------------------------------------------------
# Parse Airstrips Data
# ------------------------------------------------------------------------------
airstrips_df = pd.DataFrame()
if airstrips_file is not None:
    if airstrips_file.name.lower().endswith((".xlsx", ".xls")):
        raw_air = pd.read_excel(airstrips_file)
    else:
        raw_air = pd.read_csv(airstrips_file)
    try:
        airstrips_df = parse_airstrips_dataframe(raw_air, deduplicate=dedup_camps)
    except ValueError as err:
        st.error(str(err))
        st.stop()

# ------------------------------------------------------------------------------
# Parse Mara Elephant Project (MEP) Fencing Shapefiles (.zip) & Optional CSV
# ------------------------------------------------------------------------------
fence_features = []
fence_shp_summary_df = pd.DataFrame()
if fences_shp_zips:
    raw_fence_feats = []
    for fzip in fences_shp_zips:
        try:
            raw_fence_feats.extend(load_shapefile_zip(fzip, fallback_name="Fence"))
        except Exception as e:
            st.warning(f"Could not read Fence shapefile ({fzip.name}): {e}")
    if raw_fence_feats:
        fence_features, fence_shp_summary_df = process_mep_fence_shapefile_features(raw_fence_feats)
        if only_active_fences and len(fence_features) > 0:
            filtered_f = [f for f in fence_features if f.get("fence_active") == "Active (1)"]
            if filtered_f:
                fence_features = filtered_f
                fence_shp_summary_df = fence_shp_summary_df[fence_shp_summary_df["Status"] == "Active (1)"].reset_index(drop=True)
        fb_check = get_features_bounds(fence_features)
        if fb_check:
            st.success(
                f"✅ Loaded **{len(fence_features):,}** fence segments from shapefile "
                f"(Lon: `{fb_check[0]:.3f}°` to `{fb_check[2]:.3f}°`, Lat: `{fb_check[1]:.3f}°` to `{fb_check[3]:.3f}°`)."
            )

fence_csv_summary_df = pd.DataFrame()
if fences_attr_csv is not None:
    if fences_attr_csv.name.lower().endswith((".xlsx", ".xls")):
        raw_f_csv = pd.read_excel(fences_attr_csv)
    else:
        raw_f_csv = pd.read_csv(fences_attr_csv)
    fence_csv_summary_df = parse_mep_fence_attribute_csv(raw_f_csv)
    if only_active_fences and len(fence_csv_summary_df) > 0:
        act_sub = fence_csv_summary_df[fence_csv_summary_df["Status"] == "Active (1)"]
        if len(act_sub) > 0:
            fence_csv_summary_df = act_sub.reset_index(drop=True)
    if not fence_features:
        st.warning(
            "⚠️ You uploaded the Fences attribute CSV (`fenced Mt.csv`), which contains fence types/lengths "
            "without line coordinates. To draw the fence lines on the map, also upload the Fences shapefile **`.zip`** in box **3️⃣**."
        )

# ------------------------------------------------------------------------------
# Display Summary Tables & KPIs
# ------------------------------------------------------------------------------
if len(camps_df) > 0 or len(airstrips_df) > 0 or len(fence_features) > 0 or len(fence_csv_summary_df) > 0:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tented Camps & Lodges", f"{len(camps_df):,}")
    k2.metric("Airstrips", f"{len(airstrips_df):,}")

    active_fence_df = fence_shp_summary_df if len(fence_shp_summary_df) > 0 else fence_csv_summary_df
    len_col_name = "Length (km)" if "Length (km)" in active_fence_df.columns else "Est. Length (km)"
    if len(active_fence_df) > 0:
        elec_segs = int((active_fence_df["Broad Category"] == "Electric Fence").sum())
        wire_segs = int((active_fence_df["Broad Category"] == "Wire Fence").sum())
        tot_km = float(active_fence_df[len_col_name].sum())
        k3.metric("Electric Fence Segments", f"{elec_segs:,}")
        k4.metric("Wire Fence Segments", f"{wire_segs:,}", delta=f"{tot_km:,.1f} km total")
    else:
        k3.metric("Electric Fences", "0")
        k4.metric("Wire Fences", "0")

    if len(camps_df) > 0:
        with st.expander(f"📋 View Tented Camps & Lodges Directory ({len(camps_df):,} locations)", expanded=False):
            st.dataframe(
                camps_df[["Name", "Facility_Type", "Lat", "Lon", "Reported_By"]],
                use_container_width=True,
                hide_index=True
            )

    if len(airstrips_df) > 0:
        with st.expander(f"✈️ View Airstrips Directory ({len(airstrips_df):,} airstrips)", expanded=False):
            st.dataframe(
                airstrips_df[["Name", "Code", "Surface", "Lat", "Lon"]],
                use_container_width=True,
                hide_index=True
            )

    if len(active_fence_df) > 0:
        st.subheader("🐘 Fencing Breakdown (Data Source: Mara Elephant Project)")
        f_agg = (
            active_fence_df
            .groupby(["Broad Category", "Detailed Fence Type", "Status"], as_index=False)
            .agg(Segments=("Raw MEP Code", "count"), Total_Length_km=(len_col_name, "sum"))
            .sort_values(by=["Broad Category", "Segments"], ascending=[True, False])
        )
        f_agg["Total_Length_km"] = f_agg["Total_Length_km"].round(2)
        st.dataframe(f_agg, use_container_width=True, hide_index=True)

    # Load optional boundary/infrastructure layers with distinct non-opaque colors
    conservancies_loaded = []
    if conservancy_zips:
        st.markdown("**Conservancy / Park Boundary Polygon Colors (Distinct & Translucent):**")
        name_cols = st.columns(min(len(conservancy_zips), 3))
        for idx, zfile in enumerate(conservancy_zips):
            default_label = os.path.splitext(zfile.name)[0].replace("_", " ").title()
            default_col = CONSERVANCY_DISTINCT_PALETTE[idx % len(CONSERVANCY_DISTINCT_PALETTE)]
            with name_cols[idx % len(name_cols)]:
                custom_name = st.text_input(
                    f"Boundary {idx + 1} Name ({zfile.name})",
                    value=default_label,
                    key=f"camps_cons_name_{idx}_{zfile.name}"
                )
                chosen_col = st.color_picker(
                    f"Boundary {idx + 1} Distinct Color",
                    value=default_col,
                    key=f"camps_cons_col_{idx}_{zfile.name}"
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
                st.warning(f"Could not read Rivers shapefile ({rzip.name}): {e}")

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
            st.warning(f"Could not read logo ({logo_file.name}): {e}")

    st.divider()
    safe_period = report_period.replace(" ", "_")

    if st.button("🗺️ Generate Large-Format Camps, Airstrips & Fencing Map", type="primary", key="btn_gen_camps_fences_map"):
        with st.spinner("Rendering bright high-resolution large-format map..."):
            try:
                map_png = render_camps_and_fences_map(
                    map_title=map_title,
                    camps_df=camps_df,
                    airstrips_df=airstrips_df,
                    fence_features=fence_features,
                    fence_breakdown_mode=fence_breakdown_mode,
                    focus_on_facilities_extent=focus_on_facilities_extent,
                    group_all_facilities_as_one=group_all_facilities_as_one,
                    show_legend_counts=show_legend_counts,
                    paper_size_label=paper_size_label,
                    basemap_provider=basemap_provider,
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
                    include_camp_names_in_legend=include_camp_names_in_legend,
                    label_camps_on_map=label_camps_on_map,
                    label_airstrips_on_map=label_airstrips_on_map,
                    label_conservancies_on_map=show_poly_labels,
                    logo_scale_pct=logo_scale_pct,
                    area_fill_alpha=area_fill_opacity / 100.0,
                    marker_scale_pct=marker_scale_pct,
                )
                st.session_state["camps_fences_map_png"] = map_png
                st.session_state["camps_fences_period"] = safe_period
            except Exception as exc:
                st.error(f"Map generation error: {exc}")

    if "camps_fences_map_png" in st.session_state:
        saved_png = st.session_state["camps_fences_map_png"]
        saved_per = st.session_state.get("camps_fences_period", safe_period)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"Tented_Camps_Lodges_Fences_Map_{saved_per}.png", saved_png)
            if len(camps_df) > 0:
                zf.writestr(f"Tented_Camps_Lodges_Directory_{saved_per}.csv", camps_df.to_csv(index=False))
            if len(airstrips_df) > 0:
                zf.writestr(f"Airstrips_Directory_{saved_per}.csv", airstrips_df.to_csv(index=False))
            if len(active_fence_df) > 0:
                zf.writestr(f"Fencing_Breakdown_MEP_{saved_per}.csv", active_fence_df.to_csv(index=False))
        zip_buf.seek(0)

        st.download_button(
            label="📦 Download High-Res Map + Directories + Fence Summary (.ZIP)",
            data=zip_buf.getvalue(),
            file_name=f"Mara_Tented_Camps_Lodges_Airstrips_Fences_{saved_per}.zip",
            mime="application/zip",
            key="dl_camps_fences_zip"
        )

        st.image(saved_png, caption=f"Tented Camps, Lodges, Airstrips & Fencing Map ({report_period})", use_container_width=True)
        st.download_button(
            label="📥 Download High-Res Map Image (.png)",
            data=saved_png,
            file_name=f"Tented_Camps_Lodges_Airstrips_Fences_Map_{saved_per}.png",
            mime="image/png",
            key="dl_camps_fences_png"
        )
else:
    st.info(
        "👆 Upload your **Camps & Facilities CSV**, **Airstrips CSV**, and/or your "
        "**Fences Shapefile (`.zip`)** above to generate the map."
    )
