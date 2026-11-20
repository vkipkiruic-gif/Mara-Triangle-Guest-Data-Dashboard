import io
import math
import os
import tempfile
import urllib.request
import zipfile

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

# ==============================================================================
# 1. OPENTOPOMAP TILE STITCHER (https://tile.opentopomap.org/{z}/{x}/{y}.png)
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
# 2. SHAPEFILE ZIP LOADER (CONSERVANCIES, ROADS, RIVERS & LUGGAS, RANGER POSTS)
# ==============================================================================
def load_shapefile_zip(uploaded_zip, fallback_name: str):
    """
    Extracts a zipped shapefile (.shp, .shx, .dbf, .prj) and reprojects to EPSG:4326 (WGS84).
    Detects feature names from attribute columns (Name, Station, Post, Conservancy, River, Road, etc.).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "archive.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        shp_files = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp") and not fname.startswith("._"):
                    shp_files.append(os.path.join(root, fname))

        if not shp_files:
            raise ValueError(f"No .shp file found inside {uploaded_zip.name}.")

        gdf = gpd.read_file(shp_files[0])
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        candidate_cols = [
            c for c in gdf.columns
            if any(k in c.lower() for k in [
                "name", "conserv", "area", "zone", "sector", "patrol",
                "river", "lugga", "road", "track", "post", "station", "gate", "camp"
            ])
        ]
        if candidate_cols:
            name_col = candidate_cols[0]
            gdf["Display_Name"] = gdf[name_col].fillna(fallback_name).astype(str).str.strip()
        else:
            gdf["Display_Name"] = fallback_name

        gdf.loc[gdf["Display_Name"].isin(["", "None", "nan", "NaN"]), "Display_Name"] = fallback_name
        return gdf


# ==============================================================================
# 3. CARTOGRAPHIC HELPERS: NORTH ARROW + BEARING ROSE, SCALE BAR, LOGO
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


def add_top_right_logo(ax, logo_bytes: bytes | None):
    """Places the uploaded company logo in the top-right corner of the map."""
    if not logo_bytes:
        return
    try:
        pil_logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
        max_h = 95
        ratio = max_h / float(pil_logo.height)
        new_w = max(int(pil_logo.width * ratio), 1)
        pil_logo = pil_logo.resize((new_w, max_h), Image.Resampling.LANCZOS)

        imagebox = OffsetImage(np.asarray(pil_logo), zoom=0.85)
        ab = AnnotationBbox(
            imagebox,
            (0.985, 0.985),
            xycoords="axes fraction",
            box_alignment=(1.0, 1.0),
            frameon=True,
            bboxprops=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.88, pad=0.25),
            zorder=25
        )
        ax.add_artist(ab)
    except Exception:
        pass


# ==============================================================================
# 4. RENDER INDIVIDUAL SPECIES MAP WITH ALL GIS LAYERS & LEGEND
# ==============================================================================
def render_species_map(
    species_name: str,
    sp_df: pd.DataFrame,
    conservancies: list[tuple[str, gpd.GeoDataFrame, str]],
    rivers_gdf: gpd.GeoDataFrame | None,
    rivers_label: str,
    roads_gdf: gpd.GeoDataFrame | None,
    roads_label: str,
    ranger_posts_gdf: gpd.GeoDataFrame | None,
    ranger_posts_label: str,
    label_ranger_posts: bool,
    logo_bytes: bytes | None,
    report_period: str,
    marker_color: str = "#D92323",
    zoom_level: int = 11,
    label_conservancies_on_map: bool = True
) -> bytes:
    """
    Renders a publication-grade map for a single wildlife species with:
      - OpenTopoMap basemap (https://tile.opentopomap.org/{z}/{x}/{y}.png)
      - Multiple conservancy / patrol area boundaries (distinct colors + names on map & legend)
      - Rivers & Luggas overlay (in legend)
      - Roads & Tracks overlay (in legend)
      - Ranger Posts / Stations overlay (in legend + optional labels)
      - Species sighting points + total sightings count in legend & title
      - Company logo in top-right
      - North arrow & 360° bearing rose + Geodesic scale bar
    """
    bounds_list = []
    if len(sp_df) > 0:
        bounds_list.append((sp_df["Lon"].min(), sp_df["Lat"].min(), sp_df["Lon"].max(), sp_df["Lat"].max()))
    for _, c_gdf, _ in conservancies:
        if not c_gdf.empty:
            b = c_gdf.total_bounds
            bounds_list.append((b[0], b[1], b[2], b[3]))

    if bounds_list:
        min_lon = min(b[0] for b in bounds_list)
        min_lat = min(b[1] for b in bounds_list)
        max_lon = max(b[2] for b in bounds_list)
        max_lat = max(b[3] for b in bounds_list)
    else:
        min_lon, min_lat, max_lon, max_lat = 34.75, -1.65, 35.30, -1.24

    lon_span = max(max_lon - min_lon, 0.05)
    lat_span = max(max_lat - min_lat, 0.05)
    min_lon -= lon_span * 0.10
    max_lon += lon_span * 0.10
    min_lat -= lat_span * 0.10
    max_lat += lat_span * 0.10

    fig, ax = plt.
