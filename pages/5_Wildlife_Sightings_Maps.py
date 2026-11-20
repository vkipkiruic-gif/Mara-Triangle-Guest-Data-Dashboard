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
import openpyxl
import pandas as pd
import streamlit as st
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from shapely.geometry import Point

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
# 2. SHAPEFILE ZIP LOADER (CONSERVANCIES & RIVERS / LUGGAS)
# ==============================================================================
def load_shapefile_zip(uploaded_zip, fallback_name: str):
    """
    Extracts a zipped shapefile (.shp, .shx, .dbf, .prj) and reprojects to EPSG:4326 (WGS84).
    Detects feature names from attribute columns (Name, Conservancy, Area_Name, etc.).
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

        # Detect best name column
        candidate_cols = [
            c for c in gdf.columns
            if any(k in c.lower() for k in ["name", "conserv", "area", "zone", "sector", "patrol", "river", "lugga"])
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

    # Background circle
    bg = mpatches.Circle((0, 0), 1.32, facecolor="white", edgecolor="#1F4E78", linewidth=1.3, alpha=0.92, zorder=1)
    ax_inset.add_patch(bg)

    # Bearing ticks every 30 degrees
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

    # North pointer (filled dark blue) and South pointer (white)
    north_poly = mpatches.Polygon([[0, 0.85], [-0.20, 0], [0, 0.14]], closed=True, facecolor="#1F4E78", edgecolor="#1F4E78", zorder=3)
    north_poly_r = mpatches.Polygon([[0, 0.85], [0.20, 0], [0, 0.14]], closed=True, facecolor="#4A90E2", edgecolor="#1F4E78", zorder=3)
    south_poly = mpatches.Polygon([[0, -0.85], [-0.20, 0], [0.20, 0]], closed=True, facecolor="white", edgecolor="#1F4E78", linewidth=0.9, zorder=3)
    ax_inset.add_patch(north_poly)
    ax_inset.add_patch(north_poly_r)
    ax_inset.add_patch(south_poly)

    # Cardinal & bearing labels
    ax_inset.text(0, 1.16, "N (000°)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1F4E78", zorder=4)
    ax_inset.text(1.14, 0, "090°", ha="left", va="center", fontsize=6.2, fontweight="bold", color="#333333", zorder=4)
    ax_inset.text(0, -1.16, "180°", ha="center", va="top", fontsize=6.2, fontweight="bold", color="#333333", zorder=4)
    ax_inset.text(-1.14, 0, "270°", ha="right", va="center", fontsize=6.2, fontweight="bold", color="#333333", zorder=4)


def draw_scale_bar(ax, min_lon, max_lon, min_lat, max_lat):
    """Draws an accurate geodesic km scale bar in the bottom-left corner."""
    mid_lat = (min_lat + max_lat) / 2.0
    km_per_deg_lon = 111.32 * math.cos(math.radians(mid_lat))
    map_width_km = (max_lon - min_lon) * km_per_deg_lon

    # Choose clean round scale length (~20% of map width)
    target_km = map_width_km * 0.22
    nice_steps = [1, 2, 5, 10, 15, 20, 25, 30, 50, 100]
    scale_km = min(nice_steps, key=lambda s: abs(s - target_km))
    bar_deg = scale_km / max(km_per_deg_lon, 1e-6)

    x0 = min_lon + (max_lon - min_lon) * 0.04
    y0 = min_lat + (max_lat - min_lat) * 0.045
    h = (max_lat - min_lat) * 0.014

    # White backing box
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

    # Alternating black & white segments
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
        # Resize smoothly to max height 95px
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
# 4. RENDER INDIVIDUAL SPECIES MAP
# ==============================================================================
def render_species_map(
    species_name: str,
    sp_df: pd.DataFrame,
    conservancies: list[tuple[str, gpd.GeoDataFrame, str]],
    rivers_gdf: gpd.GeoDataFrame | None,
    rivers_label: str,
    logo_bytes: bytes | None,
    report_period: str,
    marker_color: str = "#D92323",
    zoom_level: int = 11,
    label_conservancies_on_map: bool = True
) -> bytes:
    """
    Renders a publication-grade 300 DPI map for a single wildlife species with:
      - OpenTopoMap basemap
      - Multiple conservancy boundaries (distinct colors + names on map & legend)
      - Rivers & Luggas overlay
      - Species sighting points + total sightings count
      - Company logo in top-right
      - North arrow & 360° bearing rose + Geodesic scale bar
    """
    # Compute map bounding box from conservancies + sighting points
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

    # Add 10% geographic margin
    lon_span = max(max_lon - min_lon, 0.05)
    lat_span = max(max_lat - min_lat, 0.05)
    min_lon -= lon_span * 0.10
    max_lon += lon_span * 0.10
    min_lat -= lat_span * 0.10
    max_lat += lat_span * 0.10

    fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=220)

    # 1. OpenTopoMap Basemap
    basemap_img, extent = fetch_opentopomap_basemap(min_lon, min_lat, max_lon, max_lat, zoom=zoom_level)
    ax.imshow(basemap_img, extent=extent, aspect="auto", zorder=1)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    legend_handles = []

    # 2. Conservancy / Patrol Area Boundaries
    for idx, (layer_label, c_gdf, edge_color) in enumerate(conservancies):
        if c_gdf.empty:
            continue
        c_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor=edge_color,
            linewidth=2.3,
            linestyle="-" if idx == 0 else "--",
            zorder=5 + idx
        )
        # Subtle tint fill for primary conservancy
        c_gdf.plot(ax=ax, facecolor=edge_color, alpha=0.06, edgecolor="none", zorder=4)

        # Label conservancy / polygon names on map
        unique_names = [n for n in c_gdf["Display_Name"].unique() if n]
        if label_conservancies_on_map:
            for _, row in c_gdf.iterrows():
                if row.geometry is not None and not row.geometry.is_empty:
                    pt = row.geometry.representative_point()
                    if min_lon <= pt.x <= max_lon and min_lat <= pt.y <= max_lat:
                        ax.text(
                            pt.x,
                            pt.y,
                            str(row["Display_Name"]).upper(),
                            fontsize=8.2,
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
                facecolor="none",
                edgecolor=edge_color,
                linewidth=2.2,
                linestyle="-" if idx == 0 else "--",
                label=f"{disp_legend_name}"
            )
        )

    # 3. Rivers & Luggas Overlay
    if rivers_gdf is not None and not rivers_gdf.empty:
        rivers_gdf.plot(ax=ax, color="#0066CC", linewidth=1.35, alpha=0.88, zorder=7)
        legend_handles.append(
            mlines.Line2D([], [], color="#0066CC", linewidth=1.8, label=rivers_label)
        )

    # 4. Species Sightings Points
    total_sightings = len(sp_df)
    if total_sightings > 0:
        ax.scatter(
            sp_df["Lon"],
            sp_df["Lat"],
            s=68,
            c=marker_color,
            edgecolors="white",
            linewidths=1.1,
            alpha=0.92,
            zorder=12
        )

    legend_handles.insert(
        0,
        mlines.Line2D(
            [], [],
            color="white",
            marker="o",
            markerfacecolor=marker_color,
            markeredgecolor="white",
            markersize=9,
            label=f"{species_name} Sightings (n = {total_sightings})"
        )
    )

    # 5. Cartographic North Arrow & Bearing Rose + Scale Bar + Logo
    draw_compass_and_bearing(ax, x_frac=0.075, y_frac=0.88)
    draw_scale_bar(ax, min_lon, max_lon, min_lat, max_lat)
    add_top_right_logo(ax, logo_bytes)

    # 6. Legend (Bottom-Right)
    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        title="MAP LEGEND",
        frameon=True,
        facecolor="white",
        edgecolor="#1F4E78",
        framealpha=0.94,
        fontsize=8.5,
        title_fontsize=9.2
    )
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(22)

    # 7. Title & Coordinate Frame
    ax.set_title(
        f"{species_name.upper()} DISTRIBUTION MAP — {report_period.upper()}\n"
        f"Total Sightings Recorded: {total_sightings}   |   Basemap: OpenTopoMap",
        fontsize=12.5,
        fontweight="bold",
        color="#1F4E78",
        pad=12
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Latitude (°S)", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#555555", alpha=0.45, zorder=2)

    # Attribution note
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

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# STREAMLIT PAGE UI
# ==============================================================================
st.title("🗺️ Wildlife Sightings Distribution Maps (Per Species)")
st.markdown(
    "Generate individual cartographic distribution maps for each wildlife species over **OpenTopoMap** (`tile.opentopomap.org`), "
    "complete with **multiple conservancy/patrol boundaries**, **rivers & luggas**, **company logo**, **North/bearing rose**, and **scale bar**."
)

# --- TOP UPLOADERS ---
col1, col2 = st.columns(2)
with col1:
    sightings_file = st.file_uploader(
        "1️⃣ Upload EarthRanger Wildlife Sightings CSV / Excel (`Event Export ...csv`)",
        type=["csv", "xlsx", "xls"],
        key="wildlife_csv_uploader"
    )
    logo_file = st.file_uploader(
        "2️⃣ Upload Company Logo (PNG / JPG — placed in Top-Right)",
        type=["png", "jpg", "jpeg"],
        key="company_logo_uploader"
    )

with col2:
    conservancy_zips = st.file_uploader(
        "3️⃣ Upload Conservancy / Patrol Area Boundary Shapefiles (`.zip` — upload one or multiple)",
        type=["zip"],
        accept_multiple_files=True,
        key="conservancy_zip_uploader"
    )
    rivers_zip = st.file_uploader(
        "4️⃣ Upload Rivers & Luggas Shapefile (`.zip`)",
        type=["zip"],
        key="rivers_zip_uploader"
    )

with st.sidebar:
    st.subheader("Map Styling & Cartography")
    report_period = st.text_input("Report Period Label", value="November 2026", key="map_period_label")
    zoom_level = st.slider("OpenTopoMap Tile Zoom Level", min_value=9, max_value=13, value=11, key="topo_zoom")
    rivers_legend_name = st.text_input("Rivers Layer Legend Label", value="Rivers & Luggas", key="rivers_legend_txt")
    show_poly_labels = st.checkbox("Display Conservancy / Patrol Area Names on Map", value=True, key="show_poly_names")

if sightings_file is not None:
    if sightings_file.name.lower().endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(sightings_file)
    else:
        raw_df = pd.read_csv(sightings_file)

    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    sp_col = next((c for c in raw_df.columns if c.lower() == "species"), None)
    lat_col = next((c for c in raw_df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in raw_df.columns if "lon" in c.lower()), None)

    if not sp_col or not lat_col or not lon_col:
        st.error("Uploaded sightings file must contain 'Species', 'Latitude', and 'Longitude' columns.")
        st.stop()

    raw_df["Species_Clean"] = (
        raw_df[sp_col]
        .fillna("Unknown")
        .astype(str)
        .str.replace(r"\s*\(inform Monitoring Team\)", "", regex=True)
        .str.strip()
    )
    raw_df["Lat"] = pd.to_numeric(raw_df[lat_col], errors="coerce")
    raw_df["Lon"] = pd.to_numeric(raw_df[lon_col], errors="coerce")
    raw_df["Count_Raw"] = pd.to_numeric(raw_df["Count"], errors="coerce").fillna(1) if "Count" in raw_df.columns else 1

    # Flag extreme count typos (> 50,000 in a single record) while preserving sighting count
    raw_df["Count_Clean"] = np.where(raw_df["Count_Raw"] > 50000, np.nan, raw_df["Count_Raw"])

    valid_geo_df = raw_df.dropna(subset=["Lat", "Lon"]).copy()

    # Load Conservancy ZIP shapefiles
    conservancies_loaded = []
    boundary_palette = ["#1B4F72", "#7D3C98", "#B9770E", "#117864", "#922B21", "#2874A6"]
    if conservancy_zips:
        st.subheader("Uploaded Conservancy / Patrol Area Boundaries")
        name_cols = st.columns(min(len(conservancy_zips), 3))
        for idx, zfile in enumerate(conservancy_zips):
            default_label = os.path.splitext(zfile.name)[0].replace("_", " ").title()
            with name_cols[idx % len(name_cols)]:
                custom_name = st.text_input(
                    f"Boundary {idx + 1} Name ({zfile.name})",
                    value=default_label,
                    key=f"cons_name_{idx}_{zfile.name}"
                )
            try:
                gdf_c = load_shapefile_zip(zfile, fallback_name=custom_name)
                # If shapefile has a single generic name or user edited the label, apply custom name if needed
                if gdf_c["Display_Name"].nunique() == 1 and gdf_c["Display_Name"].iloc[0] == default_label:
                    gdf_c["Display_Name"] = custom_name
                conservancies_loaded.append((custom_name, gdf_c, boundary_palette[idx % len(boundary_palette)]))
            except Exception as e:
                st.warning(f"Could not read {zfile.name}: {e}")

    # Load Rivers & Luggas ZIP shapefile
    rivers_gdf = None
    if rivers_zip is not None:
        try:
            rivers_gdf = load_shapefile_zip(rivers_zip, fallback_name=rivers_legend_name)
        except Exception as e:
            st.warning(f"Could not read Rivers/Luggas shapefile: {e}")

    logo_bytes = logo_file.getvalue() if logo_file is not None else None

    # --- SUMMARY TABLE: TOTAL SIGHTINGS PER SPECIES ---
    species_summary = (
        raw_df.groupby("Species_Clean", as_index=False)
        .agg(
            **{
                "Total Sightings": ("Species_Clean", "count"),
                "Mapped GPS Sightings": ("Lat", lambda s: int(s.notna().sum())),
                "Recorded Individuals (Cleaned)": ("Count_Clean", lambda s: int(s.sum())),
            }
        )
        .sort_values(by="Total Sightings", ascending=False)
        .reset_index(drop=True)
    )
    species_summary["Share of Sightings (%)"] = (
        species_summary["Total Sightings"] / species_summary["Total Sightings"].sum() * 100
    ).round(2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Wildlife Sightings", f"{len(raw_df):,}")
    m2.metric("Unique Species Recorded", f"{len(species_summary)}")
    m3.metric("Top Sighted Species", species_summary.iloc[0]["Species_Clean"], delta=f"{species_summary.iloc[0]['Total Sightings']} sightings", delta_color="off")
    m4.metric("Conservancy Layers Loaded", f"{len(conservancies_loaded)} boundaries")

    st.subheader(f"1. Total Number of Sightings per Species ({report_period})")
    st.dataframe(species_summary, use_container_width=True, hide_index=True)

    st.divider()

    # --- GENERATE PER-SPECIES MAPS ---
    st.subheader(f"2. Per-Species Wildlife Distribution Maps ({report_period})")
    all_species_list = species_summary["Species_Clean"].tolist()
    default_selected = all_species_list[:6]  # Top 6 species by default (Lion, Elephant, Buffalo, Giraffe, Leopard, Cheetah)

    selected_species = st.multiselect(
        "Select Species to Generate Individual Maps For (each species gets its own full map):",
        options=all_species_list,
        default=default_selected
    )

    marker_palette = [
        "#D92323", "#E67E22", "#8E44AD", "#16A085", "#C0392B",
        "#2980B9", "#D35400", "#27AE60", "#884EA0", "#B03A2E"
    ]

    if selected_species:
        if st.button("🗺️ Generate All Selected Species Maps & ZIP Bundle", type="primary"):
            zip_buffer = io.BytesIO()
            generated_maps = {}

            progress = st.progress(0)
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, sp in enumerate(selected_species):
                    sp_subset = valid_geo_df[valid_geo_df["Species_Clean"] == sp]
                    color = marker_palette[idx % len(marker_palette)]
                    png_bytes = render_species_map(
                        species_name=sp,
                        sp_df=sp_subset,
                        conservancies=conservancies_loaded,
                        rivers_gdf=rivers_gdf,
                        rivers_label=rivers_legend_name,
                        logo_bytes=logo_bytes,
                        report_period=report_period,
                        marker_color=color,
                        zoom_level=zoom_level,
                        label_conservancies_on_map=show_poly_labels
                    )
                    generated_maps[sp] = png_bytes
                    safe_sp = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in sp).strip().replace(" ", "_")
                    zf.writestr(f"{safe_sp}_Sightings_Map_{report_period.replace(' ', '_')}.png", png_bytes)
                    progress.progress((idx + 1) / len(selected_species))

            zip_buffer.seek(0)
            st.download_button(
                label=f"📦 Download All {len(selected_species)} Species Maps (.ZIP)",
                data=zip_buffer,
                file_name=f"Mara_Triangle_Species_Maps_{report_period.replace(' ', '_')}.zip",
                mime="application/zip"
            )

            # Display each species map cleanly in 2-column grid with individual download buttons
            cols = st.columns(2)
            for idx, sp in enumerate(selected_species):
                with cols[idx % 2]:
                    st.image(generated_maps[sp], caption=f"{sp} — Total Sightings: {len(valid_geo_df[valid_geo_df['Species_Clean'] == sp])}", use_container_width=True)
                    safe_sp = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in sp).strip().replace(" ", "_")
                    st.download_button(
                        label=f"📥 Download {sp} Map (.png)",
                        data=generated_maps[sp],
                        file_name=f"{safe_sp}_Map.png",
                        mime="image/png",
                        key=f"dl_map_{idx}_{safe_sp}"
                    )
else:
    st.info("👆 Upload your **EarthRanger Wildlife Sightings CSV** (`Event Export 2026-11-20.csv`), **Conservancy Boundary ZIP Shapefiles**, **Rivers & Luggas ZIP Shapefile**, and **Company Logo** above to generate maps for each species.")
