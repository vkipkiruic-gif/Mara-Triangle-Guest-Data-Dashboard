import glob
import os
import streamlit as st

st.set_page_config(
    page_title="Mara Conservancy | Mara Triangle Operations & Mapping Dashboard",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# MARA CONSERVANCY BRAND STYLING (Khaki/Sand #BEB29A + Deep Slate/Charcoal)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Subtle top brand bar */
        .mara-topbar {
            background: linear-gradient(90deg, #1E242B 0%, #2C343D 65%, #3E382E 100%);
            border-bottom: 3px solid #BEB29A;
            border-radius: 10px;
            padding: 12px 20px;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.22);
        }
        .mara-topbar-title {
            color: #F5F2EB;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin: 0;
        }
        .mara-topbar-sub {
            color: #BEB29A;
            font-size: 0.80rem;
            margin: 2px 0 0 0;
            letter-spacing: 0.3px;
        }
        .mara-badge {
            background-color: rgba(190, 178, 154, 0.16);
            border: 1px solid #BEB29A;
            color: #EAE3D2;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            white-space: nowrap;
        }

        /* Sidebar section headers ("Arrivals & Operations Reports", "Mapping & GIS Reports") */
        [data-testid="stSidebarNavSectionHeader"] {
            color: #BEB29A !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            border-bottom: 1px solid rgba(190, 178, 154, 0.28);
            padding-bottom: 4px;
            margin-top: 10px;
        }

        /* Sidebar logo card */
        .mara-sidebar-logo-card {
            background: #FFFFFF;
            border: 2px solid #BEB29A;
            border-radius: 10px;
            padding: 10px 12px 8px 12px;
            margin-bottom: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _find_logo_path():
    candidates = [
        "mara_conservancy_logo.png",
        "mara_conservancy_logo.jpg",
        "logo.png",
        "logo.jpg",
        "assets/mara_conservancy_logo.png",
        "assets/logo.png",
        "images/mara_conservancy_logo.png",
        "images/logo.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


logo_path = _find_logo_path()

# Render Mara Conservancy Logo prominently in the sidebar header
with st.sidebar:
    if logo_path:
        st.markdown('<div class="mara-sidebar-logo-card">', unsafe_allow_html=True)
        st.image(logo_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="background:#1E242B;border:1.5px solid #BEB29A;border-radius:10px;padding:12px;margin-bottom:10px;text-align:center;">
                <div style="color:#F5F2EB;font-weight:800;font-size:1.05rem;letter-spacing:0.6px;">MARA CONSERVANCY</div>
                <div style="color:#BEB29A;font-size:0.76rem;margin-top:2px;">Protecting the Mara Triangle</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Top branded header bar across the main workspace
st.markdown(
    """
    <div class="mara-topbar">
        <div>
            <div class="mara-topbar-title">MARA CONSERVANCY — MARA TRIANGLE REPORTING PORTAL</div>
            <div class="mara-topbar-sub">Protecting the Mara Triangle &nbsp;|&nbsp; Guest Arrivals, Operations &amp; GIS Mapping Suite</div>
        </div>
        <div class="mara-badge">www.maratriangle.org</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# ORGANIZE ALL PAGES UNDER TWO MAIN TITLES:
#   1. Arrivals & Operations Reports
#   2. Mapping & GIS Reports
# ------------------------------------------------------------------------------
KNOWN_PAGE_META = {
    "guest": ("Guest Data", "🚙", "arrivals"),
    "arrival": ("Guest Arrivals", "🚙", "arrivals"),
    "speed": ("Speed Tracker Reports", "⚡", "arrivals"),
    "weather": ("Weather Summary", "🌦️", "arrivals"),
    "rainfall": ("Weather Summary", "🌦️", "arrivals"),
    "balloon": ("Balloon Reports", "🎈", "arrivals"),
    "wildlife": ("Wildlife Sightings Maps", "🗺️", "mapping"),
    "sighting": ("Wildlife Sightings Maps", "🗺️", "mapping"),
    "patrol": ("Patrol Tracks Maps", "🛰️", "mapping"),
    "track": ("Patrol Tracks Maps", "🛰️", "mapping"),
    "map": ("GIS Maps", "🗺️", "mapping"),
}

arrivals_pages = []
mapping_pages = []

page_files = sorted(glob.glob("pages/*.py"))
for idx, path in enumerate(page_files):
    fname = os.path.basename(path)
    if fname.startswith("__"):
        continue

    low = fname.lower()
    matched = None
    for key_tok, meta in KNOWN_PAGE_META.items():
        if key_tok in low:
            matched = meta
            break

    if matched:
        title, icon, section = matched
    else:
        clean_title = os.path.splitext(fname)[0].lstrip("0123456789_- ").replace("_", " ").strip()
        title = clean_title or fname
        icon = "📊"
        section = "arrivals"

    is_default = (len(arrivals_pages) == 0 and section == "arrivals")
    page_obj = st.Page(path, title=title, icon=icon, default=is_default)

    if section == "mapping":
        mapping_pages.append(page_obj)
    else:
        arrivals_pages.append(page_obj)

# Fallback default if only mapping pages exist
if not arrivals_pages and mapping_pages:
    first_path = sorted(glob.glob("pages/*.py"))[0]
    mapping_pages[0] = st.Page(first_path, title="Wildlife Sightings Maps", icon="🗺️", default=True)

nav_sections = {}
if arrivals_pages:
    nav_sections["Arrivals & Operations Reports"] = arrivals_pages
if mapping_pages:
    nav_sections["Mapping & GIS Reports"] = mapping_pages

if nav_sections:
    pg = st.navigation(nav_sections)
    pg.run()
else:
    st.error("No page files found inside the `pages/` folder.")
