import glob
import os
import streamlit as st

st.set_page_config(
    page_title="Mara Triangle Data & Mapping Dashboard",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom icons/titles for known pages; any other files in pages/ are included automatically
CUSTOM_META = {
    "5_Wildlife_Sightings_Maps.py": ("Wildlife Sightings Maps", "🗺️"),
    "6_Patrol_Tracks_Maps.py": ("Patrol Tracks Maps", "🛰️"),
}

page_files = sorted(glob.glob("pages/*.py"))
nav_pages = []

for idx, path in enumerate(page_files):
    fname = os.path.basename(path)
    if fname.startswith("__"):
        continue
    if fname in CUSTOM_META:
        title, icon = CUSTOM_META[fname]
        nav_pages.append(st.Page(path, title=title, icon=icon, default=(idx == 0)))
    else:
        # Turn e.g. "1_Guest_Analysis.py" into "Guest Analysis"
        clean_title = os.path.splitext(fname)[0].lstrip("0123456789_- ").replace("_", " ")
        nav_pages.append(st.Page(path, title=clean_title or fname, default=(idx == 0)))

if nav_pages:
    pg = st.navigation(nav_pages)
    pg.run()
else:
    st.error("No page files found in the `pages/` folder.")
