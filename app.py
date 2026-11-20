import streamlit as st

st.set_page_config(
    page_title="Mara Triangle Data & Mapping Dashboard",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define all dashboard pages (including Wildlife Sightings Maps & Patrol Tracks Maps)
wildlife_page = st.Page(
    "pages/5_Wildlife_Sightings_Maps.py",
    title="Wildlife Sightings Maps",
    icon="🗺️",
)
patrol_tracks_page = st.Page(
    "pages/6_Patrol_Tracks_Maps.py",
    title="Patrol Tracks Maps",
    icon="🛰️",
)

pg = st.navigation(
    {
        "GIS & Mapping Reports": [
            wildlife_page,
            patrol_tracks_page,
        ]
    }
)

pg.run()
