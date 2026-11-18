import streamlit as st

st.set_page_config(
    page_title="Mara Triangle Mapping and Arrivals Report",
    page_icon="🦁",
    layout="wide"
)

# Sidebar Branding / Header
with st.sidebar:
    st.markdown("## 🦁 Mara Triangle")
    st.markdown("**Mapping & Arrivals Report**")
    st.divider()

# Define Navigation Pages under the main report umbrella
guest_page = st.Page("pages/1_Guest_Data.py", title="Guest Data", icon="👥", default=True)
speed_page = st.Page("pages/2_Speed_Tracker_Reports.py", title="Speed Tracker Reports", icon="🚦")
weather_page = st.Page("pages/3_Weather_Summary.py", title="Weather Summary", icon="🌧️")

pg = st.navigation({
    "Mara Triangle Mapping & Arrivals Report": [
        guest_page,
        speed_page,
        weather_page,
    ]
})

pg.run()
