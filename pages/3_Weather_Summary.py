import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st

# ==============================================================================
# TAHMO / ZENTRA WEATHER FILE PARSER
# ==============================================================================
def parse_tahmo_file(uploaded_file, timezone_offset_hours: int = 0):
    """
    Parses a TAHMO / ZENTRA Cloud CSV or Excel export (e.g., 06-01522 Serena Airstrip / TA00569).
    Automatically skips the 2 metadata header rows and extracts daily rainfall, temperature,
    humidity/VPD, wind speed, and solar radiation summaries.
    """
    fname = uploaded_file.name
    if fname.lower().endswith((".xlsx", ".xls")):
        meta_df = pd.read_excel(uploaded_file, header=None, nrows=2)
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, skiprows=2)
    else:
        meta_df = pd.read_csv(uploaded_file, header=None, nrows=2)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=2)

    logger_id = str(meta_df.iloc[0, 0]).replace("\ufeff", "").strip() if pd.notna(meta_df.iloc[0, 0]) else "TA00569"

    df.columns = [str(c).strip() for c in df.columns]

    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), df.columns[0])
    precip_col = next((c for c in df.columns if "precipitation" in c.lower()), None)
    temp_col = next((c for c in df.columns if "air temperature" in c.lower()), None)
    wind_col = next((c for c in df.columns if "wind speed" in c.lower()), None)
    gust_col = next((c for c in df.columns if "gust speed" in c.lower()), None)
    solar_col = next((c for c in df.columns if "solar radiation" in c.lower()), None)

    df["dt"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    if timezone_offset_hours != 0:
        df["dt"] = df["dt"] + pd.Timedelta(hours=timezone_offset_hours)

    df["Date"] = df["dt"].dt.date

    df["Precip_mm"] = pd.to_numeric(df[precip_col], errors="coerce").fillna(0.0) if precip_col else 0.0
    df["Temp_C"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else float("nan")
    df["Wind_ms"] = pd.to_numeric(df[wind_col], errors="coerce") if wind_col else float("nan")
    df["Gust_ms"] = pd.to_numeric(df[gust_col], errors="coerce") if gust_col else float("nan")
    df["Solar_Wm2"] = pd.to_numeric(df[solar_col], errors="coerce") if solar_col else float("nan")

    daily = df.groupby("Date", as_index=False).agg(
        **{
            "Daily Rainfall (mm)": ("Precip_mm", "sum"),
            "Avg Temperature (°C)": ("Temp_C", "mean"),
            "Min Temperature (°C)": ("Temp_C", "min"),
            "Max Temperature (°C)": ("Temp_C", "max"),
            "Avg Wind Speed (m/s)": ("Wind_ms", "mean"),
            "Peak Gust (m/s)": ("Gust_ms", "max"),
            "Avg Solar Radiation (W/m²)": ("Solar_Wm2", "mean"),
            "Records Count": ("Precip_mm", "count")
        }
    )

    daily["Daily Rainfall (mm)"] = daily["Daily Rainfall (mm)"].round(2)
    daily["Cumulative Rainfall (mm)"] = daily["Daily Rainfall (mm)"].cumsum().round(2)
    for col in ["Avg Temperature (°C)", "Min Temperature (°C)", "Max Temperature (°C)", "Avg Wind Speed (m/s)", "Peak Gust (m/s)"]:
        daily[col] = daily[col].round(2)
    daily["Avg Solar Radiation (W/m²)"] = daily["Avg Solar Radiation (W/m²)"].round(1)

    return {
        "logger_id": logger_id,
        "daily": daily
    }


# ==============================================================================
# EXCEL WORKBOOK BUILDER (IN-MEMORY)
# ==============================================================================
def build_weather_excel(daily_df: pd.DataFrame, station_label: str, period_label: str):
    wb = openpyxl.Workbook()
    title_font = Font(name="Calibri", size=14, bold=True, color="1F4E78")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=11)

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    total_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"), right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"), bottom=Side(style="thin", color="D9D9D9")
    )
    total_border = Border(top=Side(style="thin", color="000000"), bottom=Side(style="double", color="000000"))

    ws = wb.active
    ws.title = "Daily Weather Summary"
    ws["A1"] = f"MARA TRIANGLE WEATHER SUMMARY ({station_label.upper()}) — {period_label.upper()}"
    ws["A1"].font = title_font

    export_df = daily_df.copy()
    export_df["Date"] = export_df["Date"].astype(str)

    for c_idx, col_name in enumerate(export_df.columns, 1):
        cell = ws.cell(row=3, column=c_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center" if c_idx > 1 else "left")

    for r_idx, row in enumerate(export_df.itertuples(index=False), 4):
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            if c_idx > 1:
                col_header = str(export_df.columns[c_idx - 1])
                cell.number_format = "#,##0" if "Records" in col_header else "0.00"
                cell.alignment = Alignment(horizontal="right")

    if len(export_df) > 0:
        tot_row = len(export_df) + 4
        ws.cell(row=tot_row, column=1, value="TOTAL / AVERAGE").font = bold_font
        ws.cell(row=tot_row, column=1).fill = total_fill
        ws.cell(row=tot_row, column=1).border = total_border

        for c_idx in range(2, len(export_df.columns) + 1):
            col_let = get_column_letter(c_idx)
            col_header = str(export_df.columns[c_idx - 1])
            if col_header in ("Daily Rainfall (mm)", "Records Count"):
                formula = f"=SUM({col_let}4:{col_let}{tot_row - 1})"
            elif col_header == "Cumulative Rainfall (mm)" or "Max" in col_header or "Peak" in col_header:
                formula = f"=MAX({col_let}4:{col_let}{tot_row - 1})"
            elif "Min" in col_header:
                formula = f"=MIN({col_let}4:{col_let}{tot_row - 1})"
            else:
                formula = f"=AVERAGE({col_let}4:{col_let}{tot_row - 1})"

            cell = ws.cell(row=tot_row, column=c_idx, value=formula)
            cell.font = bold_font
            cell.fill = total_fill
            cell.border = total_border
            cell.number_format = "#,##0" if "Records" in col_header else "0.00"
            cell.alignment = Alignment(horizontal="right")

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ==============================================================================
# STREAMLIT PAGE UI
# ==============================================================================
st.title("🌧️ Weather Summary — Serena Airstrip (TAHMO)")
st.markdown(
    "Upload the monthly **TAHMO / ZENTRA Weather Station** export (`06-01522` / `TA00569 - Serena Airstrip`) "
    "to calculate daily rainfall amounts, total monthly rainfall, temperature trends, and export a formatted Excel report."
)

with st.sidebar:
    st.subheader("Upload TAHMO Weather Data")
    uploaded_weather = st.file_uploader(
        "Choose TAHMO CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="tahmo_file"
    )
    station_name = st.text_input("Weather Station Label", value="TA00569 - Serena Airstrip", key="weather_station")
    report_month = st.text_input("Report Month / Label", value="October 2026", key="weather_month")
    tz_shift = st.selectbox(
        "Timestamp Timezone Adjustment",
        options=[0, 3],
        format_func=lambda x: "As Exported in File (+0h)" if x == 0 else "Convert UTC to EAT (+3h)",
        index=0,
        key="weather_tz"
    )
    filter_month_only = st.checkbox("Exclude partial boundary days outside main month", value=False, key="weather_filter")

if uploaded_weather is not None:
    parsed = parse_tahmo_file(uploaded_weather, timezone_offset_hours=tz_shift)
    daily_df = parsed["daily"].copy()

    if filter_month_only and not daily_df.empty:
        dominant_ym = pd.Series([d.strftime("%Y-%m") for d in daily_df["Date"]]).mode()[0]
        daily_df = daily_df[[d.strftime("%Y-%m") == dominant_ym for d in daily_df["Date"]]].reset_index(drop=True)
        daily_df["Cumulative Rainfall (mm)"] = daily_df["Daily Rainfall (mm)"].cumsum().round(2)

    total_rain = float(daily_df["Daily Rainfall (mm)"].sum())
    rainy_days = int((daily_df["Daily Rainfall (mm)"] > 0).sum())
    wettest_row = daily_df.loc[daily_df["Daily Rainfall (mm)"].idxmax()]
    avg_temp = float(daily_df["Avg Temperature (°C)"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Monthly Rainfall", f"{total_rain:.2f} mm")
    c2.metric("Rainy Days (>0 mm)", f"{rainy_days} days")
    c3.metric("Wettest Single Day", f"{wettest_row['Daily Rainfall (mm)']:.2f} mm", delta=f"Date: {wettest_row['Date']}", delta_color="off")
    c4.metric("Average Daily Temp", f"{avg_temp:.1f} °C")

    st.info(
        f"**Auto-Generated Weather Narrative ({report_month}):** The total amount of rainfall recorded at **Serena Airstrip** "
        f"({station_name}, Logger `{parsed['logger_id']}`) was **{total_rain:.2f} mm** across **{rainy_days} rainy days**. "
        f"The highest single-day rainfall was **{wettest_row['Daily Rainfall (mm)']:.2f} mm** recorded on **{wettest_row['Date']}**, "
        f"and the average daily air temperature was **{avg_temp:.1f} °C**."
    )

    excel_bytes = build_weather_excel(daily_df, station_name, report_month)
    st.download_button(
        label="📥 Download Serena Airstrip Weather Report (.xlsx)",
        data=excel_bytes,
        file_name=f"Serena_Airstrip_Weather_{report_month.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    st.subheader(f"1. Daily Rainfall Amount — Serena Airstrip ({report_month})")
    fig1, ax1 = plt.subplots(figsize=(11, 4.3))
    dates_dt = pd.to_datetime(daily_df["Date"])
    bars = ax1.bar(dates_dt, daily_df["Daily Rainfall (mm)"], color="#4a6fe3", alpha=0.85, label="Daily Precipitation (mm)")
    ax1.plot(dates_dt, daily_df["Daily Rainfall (mm)"], color="#1f4e78", marker="o", markersize=4, linewidth=1.3)

    ax1.set_ylabel("Precipitation (mm)")
    ax1.set_xlabel("Date")
    ax1.set_title(f"Total Daily Precipitation — {station_name} ({report_month})", fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if h >= 0.5:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1f4e78")

    st.pyplot(fig1)

    st.subheader(f"2. Average Daily Temperature (°C) — Serena Airstrip ({report_month})")
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2
