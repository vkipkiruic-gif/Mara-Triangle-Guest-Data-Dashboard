import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st

# ==============================================================================
# 1. PARSE EARTHRANGER MANUAL RAINFALL STATIONS (PURUNGAT, ISEIYA, KILO 2, OLOOLOLO)
# ==============================================================================
def parse_earthranger_rainfall(uploaded_file):
    """
    Parses an EarthRanger Rainfall CSV/Excel export containing station-level rainfall gauges.
    Returns normalized daily station records: ['Date', 'Station', 'Rain_mm'].
    """
    fname = uploaded_file.name
    if fname.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df.columns = [str(c).strip() for c in df.columns]

    ts_col = next((c for c in df.columns if "reported_at" in c.lower() or "date" in c.lower() or "time" in c.lower()), None)
    st_col = next((c for c in df.columns if c.lower() == "station" or "station" in c.lower()), None)
    rain_col = next((c for c in df.columns if "rainfall" in c.lower() or "amount" in c.lower() or "rain" in c.lower()), None)

    if not ts_col or not st_col or not rain_col:
        raise ValueError("EarthRanger file must contain 'Reported_At', 'Station', and 'Amount_Of_Rainfall_(mm)' columns.")

    df["dt"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    df["Date"] = df["dt"].dt.date
    df["Station"] = df[st_col].fillna("Unknown").astype(str).str.strip()
    df["Rain_mm"] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0)

    er_daily = df.groupby(["Date", "Station"], as_index=False)["Rain_mm"].sum()
    return er_daily


# ==============================================================================
# 2. PARSE TAHMO / ZENTRA WEATHER STATION (SERENA AIRSTRIP)
# ==============================================================================
def parse_tahmo_weather(uploaded_file, station_label: str = "Serena Airstrip (TAHMO)", timezone_offset_hours: int = 0):
    """
    Parses a TAHMO / ZENTRA Cloud CSV or Excel export (e.g., 06-01522 Serena Airstrip / TA00569).
    Returns:
      - tahmo_station_records: ['Date', 'Station', 'Rain_mm'] for merging with EarthRanger stations
      - tahmo_full_daily: full meteorological daily table (rainfall, temp, wind, solar)
      - logger_id: extracted logger ID string
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

    tahmo_station_records = pd.DataFrame({
        "Date": daily["Date"],
        "Station": station_label.strip(),
        "Rain_mm": daily["Daily Rainfall (mm)"]
    })

    return tahmo_station_records, daily, logger_id


# ==============================================================================
# 3. MERGE & SUMMARIZE ALL STATIONS (EARTHRANGER + TAHMO)
# ==============================================================================
def build_combined_station_summaries(records_df: pd.DataFrame, fill_missing_calendar_days: bool = True):
    """
    Combines EarthRanger stations and TAHMO station into:
      - station_summary: Station totals, rainy days, avg per rainy day, highest single day, share (%)
      - daily_matrix_df: Date x Station daily rainfall table + Daily Station Mean + Combined Total
      - station_cols: list of station column names
    """
    wet_events = records_df[records_df["Rain_mm"] > 0].copy()

    all_stations = sorted(records_df["Station"].unique().tolist())
    summary_rows = []
    for st_name in all_stations:
        st_wet = wet_events[wet_events["Station"] == st_name]
        tot_rain = float(st_wet["Rain_mm"].sum())
        rainy_days = int(len(st_wet))
        avg_rainy = float(st_wet["Rain_mm"].mean()) if rainy_days > 0 else 0.0
        max_day = float(st_wet["Rain_mm"].max()) if rainy_days > 0 else 0.0
        peak_date = str(st_wet.loc[st_wet["Rain_mm"].idxmax(), "Date"]) if rainy_days > 0 else "—"
        summary_rows.append({
            "Station": st_name,
            "Total Rainfall (mm)": round(tot_rain, 2),
            "Rainy Days (>0 mm)": rainy_days,
            "Avg per Rainy Day (mm)": round(avg_rainy, 2),
            "Highest Single Day (mm)": round(max_day, 2),
            "Peak Rainfall Date": peak_date
        })

    station_summary = pd.DataFrame(summary_rows).sort_values(by="Total Rainfall (mm)", ascending=False).reset_index(drop=True)
    total_all = station_summary["Total Rainfall (mm)"].sum()
    station_summary["Share (%)"] = (station_summary["Total Rainfall (mm)"] / total_all * 100).round(2) if total_all > 0 else 0.0

    pivot_df = records_df.pivot_table(index="Date", columns="Station", values="Rain_mm", aggfunc="sum", fill_value=0.0)
    ordered_cols = [s for s in station_summary["Station"].tolist() if s in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]

    if fill_missing_calendar_days and not pivot_df.empty:
        full_dates = pd.date_range(pivot_df.index.min(), pivot_df.index.max(), freq="D").date
        pivot_df = pivot_df.reindex(full_dates, fill_value=0.0)
        pivot_df.index.name = "Date"

    pivot_df = pivot_df.round(2)
    pivot_df["Daily Station Mean (mm)"] = pivot_df[ordered_cols].mean(axis=1).round(2)
    pivot_df["Combined Stations Total (mm)"] = pivot_df[ordered_cols].sum(axis=1).round(2)
    daily_matrix_df = pivot_df.reset_index()

    return station_summary, daily_matrix_df, ordered_cols


# ==============================================================================
# 4. EXCEL WORKBOOK GENERATOR
# ==============================================================================
def _style_sheet(ws, title, df_export, sum_all_numeric=False):
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

    ws["A1"] = title
    ws["A1"].font = title_font

    df_copy = df_export.copy()
    if "Date" in df_copy.columns:
        df_copy["Date"] = df_copy["Date"].astype(str)

    for c_idx, col_name in enumerate(df_copy.columns, 1):
        cell = ws.cell(row=3, column=c_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center" if c_idx > 1 else "left", wrap_text=True)

    for r_idx, row in enumerate(df_copy.itertuples(index=False), 4):
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            col_header = str(df_copy.columns[c_idx - 1])
            if isinstance(val, (int, float)) and not pd.isna(val):
                cell.number_format = "#,##0" if ("Records" in col_header or "Days" in col_header) else "0.00"
                cell.alignment = Alignment(horizontal="right")
            else:
                cell.alignment = Alignment(horizontal="center" if c_idx > 1 else "left")

    if len(df_copy) > 0:
        tot_row = len(df_copy) + 4
        ws.cell(row=tot_row, column=1, value="TOTAL / AVERAGE").font = bold_font
        ws.cell(row=tot_row, column=1).fill = total_fill
        ws.cell(row=tot_row, column=1).border = total_border

        for c_idx in range(2, len(df_copy.columns) + 1):
            col_let = get_column_letter(c_idx)
            col_header = str(df_copy.columns[c_idx - 1])
            if col_header == "Peak Rainfall Date":
                cell = ws.cell(row=tot_row, column=c_idx, value="—")
                cell.alignment = Alignment(horizontal="center")
            elif sum_all_numeric or col_header in ("Total Rainfall (mm)", "Share (%)", "Daily Rainfall (mm)", "Records Count"):
                cell = ws.cell(row=tot_row, column=c_idx, value=f"=SUM({col_let}4:{col_let}{tot_row - 1})")
                cell.number_format = "#,##0" if ("Records" in col_header or "Days" in col_header) else "0.00"
                cell.alignment = Alignment(horizontal="right")
            elif col_header == "Cumulative Rainfall (mm)" or "Highest" in col_header or "Max" in col_header or "Peak" in col_header:
                cell = ws.cell(row=tot_row, column=c_idx, value=f"=MAX({col_let}4:{col_let}{tot_row - 1})")
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="right")
            elif "Min" in col_header:
                cell = ws.cell(row=tot_row, column=c_idx, value=f"=MIN({col_let}4:{col_let}{tot_row - 1})")
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="right")
            else:
                cell = ws.cell(row=tot_row, column=c_idx, value=f"=AVERAGE({col_let}4:{col_let}{tot_row - 1})")
                cell.number_format = "0.0" if "Days" in col_header else "0.00"
                cell.alignment = Alignment(horizontal="right")

            cell.font = bold_font
            cell.fill = total_fill
            cell.border = total_border

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)


def build_combined_weather_excel(
    station_summary: pd.DataFrame,
    daily_matrix_df: pd.DataFrame,
    tahmo_daily_df: pd.DataFrame | None,
    period_label: str,
    tahmo_station_name: str
):
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "All Stations Summary"
    _style_sheet(ws1, f"MARA TRIANGLE RAINFALL BY STATION — {period_label.upper()}", station_summary, sum_all_numeric=False)

    ws2 = wb.create_sheet("Daily Rainfall by Station")
    _style_sheet(ws2, f"MARA TRIANGLE DAILY RAINFALL BY STATION (MM) — {period_label.upper()}", daily_matrix_df, sum_all_numeric=True)

    if tahmo_daily_df is not None and not tahmo_daily_df.empty:
        ws3 = wb.create_sheet("Serena TAHMO Met Data")
        _style_sheet(ws3, f"TAHMO WEATHER STATION ({tahmo_station_name.upper()}) — {period_label.upper()}", tahmo_daily_df, sum_all_numeric=False)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ==============================================================================
# STREAMLIT PAGE UI (DUAL UPLOADER: EARTHRANGER + TAHMO UNIFIED)
# ==============================================================================
st.title("🌧️ Weather & Rainfall Station Summary")
st.markdown(
    "Upload **both** your **EarthRanger Station Rainfall file** (`Purungat`, `Iseiya`, `Kilo 2`, `Oloololo`, etc.) "
    "and your **TAHMO Weather Station file** (`Serena Airstrip`) to analyze all stations together in one unified report."
)

col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_er = st.file_uploader(
        "1️⃣ Upload EarthRanger Rainfall CSV / Excel (`weather rainfall.csv`)",
        type=["csv", "xlsx", "xls"],
        key="er_rainfall_uploader"
    )
with col_up2:
    uploaded_tahmo = st.file_uploader(
        "2️⃣ Upload TAHMO Weather CSV / Excel (`06-01522` Serena Airstrip)",
        type=["csv", "xlsx", "xls"],
        key="tahmo_rainfall_uploader"
    )

with st.sidebar:
    st.subheader("Weather Report Settings")
    report_period = st.text_input("Report Period Label", value="November 2026", key="unified_weather_period")
    tahmo_label = st.text_input("TAHMO Station Display Name", value="Serena Airstrip (TAHMO)", key="tahmo_station_label")
    tz_shift = st.selectbox(
        "TAHMO Timestamp Adjustment",
        options=[0, 3],
        format_func=lambda x: "As Exported (+0h)" if x == 0 else "Convert UTC to EAT (+3h)",
        index=0,
        key="unified_tz_shift"
    )
    fill_dry_days = st.checkbox("Include 0 mm dry days in daily breakdown", value=True, key="unified_fill_dry")

if uploaded_er is not None or uploaded_tahmo is not None:
    try:
        records_list = []
        tahmo_full_daily = None
        tahmo_logger = None

        if uploaded_er is not None:
            er_df = parse_earthranger_rainfall(uploaded_er)
            records_list.append(er_df)

        if uploaded_tahmo is not None:
            tahmo_st_df, tahmo_full_daily, tahmo_logger = parse_tahmo_weather(
                uploaded_tahmo, station_label=tahmo_label, timezone_offset_hours=tz_shift
            )
            records_list.append(tahmo_st_df)

        combined_records = pd.concat(records_list, ignore_index=True)
        station_summary, daily_matrix_df, ordered_stations = build_combined_station_summaries(
            combined_records, fill_missing_calendar_days=fill_dry_days
        )

        wettest_st = station_summary.iloc[0]
        driest_st = station_summary.iloc[-1]
        conservancy_avg = float(station_summary["Total Rainfall (mm)"].mean())
        wet_events = combined_records[combined_records["Rain_mm"] > 0]
        if not wet_events.empty:
            peak_row = wet_events.loc[wet_events["Rain_mm"].idxmax()]
            peak_text = f"{peak_row['Rain_mm']:.2f} mm ({peak_row['Station']} on {peak_row['Date']})"
        else:
            peak_text = "0.00 mm"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Active Weather Stations", f"{len(ordered_stations)} stations")
        k2.metric("Wettest Station", f"{wettest_st['Station']}", delta=f"{wettest_st['Total Rainfall (mm)']:.2f} mm", delta_color="off")
        k3.metric("All-Stations Mean Rainfall", f"{conservancy_avg:.2f} mm", delta=f"Lowest: {driest_st['Station']} ({driest_st['Total Rainfall (mm)']:.2f} mm)", delta_color="off")
        k4.metric("Highest Single-Day Downpour", peak_text.split(" (")[0], delta=peak_text.split(" (")[1].rstrip(")") if "(" in peak_text else "", delta_color="off")

        station_list_text = ", ".join(
            f"**{r['Station']}** ({r['Total Rainfall (mm)']:.2f} mm)" for _, r in station_summary.iterrows()
        )
        st.info(
            f"**Auto-Generated Weather & Rainfall Narrative ({report_period}):** Across the **{len(ordered_stations)} monitoring stations** "
            f"in the Mara Triangle, **{wettest_st['Station']}** recorded the highest cumulative rainfall at **{wettest_st['Total Rainfall (mm)']:.2f} mm**, "
            f"followed by {', '.join(station_list_text.split(', ')[1:]) if len(station_summary) > 1 else station_list_text}. "
            f"The conservancy-wide station average was **{conservancy_avg:.2f} mm**, with the highest single-day rainfall of **{peak_text}**."
        )

        excel_bytes = build_combined_weather_excel(
            station_summary, daily_matrix_df, tahmo_full_daily, report_period, tahmo_label
        )
        st.download_button(
            label="📥 Download Unified Weather & Rainfall Report (.xlsx)",
            data=excel_bytes,
            file_name=f"Mara_Triangle_Weather_Report_{report_period.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        st.subheader(f"1. Total Rainfall Comparison Across All Stations ({report_period})")
        c_left, c_right = st.columns([1.2, 1.2])
        with c_left:
            st.dataframe(station_summary, use_container_width=True, hide_index=True)
        with c_right:
            fig1, ax1 = plt.subplots(figsize=(7.8, 4.4))
            palette = ["#1F4E78", "#2E8B57", "#00AEEF", "#D95F02", "#7570B3", "#E7298A"]
            colors = [palette[i % len(palette)] for i in range(len(station_summary))]
            bars = ax1.bar(station_summary["Station"], station_summary["Total Rainfall (mm)"], color=colors, width=0.55)
            ax1.set_ylabel("Total Rainfall (mm)")
            ax1.set_title(f"Total Rainfall by Station — {report_period}", fontweight="bold")
            plt.xticks(rotation=18, ha="right", fontsize=9)
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            max_val = station_summary["Total Rainfall (mm)"].max() if len(station_summary) > 0 else 1
            for bar in bars:
                h = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (max_val * 0.02),
                    f"{h:.1f} mm",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold"
                )
            st.pyplot(fig1)

        st.subheader(f"2. Daily Rainfall Distribution Across All Stations ({report_period})")
        fig2, ax2 = plt.subplots(figsize=(11.5, 4.6))
        dates_str = [str(d) for d in daily_matrix_df["Date"]]
        x = np.arange(len(dates_str))
        n_st = len(ordered_stations)
        width = min(0.82 / max(n_st, 1), 0.20)
        palette = ["#1F4E78", "#2E8B57", "#00AEEF", "#D95F02", "#7570B3", "#E7298A"]

        for idx, st_name in enumerate(ordered_stations):
            offset = (idx - (n_st - 1) / 2) * width
            ax2.bar(x + offset, daily_matrix_df[st_name], width=width, label=st_name, color=palette[idx % len(palette)])

        ax2.set_xticks(x)
        ax2.set_xticklabels(dates_str, rotation=45, ha="right", fontsize=8.5)
        ax2.set_ylabel("Rainfall (mm)")
        ax2.set_title(f"Daily Rainfall Amount per Station (mm) — {report_period}", fontweight="bold")
        ax2.legend(title="Station", ncol=min(n_st, 5))
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig2)

        st.subheader(f"3. Daily Rainfall Breakdown Table by Station ({report_period})")
        st.dataframe(daily_matrix_df, use_container_width=True, hide_index=True)

        if tahmo_full_daily is not None and not tahmo_full_daily.empty:
            st.divider()
            st.subheader(f"4. {tahmo_label} — Temperature & Meteorological Summary ({report_period})")
            t_col1, t_col2 = st.columns([1.2, 1])
            with t_col1:
                fig3, ax3 = plt.subplots(figsize=(8, 3.8))
                t_dates = pd.to_datetime(tahmo_full_daily["Date"])
                ax3.plot(t_dates, tahmo_full_daily["Avg Temperature (°C)"], color="#D95F02", marker="o", markersize=4, linewidth=1.8, label="Avg Temp (°C)")
                ax3.fill_between(t_dates, tahmo_full_daily["Min Temperature (°C)"], tahmo_full_daily["Max Temperature (°C)"], color="#D95F02", alpha=0.15, label="Min–Max Daily Range")
                ax3.set_ylabel("Temperature (°C)")
                ax3.set_title(f"Daily Air Temperature (°C) — {tahmo_label}", fontweight="bold")
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                plt.xticks(rotation=30, ha="right", fontsize=8.5)
                ax3.legend()
                ax3.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig3)
            with t_col2:
                met_cols = [
                    "Date", "Daily Rainfall (mm)", "Avg Temperature (°C)",
                    "Min Temperature (°C)", "Max Temperature (°C)", "Avg Wind Speed (m/s)"
                ]
                st.dataframe(tahmo_full_daily[met_cols], use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error processing Weather & Rainfall files: {e}")
else:
    st.info("👆 Upload your **EarthRanger Rainfall file** (`weather rainfall.csv`) and/or your **TAHMO Weather Station file** (`06-01522`) above to generate the combined multi-station report.")
