import io
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st

st.set_page_config(page_title="Speed Tracker Analytics (MT & GM)", layout="wide")

# ==============================================================================
# SAFEPACE CSV / EXCEL PARSER
# ==============================================================================
def classify_camera(location_raw: str, filename_raw: str):
    """Auto-detects camera name and region (MT vs GM) from file metadata or filename."""
    combined = f"{location_raw} {filename_raw}".upper()
    if "4KM" in combined:
        return "MT", "4KM"
    if "OLARE" in combined or "OOLARE" in combined:
        return "MT", "Olare"
    if "ODM" in combined:
        return "GM", "ODM"
    if "OLOOLAIMUTIA" in combined or "OLOO" in combined:
        return "GM", "Keekorok-Oloolaimutia"
    if "SEKENANI" in combined or "KEEKOROK" in combined:
        return "GM", "Keekorok-Sekenani"
    return "UNKNOWN", location_raw.strip()


def parse_safepace_file(uploaded_file):
    """Parses a SafePace Cloud CSV or Excel export cleanly, stripping 'H ' flags and summary rows."""
    fname = uploaded_file.name
    if fname.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded_file, header=None)
    else:
        raw = pd.read_csv(uploaded_file, header=None)

    location_raw = str(raw.iloc[0, 1]).strip() if pd.notna(raw.iloc[0, 1]) else fname
    report_period = str(raw.iloc[0, 3]).strip() if pd.notna(raw.iloc[0, 3]) else "Unknown Period"
    total_vehicles = int(float(raw.iloc[1, 3])) if pd.notna(raw.iloc[1, 3]) else 0

    region, camera_label = classify_camera(location_raw, fname)

    daily_rows = []
    for i in range(4, len(raw)):
        date_val = str(raw.iloc[i, 0]).strip() if pd.notna(raw.iloc[i, 0]) else ""
        if not date_val or date_val.lower() == "summary" or date_val.startswith("Generated"):
            break

        viol_str = str(raw.iloc[i, 1]).strip() if pd.notna(raw.iloc[i, 1]) else "0"
        speed_str = str(raw.iloc[i, 2]).strip() if pd.notna(raw.iloc[i, 2]) else "0"

        viol_clean = int(re.sub(r"[^\d]", "", viol_str)) if re.search(r"\d", viol_str) else 0
        speed_clean = int(re.sub(r"[^\d]", "", speed_str)) if re.search(r"\d", speed_str) else 0

        daily_rows.append({
            "Date": date_val,
            "Violations": viol_clean,
            "Max_Speed": speed_clean
        })

    df_daily = pd.DataFrame(daily_rows)
    return {
        "region": region,
        "camera": camera_label,
        "period": report_period,
        "total_vehicles": total_vehicles,
        "total_violations": int(df_daily["Violations"].sum()) if not df_daily.empty else 0,
        "daily": df_daily
    }


def aggregate_region(records, cameras_ordered):
    """Combines multiple camera records for a region (MT or GM) into unified daily tables & KPIs."""
    total_vehicles = sum(r["total_vehicles"] for r in records)
    total_violations = sum(r["total_violations"] for r in records)
    compliance_pct = round((1 - total_violations / total_vehicles) * 100, 2) if total_vehicles > 0 else 100.0

    all_dates = sorted({d for r in records for d in r["daily"]["Date"].tolist()})
    merged = pd.DataFrame({"Date": all_dates})

    peak_speed = 0
    peak_date = "N/A"
    peak_camera = "N/A"

    for cam in cameras_ordered:
        rec = next((r for r in records if r["camera"] == cam), None)
        if rec is not None and not rec["daily"].empty:
            sub = rec["daily"].rename(columns={
                "Violations": f"{cam} Violations",
                "Max_Speed": f"{cam} Max Speed (km/h)"
            })
            merged = pd.merge(merged, sub, on="Date", how="left")
            merged[f"{cam} Violations"] = merged[f"{cam} Violations"].fillna(0).astype(int)
            merged[f"{cam} Max Speed (km/h)"] = merged[f"{cam} Max Speed (km/h)"].fillna(0).astype(int)

            cam_max_idx = merged[f"{cam} Max Speed (km/h)"].idxmax()
            cam_max_val = int(merged.loc[cam_max_idx, f"{cam} Max Speed (km/h)"])
            if cam_max_val > peak_speed:
                peak_speed = cam_max_val
                peak_date = merged.loc[cam_max_idx, "Date"]
                peak_camera = cam
        else:
            merged[f"{cam} Violations"] = 0
            merged[f"{cam} Max Speed (km/h)"] = 0

    viol_cols = [f"{cam} Violations" for cam in cameras_ordered]
    merged["Total Daily Violations"] = merged[viol_cols].sum(axis=1)

    cam_summary_rows = []
    for cam in cameras_ordered:
        rec = next((r for r in records if r["camera"] == cam), None)
        v_cnt = rec["total_vehicles"] if rec else 0
        viol_cnt = rec["total_violations"] if rec else 0
        max_spd = int(merged[f"{cam} Max Speed (km/h)"].max())
        comp = round((1 - viol_cnt / v_cnt) * 100, 2) if v_cnt > 0 else 100.0
        cam_summary_rows.append({
            "Camera Location": cam,
            "Total Vehicles Scanned": v_cnt,
            "Speed Violations (>50 km/h)": viol_cnt,
            "Compliance Rate (%)": comp,
            "Highest Speed (km/h)": max_spd
        })
    cam_summary_df = pd.DataFrame(cam_summary_rows)

    return {
        "total_vehicles": total_vehicles,
        "total_violations": total_violations,
        "compliance_pct": compliance_pct,
        "peak_speed": peak_speed,
        "peak_date": peak_date,
        "peak_camera": peak_camera,
        "daily_df": merged,
        "cam_summary": cam_summary_df
    }


# ==============================================================================
# EXCEL WORKBOOK BUILDER (IN-MEMORY)
# ==============================================================================
def build_speed_excel(mt_data, gm_data, period_label):
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

    def write_df_sheet(ws, title, df_export, add_total=True):
        ws["A1"] = title
        ws["A1"].font = title_font

        for c_idx, col_name in enumerate(df_export.columns, 1):
            cell = ws.cell(row=3, column=c_idx, value=col_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center" if c_idx > 1 else "left")

        for r_idx, row in enumerate(df_export.itertuples(index=False), 4):
            for c_idx, val in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.font = regular_font
                cell.border = thin_border
                if c_idx > 1:
                    col_header = str(df_export.columns[c_idx - 1])
                    cell.number_format = "0.00" if "%" in col_header else "#,##0"
                    cell.alignment = Alignment(horizontal="right")

        if add_total and len(df_export) > 0:
            tot_row = len(df_export) + 4
            ws.cell(row=tot_row, column=1, value="TOTAL / PEAK").font = bold_font
            ws.cell(row=tot_row, column=1).fill = total_fill
            ws.cell(row=tot_row, column=1).border = total_border

            for c_idx in range(2, len(df_export.columns) + 1):
                col_let = get_column_letter(c_idx)
                col_header = str(df_export.columns[c_idx - 1])
                if "Max Speed" in col_header or "Highest Speed" in col_header:
                    formula = f"=MAX({col_let}4:{col_let}{tot_row - 1})"
                elif "%" in col_header:
                    formula = f"=AVERAGE({col_let}4:{col_let}{tot_row - 1})"
                else:
                    formula = f"=SUM({col_let}4:{col_let}{tot_row - 1})"
                cell = ws.cell(row=tot_row, column=c_idx, value=formula)
                cell.font = bold_font
                cell.fill = total_fill
                cell.border = total_border
                cell.number_format = "0.00" if "%" in col_header else "#,##0"
                cell.alignment = Alignment(horizontal="right")

        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)

    ws1 = wb.active
    ws1.title = "MT Camera Summary"
    write_df_sheet(ws1, f"MARA TRIANGLE (MT) SPEED TRACKER SUMMARY - {period_label.upper()}", mt_data["cam_summary"])

    ws2 = wb.create_sheet("MT Daily Breakdown")
    write_df_sheet(ws2, f"MARA TRIANGLE (MT) DAILY VIOLATIONS & MAX SPEEDS - {period_label.upper()}", mt_data["daily_df"])

    ws3 = wb.create_sheet("GM Camera Summary")
    write_df_sheet(ws3, f"GREATER MARA (GM) SPEED TRACKER SUMMARY - {period_label.upper()}", gm_data["cam_summary"])

    ws4 = wb.create_sheet("GM Daily Breakdown")
    write_df_sheet(ws4, f"GREATER MARA (GM) DAILY VIOLATIONS & MAX SPEEDS - {period_label.upper()}", gm_data["daily_df"])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("🚦 Mara Triangle (MT) & Greater Mara (GM) — Speed Tracker Analytics")
st.markdown(
    "Upload all **5 SafePace Speed Camera CSV/Excel exports** at once (`4KM`, `Olare`, `ODM`, `Oloolaimutia`, `Sekenani-Keekorok`). "
    "The dashboard automatically separates **MT (Mara Triangle)** and **GM (Greater Mara)**, cleans `H` peak markers, and generates report charts."
)

with st.sidebar:
    st.header("Upload Speed Camera Exports")
    uploaded_files = st.file_uploader(
        "Select all 5 camera files (CSV or XLSX)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

if uploaded_files:
    parsed_records = [parse_safepace_file(f) for f in uploaded_files]
    period_detected = parsed_records[0]["period"]

    mt_records = [r for r in parsed_records if r["region"] == "MT"]
    gm_records = [r for r in parsed_records if r["region"] == "GM"]

    mt_agg = aggregate_region(mt_records, ["4KM", "Olare"])
    gm_agg = aggregate_region(gm_records, ["ODM", "Keekorok-Oloolaimutia", "Keekorok-Sekenani"])

    excel_bytes = build_speed_excel(mt_agg, gm_agg, period_detected)
    st.download_button(
        label="📥 Download Combined MT & GM Speed Report (.xlsx)",
        data=excel_bytes,
        file_name=f"Speed_Tracker_Report_{period_detected.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    tab_mt, tab_gm = st.tabs(["🦁 Mara Triangle (MT): 4KM & Olare", "🌍 Greater Mara (GM): ODM, Oloolaimutia & Sekenani"])

    # --------------------------------------------------------------------------
    # TAB 1: MARA TRIANGLE (MT)
    # --------------------------------------------------------------------------
    with tab_mt:
        st.subheader(f"MT Speed Tracker Report ({period_detected})")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Vehicles Scanned (MT)", f"{mt_agg['total_vehicles']:,}")
        k2.metric("Speed Violations (>50 km/h)", f"{mt_agg['total_violations']:,}")
        k3.metric("Speed Limit Compliance", f"{mt_agg['compliance_pct']:.1f}%")
        k4.metric("Highest Speed Recorded", f"{mt_agg['peak_speed']} km/h ({mt_agg['peak_camera']})", delta=f"Date: {mt_agg['peak_date']}", delta_color="off")

        st.info(
            f"**Auto-Generated Narrative:** In this period ({period_detected}), a total of **{mt_agg['total_violations']:,} speed violations** "
            f"were recorded out of **{mt_agg['total_vehicles']:,}** vehicles scanned (**{round(mt_agg['compliance_pct'])}%** observed the 50 km/h speed limit). "
            f"The highest speed recorded was **{mt_agg['peak_speed']} km/h** on **{mt_agg['peak_date']}** at **{mt_agg['peak_camera']}**."
        )

        st.dataframe(mt_agg["cam_summary"], use_container_width=True, hide_index=True)

        df_mt = mt_agg["daily_df"]
        if not df_mt.empty:
            fig_mt1, ax_mt1 = plt.subplots(figsize=(11, 4))
            bars = ax_mt1.bar(df_mt["Date"], df_mt["Total Daily Violations"], color="#5c5346")
            ax_mt1.set_title("MT Number of Daily Speed Violations", fontweight="bold")
            ax_mt1.set_ylabel("Violations")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            ax_mt1.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars:
                h = bar.get_height()
                ax_mt1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{int(h)}", ha="center", va="bottom", fontsize=7)
            st.pyplot(fig_mt1)

            fig_mt2, ax_mt2 = plt.subplots(figsize=(11, 4.2))
            x = np.arange(len(df_mt["Date"]))
            w = 0.38
            b1 = ax_mt2.bar(x - w / 2, df_mt["4KM Max Speed (km/h)"], width=w, label="4KM", color="#c00000")
            b2 = ax_mt2.bar(x + w / 2, df_mt["Olare Max Speed (km/h)"], width=w, label="Olare", color="#00b050")
            ax_mt2.set_title("MT Maximum Recorded Daily Speed (km/h)", fontweight="bold")
            ax_mt2.set_ylabel("Max Speed (km/h)")
            ax_mt2.set_xticks(x)
            ax_mt2.set_xticklabels(df_mt["Date"], rotation=45, ha="right", fontsize=8)
            ax_mt2.legend()
            ax_mt2.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in b1:
                h = bar.get_height()
                if h > 0:
                    ax_mt2.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}", ha="center", va="bottom", fontsize=6.5, color="#c00000")
            for bar in b2:
                h = bar.get_height()
                if h > 0:
                    ax_mt2.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}", ha="center", va="bottom", fontsize=6.5, color="#008000")
            st.pyplot(fig_mt2)

            with st.expander("View Daily MT Speed Table"):
                st.dataframe(df_mt, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: GREATER MARA (GM)
    # --------------------------------------------------------------------------
    with tab_gm:
        st.subheader(f"GM Speed Tracker Report ({period_detected})")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Vehicles Scanned (GM)", f"{gm_agg['total_vehicles']:,}")
        g2.metric("Speed Violations (>50 km/h)", f"{gm_agg['total_violations']:,}")
        g3.metric("Speed Limit Compliance", f"{gm_agg['compliance_pct']:.1f}%")
        g4.metric("Highest Speed Recorded", f"{gm_agg['peak_speed']} km/h ({gm_agg['peak_camera']})", delta=f"Date: {gm_agg['peak_date']}", delta_color="off")

        st.info(
            f"**Auto-Generated Narrative:** In this period ({period_detected}), a total of **{gm_agg['total_violations']:,} speed violations** "
            f"were recorded out of **{gm_agg['total_vehicles']:,}** vehicles scanned (**{round(gm_agg['compliance_pct'])}%** observed the 50 km/h speed limit). "
            f"The highest speed recorded was **{gm_agg['peak_speed']} km/h** on **{gm_agg['peak_date']}** at **{gm_agg['peak_camera']}**."
        )

        st.dataframe(gm_agg["cam_summary"], use_container_width=True, hide_index=True)

        df_gm = gm_agg["daily_df"]
        if not df_gm.empty:
            fig_gm1, ax_gm1 = plt.subplots(figsize=(11, 4))
            bars_g = ax_gm1.bar(df_gm["Date"], df_gm["Total Daily Violations"], color="#5c5346")
            ax_gm1.set_title("GM Daily Number of Speed Violations", fontweight="bold")
            ax_gm1.set_ylabel("Violations")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            ax_gm1.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars_g:
                h = bar.get_height()
                ax_gm1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontsize=7)
            st.pyplot(fig_gm1)

            fig_gm2, ax_gm2 = plt.subplots(figsize=(12, 4.5))
            x = np.arange(len(df_gm["Date"]))
            w = 0.27
            b_odm = ax_gm2.bar(x - w, df_gm["ODM Max Speed (km/h)"], width=w, label="ODM", color="#1f4e78")
            b_olo = ax_gm2.bar(x, df_gm["Keekorok-Oloolaimutia Max Speed (km/h)"], width=w, label="Keekorok-Oloolaimutia", color="#ed7d31")
            b_sek = ax_gm2.bar(x + w, df_gm["Keekorok-Sekenani Max Speed (km/h)"], width=w, label="Keekorok-Sekenani", color="#38761d")
            ax_gm2.set_title("GM Maximum Recorded Daily Speed (km/h)", fontweight="bold")
            ax_gm2.set_ylabel("Max Speed (km/h)")
            ax_gm2.set_xticks(x)
            ax_gm2.set_xticklabels(df_gm["Date"], rotation=45, ha="right", fontsize=8)
            ax_gm2.legend()
            ax_gm2.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig_gm2)

            with st.expander("View Daily GM Speed Table"):
                st.dataframe(df_gm, use_container_width=True, hide_index=True)
else:
    st.info("👈 Upload your 5 SafePace speed camera files in the sidebar (`4KM`, `Olare`, `ODM`, `Oloolaimutia`, `Sekenani-Keekorok`).")
