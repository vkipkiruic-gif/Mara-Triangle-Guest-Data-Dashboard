import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st

# ==============================================================================
# BALLOON DATA PARSER (HANDLES MULTI-HEADER EXCEL / CSV EXPORTS)
# ==============================================================================
def parse_balloon_file(uploaded_file):
    """
    Parses the monthly Balloon Data CSV or Excel file containing daily passenger counts
    across Skyship Balloon, Governors Balloon (LGB), and Transworld Balloon
    (broken down by Paying Adults, Paying Children, Complementary Adults, Complementary Children).
    Automatically locates the daily table and ignores trailing summary/archive blocks below Grand Total.
    """
    fname = uploaded_file.name
    if fname.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded_file, header=None)
    else:
        raw = pd.read_csv(uploaded_file, header=None)

    # Locate the row containing 'SKYSHIP' (company header row)
    comp_row_idx = None
    for idx in range(min(15, len(raw))):
        row_str = " ".join(str(x).upper() for x in raw.iloc[idx].tolist() if pd.notna(x))
        if "SKYSHIP" in row_str and ("GOVERNOR" in row_str or "TRANSWORLD" in row_str):
            comp_row_idx = idx
            break

    if comp_row_idx is None:
        raise ValueError("Could not detect the company header row (SKYSHIP / GOVERNORS / TRANSWORLD) in the uploaded file.")

    # Locate the 'ADULTS' / 'CHILDREN' sub-header row
    sub_row_idx = comp_row_idx + 2
    for idx in range(comp_row_idx + 1, min(comp_row_idx + 5, len(raw))):
        row_str = " ".join(str(x).upper() for x in raw.iloc[idx].tolist() if pd.notna(x))
        if "ADULTS" in row_str and "CHILDREN" in row_str:
            sub_row_idx = idx
            break

    # Detect the Date column (first non-empty cell in the first data row)
    data_start_idx = sub_row_idx + 1
    date_col_idx = 1
    for c_idx in range(raw.shape[1]):
        val = raw.iloc[data_start_idx, c_idx]
        if pd.notna(val) and str(val).strip() != "":
            date_col_idx = c_idx
            break

    # Extract daily rows up to 'Grand Total' or blank separator
    daily_records = []
    for r_idx in range(data_start_idx, len(raw)):
        date_val = raw.iloc[r_idx, date_col_idx]
        if pd.isna(date_val):
            break
        date_str = str(date_val).strip()
        if not date_str or "GRAND TOTAL" in date_str.upper() or "TOTAL" == date_str.upper():
            break

        def get_int(offset):
            col_i = date_col_idx + offset
            if col_i >= raw.shape[1]:
                return 0
            v = raw.iloc[r_idx, col_i]
            if pd.isna(v):
                return 0
            num = pd.to_numeric(str(v).replace(",", "").strip(), errors="coerce")
            return int(num) if pd.notna(num) else 0

        s_pa, s_pc, s_ca, s_cc = get_int(1), get_int(2), get_int(3), get_int(4)
        g_pa, g_pc, g_ca, g_cc = get_int(5), get_int(6), get_int(7), get_int(8)
        t_pa, t_pc, t_ca, t_cc = get_int(9), get_int(10), get_int(11), get_int(12)

        daily_records.append({
            "Date": date_str,
            "Skyship Paying Adults": s_pa,
            "Skyship Paying Children": s_pc,
            "Skyship Comp Adults": s_ca,
            "Skyship Comp Children": s_cc,
            "Governors Paying Adults": g_pa,
            "Governors Paying Children": g_pc,
            "Governors Comp Adults": g_ca,
            "Governors Comp Children": g_cc,
            "Transworld Paying Adults": t_pa,
            "Transworld Paying Children": t_pc,
            "Transworld Comp Adults": t_ca,
            "Transworld Comp Children": t_cc,
            "Skyship Total": s_pa + s_pc + s_ca + s_cc,
            "Governors Total": g_pa + g_pc + g_ca + g_cc,
            "Transworld Total": t_pa + t_pc + t_ca + t_cc,
            "Daily Total Guests": (s_pa + s_pc + s_ca + s_cc) + (g_pa + g_pc + g_ca + g_cc) + (t_pa + t_pc + t_ca + t_cc)
        })

    daily_df = pd.DataFrame(daily_records)
    if daily_df.empty:
        raise ValueError("No daily balloon guest records were found in the uploaded file.")

    # Build Company Summary Table (Report Format)
    company_rows = [
        {
            "Balloon Company": "Governors Balloon (LGB)",
            "Paying Adults": int(daily_df["Governors Paying Adults"].sum()),
            "Paying Children": int(daily_df["Governors Paying Children"].sum()),
            "Complementary Adults": int(daily_df["Governors Comp Adults"].sum()),
            "Complementary Children": int(daily_df["Governors Comp Children"].sum()),
        },
        {
            "Balloon Company": "Skyship Balloon",
            "Paying Adults": int(daily_df["Skyship Paying Adults"].sum()),
            "Paying Children": int(daily_df["Skyship Paying Children"].sum()),
            "Complementary Adults": int(daily_df["Skyship Comp Adults"].sum()),
            "Complementary Children": int(daily_df["Skyship Comp Children"].sum()),
        },
        {
            "Balloon Company": "Transworld Balloon",
            "Paying Adults": int(daily_df["Transworld Paying Adults"].sum()),
            "Paying Children": int(daily_df["Transworld Paying Children"].sum()),
            "Complementary Adults": int(daily_df["Transworld Comp Adults"].sum()),
            "Complementary Children": int(daily_df["Transworld Comp Children"].sum()),
        }
    ]
    comp_df = pd.DataFrame(company_rows)
    comp_df["Total Paying"] = comp_df["Paying Adults"] + comp_df["Paying Children"]
    comp_df["Total Complementary"] = comp_df["Complementary Adults"] + comp_df["Complementary Children"]
    comp_df["Total Guests"] = comp_df["Total Paying"] + comp_df["Total Complementary"]
    grand_total = int(comp_df["Total Guests"].sum())
    comp_df["Share (%)"] = (comp_df["Total Guests"] / grand_total * 100).round(2) if grand_total > 0 else 0.0

    return daily_df, comp_df


# ==============================================================================
# EXCEL WORKBOOK BUILDER (IN-MEMORY)
# ==============================================================================
def build_balloon_excel(comp_df: pd.DataFrame, daily_df: pd.DataFrame, period_label: str):
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

    def write_sheet(ws, title, df_export):
        ws["A1"] = title
        ws["A1"].font = title_font

        for c_idx, col_name in enumerate(df_export.columns, 1):
            cell = ws.cell(row=3, column=c_idx, value=col_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center" if c_idx > 1 else "left", wrap_text=True)

        for r_idx, row in enumerate(df_export.itertuples(index=False), 4):
            for c_idx, val in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.font = regular_font
                cell.border = thin_border
                if c_idx > 1:
                    col_header = str(df_export.columns[c_idx - 1])
                    cell.number_format = "0.00" if "%" in col_header else "#,##0"
                    cell.alignment = Alignment(horizontal="right")

        if len(df_export) > 0:
            tot_row = len(df_export) + 4
            ws.cell(row=tot_row, column=1, value="GRAND TOTAL").font = bold_font
            ws.cell(row=tot_row, column=1).fill = total_fill
            ws.cell(row=tot_row, column=1).border = total_border

            for c_idx in range(2, len(df_export.columns) + 1):
                col_let = get_column_letter(c_idx)
                col_header = str(df_export.columns[c_idx - 1])
                cell = ws.cell(row=tot_row, column=c_idx, value=f"=SUM({col_let}4:{col_let}{tot_row - 1})")
                cell.font = bold_font
                cell.fill = total_fill
                cell.border = total_border
                cell.number_format = "0.00" if "%" in col_header else "#,##0"
                cell.alignment = Alignment(horizontal="right")

        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 13)

    ws1 = wb.active
    ws1.title = "Balloon Company Summary"
    write_sheet(ws1, f"MARA TRIANGLE BALLOON GUESTS BY COMPANY — {period_label.upper()}", comp_df)

    ws2 = wb.create_sheet("Daily Balloon Breakdown")
    write_sheet(ws2, f"MARA TRIANGLE DAILY BALLOON PASSENGERS — {period_label.upper()}", daily_df)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ==============================================================================
# STREAMLIT PAGE UI
# ==============================================================================
st.title("🎈 Hot Air Balloon Guest Reports")
st.markdown(
    "Upload the monthly **Balloon Data CSV or Excel file** (`Skyship Balloon`, `Governors Balloon`, `Transworld Balloon`) "
    "to generate company passenger breakdowns, paying vs. complementary guest totals, daily flight trends, and an Excel workbook."
)

with st.sidebar:
    st.subheader("Upload Balloon Data")
    uploaded_balloon = st.file_uploader(
        "Choose Balloon Data CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="balloon_file"
    )
    report_period = st.text_input("Report Month / Label", value="October 2026", key="balloon_period")

if uploaded_balloon is not None:
    try:
        daily_df, comp_df = parse_balloon_file(uploaded_balloon)

        total_guests = int(comp_df["Total Guests"].sum())
        total_paying = int(comp_df["Total Paying"].sum())
        total_comp = int(comp_df["Total Complementary"].sum())
        total_adults = int(comp_df["Paying Adults"].sum() + comp_df["Complementary Adults"].sum())
        total_children = int(comp_df["Paying Children"].sum() + comp_df["Complementary Children"].sum())

        top_company = comp_df.loc[comp_df["Total Guests"].idxmax()]
        peak_day_row = daily_df.loc[daily_df["Daily Total Guests"].idxmax()]
        active_days = int((daily_df["Daily Total Guests"] > 0).sum())

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Balloon Guests", f"{total_guests:,}", delta=f"{total_paying:,} Paying | {total_comp:,} Comp", delta_color="off")
        k2.metric("Leading Operator", top_company["Balloon Company"].replace(" Balloon", ""), delta=f"{int(top_company['Total Guests']):,} guests ({top_company['Share (%)']}%)", delta_color="off")
        k3.metric("Peak Flight Day", f"{int(peak_day_row['Daily Total Guests']):,} guests", delta=f"Date: {peak_day_row['Date']}", delta_color="off")
        k4.metric("Adults vs. Children", f"{total_adults:,} / {total_children:,}", delta=f"{active_days} active flight days", delta_color="off")

        # Auto-Generated Narrative for Mapping & Arrivals Report
        gov_row = comp_df[comp_df["Balloon Company"].str.contains("Governors")].iloc[0]
        sky_row = comp_df[comp_df["Balloon Company"].str.contains("Skyship")].iloc[0]
        trn_row = comp_df[comp_df["Balloon Company"].str.contains("Transworld")].iloc[0]

        st.info(
            f"**Auto-Generated Balloon Narrative ({report_period}):** A total of **{total_guests:,} balloon guests** "
            f"(**{total_paying:,} paying** and **{total_comp:,} complementary**; **{total_adults:,} adults** and **{total_children:,} children**) "
            f"were recorded across **{active_days} active flight days**. **Governors Balloon (LGB)** recorded the highest share with "
            f"**{int(gov_row['Total Guests']):,} guests ({gov_row['Share (%)']}%)**, followed by **Skyship Balloon** with "
            f"**{int(sky_row['Total Guests']):,} guests ({sky_row['Share (%)']}%)**, and **Transworld Balloon** with "
            f"**{int(trn_row['Total Guests']):,} guests ({trn_row['Share (%)']}%)**. Peak daily volume occurred on **{peak_day_row['Date']}** "
            f"with **{int(peak_day_row['Daily Total Guests']):,} guests**."
        )

        # Excel Download Button
        excel_bytes = build_balloon_excel(comp_df, daily_df, report_period)
        st.download_button(
            label="📥 Download Balloon Guest Report (.xlsx)",
            data=excel_bytes,
            file_name=f"Mara_Triangle_Balloon_Report_{report_period.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        # Section 1: Company Summary Table & Bar Chart
        st.subheader(f"1. Balloon Guests by Company & Category ({report_period})")
        c_left, c_right = st.columns([1.35, 1])
        with c_left:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        with c_right:
            fig1, ax1 = plt.subplots(figsize=(7.5, 4.3))
            companies = comp_df["Balloon Company"]
            x = np.arange(len(companies))
            w = 0.36
            b1 = ax1.bar(x - w / 2, comp_df["Paying Adults"] + comp_df["Complementary Adults"], width=w, label="Adults", color="#1F4E78")
            b2 = ax1.bar(x + w / 2, comp_df["Paying Children"] + comp_df["Complementary Children"], width=w, label="Children", color="#ED7D31")
            ax1.set_xticks(x)
            ax1.set_xticklabels(companies, rotation=12, ha="right")
            ax1.set_ylabel("Number of Guests")
            ax1.set_title(f"Balloon Guests by Operator — {report_period}", fontweight="bold")
            ax1.legend()
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in b1:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 12, f"{int(h):,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            for bar in b2:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 12, f"{int(h):,}", ha="center", va="bottom", fontsize=8.5)
            st.pyplot(fig1)

        # Section 2: Daily Stacked Bar Chart
        st.subheader(f"2. Daily Balloon Passenger Volume by Operator ({report_period})")
        fig2, ax2 = plt.subplots(figsize=(11.5, 4.3))
        dates = daily_df["Date"]
        ax2.bar(dates, daily_df["Governors Total"], label="Governors Balloon (LGB)", color="#1F4E78")
        ax2.bar(dates, daily_df["Skyship Total"], bottom=daily_df["Governors Total"], label="Skyship Balloon", color="#2E8B57")
        ax2.bar(
            dates,
            daily_df["Transworld Total"],
            bottom=daily_df["Governors Total"] + daily_df["Skyship Total"],
            label="Transworld Balloon",
            color="#D95F02"
        )
        ax2.set_ylabel("Total Guests")
        ax2.set_title(f"Daily Balloon Guests — {report_period}", fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8.5)
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig2)

        # Section 3: Full Daily Breakdown Table
        st.subheader(f"3. Daily Balloon Flight & Guest Breakdown ({report_period})")
        st.dataframe(daily_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error parsing Balloon Data file: {e}")
else:
    st.info("👈 Upload your monthly Balloon Data CSV or Excel file (`BALLOON DATA - OCTOBER 2026.csv`) in the sidebar.")
