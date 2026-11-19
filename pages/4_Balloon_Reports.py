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

    # Detect the Date column
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
            "Governors Paying Adults": g_pa,
            "Governors Paying Children": g_pc,
            "Governors Comp Adults": g_ca,
            "Governors Comp Children": g_cc,
            "Skyship Paying Adults": s_pa,
            "Skyship Paying Children": s_pc,
            "Skyship Comp Adults": s_ca,
            "Skyship Comp Children": s_cc,
            "Transworld Paying Adults": t_pa,
            "Transworld Paying Children": t_pc,
            "Transworld Comp Adults": t_ca,
            "Transworld Comp Children": t_cc,
            "Governors Total": g_pa + g_pc + g_ca + g_cc,
            "Skyship Total": s_pa + s_pc + s_ca + s_cc,
            "Transworld Total": t_pa + t_pc + t_ca + t_cc,
            "Daily Total Guests": (g_pa + g_pc + g_ca + g_cc) + (s_pa + s_pc + s_ca + s_cc) + (t_pa + t_pc + t_ca + t_cc)
        })

    daily_df = pd.DataFrame(daily_records)
    if daily_df.empty:
        raise ValueError("No daily balloon guest records were found in the uploaded file.")

    # Build Company Summary Table ordered: Governors, Skyship, Transworld
    company_rows = [
        {
            "Company": "Governors",
            "ADULT PAYING GUEST": int(daily_df["Governors Paying Adults"].sum()),
            "CHILD PAYING GUEST": int(daily_df["Governors Paying Children"].sum()),
            "ADULT COMPLIMENTARY": int(daily_df["Governors Comp Adults"].sum()),
            "CHILD COMPLIMENTARY": int(daily_df["Governors Comp Children"].sum()),
        },
        {
            "Company": "Skyship",
            "ADULT PAYING GUEST": int(daily_df["Skyship Paying Adults"].sum()),
            "CHILD PAYING GUEST": int(daily_df["Skyship Paying Children"].sum()),
            "ADULT COMPLIMENTARY": int(daily_df["Skyship Comp Adults"].sum()),
            "CHILD COMPLIMENTARY": int(daily_df["Skyship Comp Children"].sum()),
        },
        {
            "Company": "Transworld",
            "ADULT PAYING GUEST": int(daily_df["Transworld Paying Adults"].sum()),
            "CHILD PAYING GUEST": int(daily_df["Transworld Paying Children"].sum()),
            "ADULT COMPLIMENTARY": int(daily_df["Transworld Comp Adults"].sum()),
            "CHILD COMPLIMENTARY": int(daily_df["Transworld Comp Children"].sum()),
        }
    ]
    comp_df = pd.DataFrame(company_rows)
    comp_df["Total Paying"] = comp_df["ADULT PAYING GUEST"] + comp_df["CHILD PAYING GUEST"]
    comp_df["Total Complimentary"] = comp_df["ADULT COMPLIMENTARY"] + comp_df["CHILD COMPLIMENTARY"]
    comp_df["Total Guests"] = comp_df["Total Paying"] + comp_df["Total Complimentary"]
    grand_total = int(comp_df["Total Guests"].sum())
    comp_df["Overall Share (%)"] = (comp_df["Total Guests"] / grand_total * 100).round(2) if grand_total > 0 else 0.0

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
    ws1.title = "Balloon Guests Summary"
    write_sheet(ws1, f"MARA TRIANGLE BALLOON GUESTS — {period_label.upper()}", comp_df)

    ws2 = wb.create_sheet("Daily Balloon Breakdown")
    write_sheet(ws2, f"MARA TRIANGLE DAILY BALLOON PASSENGERS — {period_label.upper()}", daily_df)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ==============================================================================
# STREAMLIT PAGE UI (EXACT REPORT FORMAT & CHART DESIGN)
# ==============================================================================
st.markdown("<h1 style='color:#9E8C2E; margin-bottom:0;'>BALLOON GUESTS</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Upload Balloon Data")
    uploaded_balloon = st.file_uploader(
        "Choose Balloon Data CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="balloon_file"
    )
    report_period = st.text_input("Report Month / Label", value="October 2026", key="balloon_period")
    prev_month_paying = st.number_input(
        "Last Month's Paying Guests (for comparison text)",
        min_value=0,
        value=2253,
        step=1,
        key="prev_balloon_paying"
    )

if uploaded_balloon is not None:
    try:
        daily_df, comp_df = parse_balloon_file(uploaded_balloon)

        total_paying = int(comp_df["Total Paying"].sum())
        total_comp = int(comp_df["Total Complimentary"].sum())
        total_guests = int(comp_df["Total Guests"].sum())

        top_row = comp_df.loc[comp_df["Total Guests"].idxmax()]
        top_company_name = top_row["Company"]
        top_company_guests = int(top_row["Total Guests"])

        if prev_month_paying > 0:
            if total_paying < prev_month_paying:
                trend_word = "a decrease"
            elif total_paying > prev_month_paying:
                trend_word = "an increase"
            else:
                trend_word = "unchanged"
            comparison_sentence = f"This is {trend_word} from last month's <b>{prev_month_paying:,}</b> paying Guests. "
        else:
            comparison_sentence = ""

        # Exact Report Narrative Header
        st.markdown(
            f"<p style='font-size:18px; line-height:1.5;'>"
            f"The total number of balloons paying guests this month was <b>{total_paying:,}</b>. "
            f"{comparison_sentence}"
            f"{top_company_name} had the highest number of guests having recorded <b>{top_company_guests:,}</b> guests."
            f"</p>",
            unsafe_allow_html=True
        )

        # KPI Strip + Excel Download
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Paying Guests", f"{total_paying:,}")
        k2.metric("Total Complimentary", f"{total_comp:,}")
        k3.metric("Total All Guests", f"{total_guests:,}")
        k4.metric(f"{top_company_name} (Highest)", f"{top_company_guests:,} guests")

        excel_bytes = build_balloon_excel(comp_df, daily_df, report_period)
        st.download_button(
            label="📥 Download Balloon Guests Report (.xlsx)",
            data=excel_bytes,
            file_name=f"Mara_Triangle_Balloon_Guests_{report_period.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        # ======================================================================
        # REPORT CHART: GROUPED PERCENTAGE BAR CHART BY GUEST CATEGORY
        # ======================================================================
        categories = [
            "ADULT PAYING GUEST",
            "CHILD PAYING GUEST",
            "ADULT COMPLIMENTARY",
            "CHILD COMPLIMENTARY"
        ]

        pct_matrix = {}
        for cat in categories:
            col_sum = comp_df[cat].sum()
            if col_sum > 0:
                pct_matrix[cat] = (comp_df[cat] / col_sum * 100).values
            else:
                pct_matrix[cat] = np.zeros(len(comp_df))

        gov_pcts = [pct_matrix[cat][0] for cat in categories]
        sky_pcts = [pct_matrix[cat][1] for cat in categories]
        trn_pcts = [pct_matrix[cat][2] for cat in categories]

        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        x = np.arange(len(categories))
        bar_width = 0.14
        gap = 0.08

        b_gov = ax.bar(x - (bar_width + gap), gov_pcts, width=bar_width, color="#00A651", label="Governors", zorder=3)
        b_sky = ax.bar(x, sky_pcts, width=bar_width, color="#FFF200", label="Skyship", zorder=3)
        b_trn = ax.bar(x + (bar_width + gap), trn_pcts, width=bar_width, color="#00AEEF", label="Transworld", zorder=3)

        for sep_x in [-0.5, 0.5, 1.5, 2.5, 3.5]:
            ax.axvline(sep_x, color="#EAEAEA", linewidth=1.0, zorder=1)

        for bars in [b_gov, b_sky, b_trn]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 2.0,
                        f"{round(h):.0f}%",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=10,
                        fontweight="bold",
                        color="#008037"
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold", color="#00A651")
        ax.set_ylim(0, 95)
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#EAEAEA")
        ax.tick_params(axis="x", length=0, pad=10)

        leg = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=3,
            frameon=False,
            handlelength=0.9,
            handleheight=0.9,
            fontsize=10.5
        )
        for text in leg.get_texts():
            text.set_color("#00A651")
            text.set_fontweight("bold")

        plt.tight_layout()
        st.pyplot(fig)

        # Supporting Data Tables
        st.subheader(f"Balloon Guests Summary Table ({report_period})")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with st.expander("View Daily Balloon Passenger Breakdown"):
            st.dataframe(daily_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error parsing Balloon Data file: {e}")
else:
    st.info("👈 Upload your monthly Balloon Data CSV or Excel file (`BALLOON DATA - OCTOBER 2026.csv`) in the sidebar.")
