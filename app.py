import ast
import io
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st

st.set_page_config(page_title="Mara Triangle Monthly Analytics", layout="wide")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TICKET_TYPES = [
    "PayingGuest", "ncg", "parktrasnsit", "kifaru", "complimentary",
    "visitors", "littleexplorers", "serenaticket", "LG", "angama", "research"
]

TICKET_LABELS = {
    "PayingGuest": "Paying Guest",
    "ncg": "NCG Tickets",
    "parktrasnsit": "Park Transit",
    "kifaru": "Kifaru",
    "complimentary": "Complementary",
    "visitors": "Visitors",
    "littleexplorers": "Little Explorers (LE)",
    "serenaticket": "Serena Ticket",
    "LG": "LG Ticket",
    "angama": "Angama Ticket",
    "research": "Research"
}

DEMO_KEYS = ["nra", "na2", "ca2", "nrc", "raA", "cs2", "cc2", "ns2", "rc2", "nc2", "le2"]

DEMO_LABELS = {
    "nra": "NRA (Non-Resident Adult)",
    "na2": "NA (Non-Resident)",
    "ca2": "CA (Citizen Adult)",
    "nrc": "NRC (Non-Resident Child)",
    "raA": "RA (Resident Adult)",
    "cs2": "CS (Citizen Student)",
    "cc2": "CC (Citizen Child)",
    "ns2": "NS (Non-Resident Special)",
    "rc2": "RC (Resident Child)",
    "nc2": "NC (Citizen Child Special)",
    "le2": "LE (Little Explorers Special)"
}

# ==============================================================================
# EXCEL BUILDER FUNCTION (IN-MEMORY)
# ==============================================================================
def build_excel_bytes(ticket_totals, guest_matrix, veh_station, report_label):
    wb = openpyxl.Workbook()
    title_font = Font(name="Calibri", size=15, bold=True, color="1F4E78")
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

    def write_sheet(ws, title, df_to_write, include_index=False):
        ws["A1"] = title
        ws["A1"].font = title_font

        if include_index:
            df_export = df_to_write.reset_index()
            df_export.rename(columns={"index": "Category"}, inplace=True)
        else:
            df_export = df_to_write.copy()

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
                    cell.number_format = "0.00" if "Share" in str(df_export.columns[c_idx - 1]) else "#,##0"
                    cell.alignment = Alignment(horizontal="right")

        tot_row = len(df_export) + 4
        ws.cell(row=tot_row, column=1, value="TOTAL").font = bold_font
        ws.cell(row=tot_row, column=1).fill = total_fill
        ws.cell(row=tot_row, column=1).border = total_border

        for c_idx in range(2, len(df_export.columns) + 1):
            col_let = get_column_letter(c_idx)
            cell = ws.cell(row=tot_row, column=c_idx, value=f"=SUM({col_let}4:{col_let}{tot_row - 1})")
            cell.font = bold_font
            cell.fill = total_fill
            cell.border = total_border
            cell.number_format = "0.00" if "Share" in str(df_export.columns[c_idx - 1]) else "#,##0"
            cell.alignment = Alignment(horizontal="right")

        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 13)

    ws1 = wb.active
    ws1.title = "Ticket Type Totals"
    write_sheet(ws1, f"MARA TRIANGLE ({report_label.upper()}) - GUEST TOTALS BY TICKET TYPE", ticket_totals, include_index=False)

    ws2 = wb.create_sheet("Guest Demographic Matrix")
    write_sheet(ws2, f"MARA TRIANGLE ({report_label.upper()}) - DEMOGRAPHIC BREAKDOWN", guest_matrix, include_index=True)

    ws3 = wb.create_sheet("Vehicles Per Station")
    write_sheet(ws3, f"MARA TRIANGLE ({report_label.upper()}) - VEHICLE TRAFFIC PER STATION", veh_station, include_index=True)

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

# ==============================================================================
# WEB INTERFACE UI
# ==============================================================================
st.title("🦁 Mara Triangle — Monthly Guest & Vehicle Dashboard")
st.markdown("Upload a monthly CSV export to generate standalone guest breakdowns, vehicle station reports, charts, and a formatted Excel workbook.")

with st.sidebar:
    st.header("1. Upload Monthly Data")
    uploaded_file = st.file_uploader("Choose your Guest Data CSV file", type=["csv"])
    report_period = st.text_input("Report Month / Label", value="October 2026")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Parse Ticket_Sales
    matrix = {dk: {tt: 0 for tt in TICKET_TYPES} for dk in DEMO_KEYS}
    for val in df["Ticket_Sales"].dropna():
        for entry in str(val).split(";"):
            entry = entry.strip()
            if not entry:
                continue
            try:
                record = ast.literal_eval(entry)
                tt = record.get("ticket_type", "").strip()
                if tt in TICKET_TYPES:
                    for k, v in record.items():
                        if k in DEMO_KEYS:
                            if isinstance(v, (int, float)):
                                matrix[k][tt] += int(v)
                            elif isinstance(v, str) and v.isdigit():
                                matrix[k][tt] += int(v)
            except Exception:
                pass

    guest_matrix = pd.DataFrame(matrix).T
    guest_matrix.index = [DEMO_LABELS[k] for k in guest_matrix.index]
    guest_matrix.columns = [TICKET_LABELS[c] for c in guest_matrix.columns]
    guest_matrix["Total Guests"] = guest_matrix.sum(axis=1)

    ticket_totals = guest_matrix.drop(columns=["Total Guests"]).sum(axis=0).reset_index()
    ticket_totals.columns = ["Ticket Type", "Total Guests"]
    ticket_totals = ticket_totals.sort_values(by="Total Guests", ascending=False).reset_index(drop=True)
    grand_total_guests = int(ticket_totals["Total Guests"].sum())
    ticket_totals["Share (%)"] = (ticket_totals["Total Guests"] / grand_total_guests * 100).round(2)

    # Parse Vehicles per Station
    df["Station_Clean"] = df["Station"].fillna("Unknown").astype(str).str.strip()
    veh_cols = [
        "Number_of_Paying_Vehicles",
        "Number_of_Vehicles_with_Anual_Stickers",
        "Number_of_Non_Paying_vehicles"
    ]
    for col in veh_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    veh_station = df.groupby("Station_Clean")[veh_cols].sum()
    veh_station.columns = ["Paying Vehicles", "AVS Vehicles", "Non-Paying Vehicles"]
    veh_station["Total Vehicles"] = veh_station.sum(axis=1)
    veh_station = veh_station.sort_values(by="Total Vehicles", ascending=False)
    grand_total_vehicles = int(veh_station["Total Vehicles"].sum())
    veh_station["Share (%)"] = (veh_station["Total Vehicles"] / grand_total_vehicles * 100).round(2)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Recorded Guests", f"{grand_total_guests:,}")
    col2.metric("Paying Guests", f"{int(guest_matrix['Paying Guest'].sum()):,}")
    col3.metric("Total Vehicles", f"{grand_total_vehicles:,}")
    col4.metric("Top Station", veh_station.index[0])

    # Excel Download Button
    excel_data = build_excel_bytes(ticket_totals, guest_matrix, veh_station, report_period)
    st.download_button(
        label="📥 Download Formatted Excel Workbook (.xlsx)",
        data=excel_data,
        file_name=f"Mara_Triangle_{report_period.replace(' ', '_')}_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    # Section 1: Guests by Ticket Type
    st.subheader(f"1. Guest Distribution by Ticket Type ({report_period})")
    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        st.dataframe(ticket_totals, use_container_width=True, hide_index=True)
    with c_right:
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        bars = ax1.bar(ticket_totals["Ticket Type"], ticket_totals["Total Guests"], color="#1F4E78")
        ax1.set_ylabel("Guests")
        ax1.set_title(f"Guests per Ticket Type — {report_period}", fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval + (grand_total_guests * 0.008), f"{int(yval):,}", ha="center", fontsize=8)
        st.pyplot(fig1)

    # Section 2: Full Demographic Matrix
    st.subheader(f"2. Full Demographic Breakdown Matrix ({report_period})")
    st.dataframe(guest_matrix, use_container_width=True)

    # Section 3: Vehicles per Station
    st.subheader(f"3. Vehicle Traffic per Station ({report_period})")
    v_left, v_right = st.columns([1.1, 1.3])
    with v_left:
        st.dataframe(veh_station, use_container_width=True)
    with v_right:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        veh_station[["Paying Vehicles", "AVS Vehicles", "Non-Paying Vehicles"]].plot(
            kind="bar", stacked=True, ax=ax2, color=["#2b5c8f", "#d95f02", "#7570b3"]
        )
        ax2.set_ylabel("Vehicles")
        ax2.set_xlabel("Station")
        ax2.set_title(f"Vehicles per Station — {report_period}", fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig2)
else:
    st.info("👈 Upload your monthly CSV file in the sidebar to generate the report.")