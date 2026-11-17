import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ==============================================================================
# CONFIGURATION & OFFICIAL GUEST TYPE MAPPINGS
# ==============================================================================
INPUT_CSV = "Guest Data.csv"
OUTPUT_EXCEL = "Mara_Triangle_October_2026_Report.xlsx"
GUEST_CHART_PNG = "october_guests_by_ticket_type.png"
VEHICLE_CHART_PNG = "october_vehicles_per_station.png"

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

GUEST_TYPE_ORDER = [
    "NRA", "NRC", "NRS",
    "RA", "RC", "RS",
    "CA", "CC", "CS",
    "NA", "NC", "NS",
    "LE"
]

GUEST_TYPE_LABELS = {
    "NRA": "NRA - Non Resident Adults",
    "NRC": "NRC - Non Resident Children",
    "NRS": "NRS - Non Resident Students",
    "RA": "RA - Resident Adults",
    "RC": "RC - Resident Children",
    "RS": "RS - Resident Students",
    "CA": "CA - Citizen Adults",
    "CC": "CC - Citizen Children",
    "CS": "CS - Citizen Students",
    "NA": "NA - Narok Adults",
    "NC": "NC - Narok Children",
    "NS": "NS - Narok Students",
    "LE": "LE - Little Explorers"
}

RAW_KEY_TO_CODE = {
    "nra": "NRA", "nra2": "NRA",
    "nrc": "NRC", "nrc2": "NRC",
    "nrs": "NRS", "nrs2": "NRS",
    "raa": "RA",  "ra": "RA",   "ra2": "RA",
    "rc2": "RC",  "rc": "RC",
    "rs2": "RS",  "rs": "RS",   "rsa": "RS",
    "ca2": "CA",  "ca": "CA",
    "cc2": "CC",  "cc": "CC",
    "cs2": "CS",  "cs": "CS",
    "na2": "NA",  "na": "NA",
    "nc2": "NC",  "nc": "NC",
    "ns2": "NS",  "ns": "NS",
    "le2": "LE",  "le": "LE"
}

# ==============================================================================
# 1. LOAD & PARSE DATA
# ==============================================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Could not find '{INPUT_CSV}' in the current directory.")

df = pd.read_csv(INPUT_CSV)

matrix = {code: {tt: 0 for tt in TICKET_TYPES} for code in GUEST_TYPE_ORDER}

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
                    if k == "ticket_type":
                        continue
                    mapped_code = RAW_KEY_TO_CODE.get(str(k).strip().lower())
                    if mapped_code in matrix:
                        if isinstance(v, (int, float)):
                            matrix[mapped_code][tt] += int(v)
                        elif isinstance(v, str) and v.strip().isdigit():
                            matrix[mapped_code][tt] += int(v.strip())
        except Exception as e:
            print(f"Skipping malformed entry: {entry} ({e})")

guest_matrix = pd.DataFrame(matrix).T
guest_matrix.index = [GUEST_TYPE_LABELS[code] for code in guest_matrix.index]
guest_matrix.columns = [TICKET_LABELS[c] for c in guest_matrix.columns]
guest_matrix["Total Guests"] = guest_matrix.sum(axis=1)

ticket_totals = guest_matrix.drop(columns=["Total Guests"]).sum(axis=0).reset_index()
ticket_totals.columns = ["Ticket Type", "Total Guests"]
ticket_totals = ticket_totals.sort_values(by="Total Guests", ascending=False).reset_index(drop=True)
grand_total_guests = ticket_totals["Total Guests"].sum()
ticket_totals["Share (%)"] = (ticket_totals["Total Guests"] / grand_total_guests * 100).round(2)

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
grand_total_vehicles = veh_station["Total Vehicles"].sum()
veh_station["Share (%)"] = (veh_station["Total Vehicles"] / grand_total_vehicles * 100).round(2)

# ==============================================================================
# 2. PRINT TABLES TO CONSOLE
# ==============================================================================
print("\n" + "=" * 75)
print(f"TOTAL GUESTS BY TICKET TYPE (Grand Total: {grand_total_guests:,})")
print("=" * 75)
print(ticket_totals.to_string(index=False))

print("\n" + "=" * 75)
print("GUEST TYPE BREAKDOWN MATRIX")
print("=" * 75)
print(guest_matrix.to_string())

print("\n" + "=" * 75)
print(f"VEHICLE TRAFFIC PER STATION (Grand Total: {grand_total_vehicles:,})")
print("=" * 75)
print(veh_station.to_string())

# ==============================================================================
# 3. GENERATE CHARTS
# ==============================================================================
plt.figure(figsize=(11, 6))
bars = plt.bar(ticket_totals["Ticket Type"], ticket_totals["Total Guests"], color="#1F4E78")
plt.title("Total Guests per Ticket Type (October 2026)", fontsize=14, fontweight="bold")
plt.xlabel("Ticket Type", fontsize=11)
plt.ylabel("Number of Guests", fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 80, f"{int(yval):,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(GUEST_CHART_PNG, dpi=150)
plt.close()

ax = veh_station[["Paying Vehicles", "AVS Vehicles", "Non-Paying Vehicles"]].plot(
    kind="bar", stacked=True, figsize=(10, 6), color=["#2b5c8f", "#d95f02", "#7570b3"]
)
plt.title("Vehicle Count by Station (October 2026)", fontsize=14, fontweight="bold")
plt.xlabel("Station / Access Point", fontsize=11)
plt.ylabel("Number of Vehicles", fontsize=11)
plt.xticks(rotation=15, ha="right")
plt.legend(title="Vehicle Type")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(VEHICLE_CHART_PNG, dpi=150)
plt.close()

# ==============================================================================
# 4. EXPORT FORMATTED EXCEL WORKBOOK
# ==============================================================================
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
        df_export.rename(columns={"index": "Guest Type / Category"}, inplace=True)
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
write_sheet(ws1, "MARA TRIANGLE - GUEST TOTALS BY TICKET TYPE", ticket_totals, include_index=False)

ws2 = wb.create_sheet("Guest Type Breakdown")
write_sheet(ws2, "MARA TRIANGLE - GUEST TYPE BREAKDOWN BY TICKET", guest_matrix, include_index=True)

ws3 = wb.create_sheet("Vehicles Per Station")
write_sheet(ws3, "MARA TRIANGLE - VEHICLE TRAFFIC PER STATION", veh_station, include_index=True)

wb.save(OUTPUT_EXCEL)
print(f"\nSuccessfully generated:\n - {OUTPUT_EXCEL}\n - {GUEST_CHART_PNG}\n - {VEHICLE_CHART_PNG}")
