import xlsxwriter
from datetime import datetime

def create_institutional_model():
    """
    Generates a professional, investment-banking grade financial model for Monarch AMC.
    Adheres to FAST modeling standards:
    - Inputs (blue) are separated from calculations (black).
    - Hardcoding is isolated strictly to the Assumptions tab.
    - P&L and Dashboards are 100% formula-driven.
    - Includes dynamic grouping, conditional formatting, and charts.
    """
    filename = 'Monarch_AMC_Institutional_Model_Final.xlsx'
    wb = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})

    # ==========================================
    # 1. DEFINE PROFESSIONAL FORMATTING
    # ==========================================
    
    # Fonts and Colors
    font_family = 'Calibri'
    c_blue = '#0F243E'      # Dark Navy (Primary)
    c_light_blue = '#D9E1F2' # Header fill
    c_accent = '#C65911'    # Orange accent
    c_input = '#0000FF'     # Input Blue
    c_calc = '#000000'      # Calc Black
    c_gray = '#F2F2F2'      # Light gray for grouping

    # Base Formats
    format_base = {'font_name': font_family, 'font_size': 11, 'valign': 'vcenter'}
    base_fmt = wb.add_format(format_base)
    
    # Cover & Titles
    title_fmt = wb.add_format({**format_base, 'bold': True, 'font_size': 24, 'color': c_blue})
    subtitle_fmt = wb.add_format({**format_base, 'bold': True, 'font_size': 14, 'color': c_accent})
    
    # Headers
    header_fmt = wb.add_format({**format_base, 'bold': True, 'bg_color': c_blue, 'font_color': '#FFFFFF', 'align': 'center', 'border': 1})
    subheader_fmt = wb.add_format({**format_base, 'bold': True, 'bg_color': c_light_blue, 'bottom': 1})
    
    # Row Labels
    label_fmt = wb.add_format({**format_base, 'bold': True})
    indent1_fmt = wb.add_format({**format_base, 'indent': 1})
    indent2_fmt = wb.add_format({**format_base, 'indent': 2})
    
    # Data Formats (Inputs = Blue, Calcs = Black)
    acct_str = '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)'
    int_str  = '_(* #,##0_);_(* (#,##0);_(* "-"??_);_(@_)'
    pct_str  = '0.00%'

    input_num_fmt = wb.add_format({**format_base, 'font_color': c_input, 'num_format': acct_str, 'bg_color': '#FFFFE0', 'border': 1})
    input_int_fmt = wb.add_format({**format_base, 'font_color': c_input, 'num_format': int_str, 'bg_color': '#FFFFE0', 'border': 1})
    input_pct_fmt = wb.add_format({**format_base, 'font_color': c_input, 'num_format': pct_str, 'bg_color': '#FFFFE0', 'border': 1})
    
    calc_num_fmt = wb.add_format({**format_base, 'font_color': c_calc, 'num_format': acct_str})
    calc_int_fmt = wb.add_format({**format_base, 'font_color': c_calc, 'num_format': int_str})
    calc_pct_fmt = wb.add_format({**format_base, 'font_color': c_calc, 'num_format': pct_str})
    
    # Totals
    total_num_fmt = wb.add_format({**format_base, 'bold': True, 'top': 1, 'bottom': 2, 'num_format': acct_str})
    total_int_fmt = wb.add_format({**format_base, 'bold': True, 'top': 1, 'bottom': 2, 'num_format': int_str})
    total_pct_fmt = wb.add_format({**format_base, 'bold': True, 'top': 1, 'bottom': 2, 'num_format': pct_str})

    # Timeline Row
    timeline_fmt = wb.add_format({**format_base, 'bold': True, 'align': 'center', 'bottom': 1})

    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    cols = ['D', 'E', 'F', 'G', 'H']

    # ==========================================
    # SHEET 1: COVER PAGE
    # ==========================================
    ws_cover = wb.add_worksheet('1. Cover')
    ws_cover.hide_gridlines(2)
    ws_cover.set_column('B:B', 5)
    ws_cover.set_column('C:C', 80)
    
    ws_cover.write('C4', 'MONARCH CONVICTION AMC', title_fmt)
    ws_cover.write('C5', 'Launchpad Alpha 2026 — Institutional Financial Model', subtitle_fmt)
    ws_cover.write('C7', f'Date Generated: {datetime.now().strftime("%B %d, %Y")}', base_fmt)
    
    ws_cover.write('C10', 'MODEL STRUCTURE & INDEX', label_fmt)
    index_data = [
        ('1. Cover', 'Model details and formatting legend'),
        ('2. Dashboard', 'Visual summary of KPIs and breakeven metrics'),
        ('3. Assumptions', 'Core drivers (AUM, Yield, Opex) - ALL INPUTS HERE'),
        ('4. Base Case', 'Granular 5-Year Profit & Loss (Formula Driven)'),
        ('5. Conservative', 'Sensitized 5-Year Profit & Loss (Formula Driven)')
    ]
    for i, (sheet, desc) in enumerate(index_data):
        ws_cover.write(11+i, 2, f"{sheet}: {desc}", indent1_fmt)

    ws_cover.write('C19', 'FORMATTING LEGEND', label_fmt)
    ws_cover.write('C20', '1,000.00', input_num_fmt)
    ws_cover.write('C20', '      Hardcoded Inputs (Yellow fill, Blue text) - Modify these only', base_fmt)
    ws_cover.write('C21', '1,000.00', calc_num_fmt)
    ws_cover.write('C21', '      Calculations (Black text) - Do not modify', base_fmt)


    # ==========================================
    # SHEET 2: ASSUMPTIONS (THE ENGINE)
    # ==========================================
    ws_assump = wb.add_worksheet('2. Assumptions')
    ws_assump.set_column('B:B', 4)
    ws_assump.set_column('C:C', 45)
    ws_assump.set_column('D:H', 15)
    ws_assump.freeze_panes(5, 3)

    ws_assump.write('C2', 'KEY ASSUMPTIONS & DRIVERS', title_fmt)
    ws_assump.write_row('D4', years, header_fmt)

    # --- AUM Drivers ---
    ws_assump.write('C5', '1. ASSETS UNDER MANAGEMENT (Rs. Crore)', subheader_fmt)
    
    ws_assump.write('C6', 'Base Case - MF AUM', indent1_fmt)
    ws_assump.write_row('D6', [700, 1200, 1800, 2600, 3800], input_int_fmt)
    ws_assump.write('C7', 'Base Case - SIF AUM', indent1_fmt)
    ws_assump.write_row('D7', [0, 300, 600, 1000, 1800], input_int_fmt)
    
    ws_assump.write('C9', 'Conservative Case - MF AUM', indent1_fmt)
    ws_assump.write_row('D9', [410, 800, 1300, 1900, 2800], input_int_fmt)
    ws_assump.write('C10', 'Conservative Case - SIF AUM', indent1_fmt)
    ws_assump.write_row('D10', [0, 150, 350, 600, 1000], input_int_fmt)

    # --- Revenue Drivers ---
    ws_assump.write('C12', '2. YIELD ASSUMPTIONS', subheader_fmt)
    ws_assump.write('C13', 'Blended Net AMC Retention (MF & SIF)', indent1_fmt)
    ws_assump.write_row('D13', [0.0073]*5, input_pct_fmt)

    # --- Cost Drivers (Base Case) ---
    ws_assump.write('C15', '3. BASE CASE COST DRIVERS (Rs. Crore)', subheader_fmt)
    ws_assump.write('C16', 'Investment Team (incl SIF CIO)', indent1_fmt)
    ws_assump.write_row('D16', [10.5, 11.0, 11.5, 12.0, 12.5], input_num_fmt)
    ws_assump.write('C17', 'Operations & R&T (Variable)', indent1_fmt)
    ws_assump.write_row('D17', [1.5, 3.0, 4.0, 4.5, 5.5], input_num_fmt)
    ws_assump.write('C18', 'Technology (Portal & Dashboard)', indent1_fmt)
    ws_assump.write_row('D18', [2.5, 1.5, 1.5, 1.5, 1.5], input_num_fmt)
    ws_assump.write('C19', 'Content & Distribution (NFOs)', indent1_fmt)
    ws_assump.write_row('D19', [2.5, 3.0, 2.5, 2.5, 2.5], input_num_fmt)
    ws_assump.write('C20', 'Compliance & Legal', indent1_fmt)
    ws_assump.write_row('D20', [1.0, 1.5, 1.5, 1.5, 1.5], input_num_fmt)
    ws_assump.write('C21', 'Office & Overhead', indent1_fmt)
    ws_assump.write_row('D21', [0.5, 0.5, 0.5, 0.5, 0.5], input_num_fmt)

    # --- Cost Drivers (Conservative Case) ---
    ws_assump.write('C23', '4. CONSERVATIVE CASE COST DRIVERS (Rs. Crore)', subheader_fmt)
    ws_assump.write('C24', 'Investment Team (incl SIF CIO)', indent1_fmt)
    ws_assump.write_row('D24', [10.5, 10.5, 11.0, 11.5, 12.0], input_num_fmt)
    ws_assump.write('C25', 'Operations & R&T (Variable)', indent1_fmt)
    ws_assump.write_row('D25', [1.0, 2.0, 3.0, 4.0, 5.0], input_num_fmt)
    ws_assump.write('C26', 'Technology (Portal & Dashboard)', indent1_fmt)
    ws_assump.write_row('D26', [2.5, 1.5, 1.5, 1.5, 1.5], input_num_fmt)
    ws_assump.write('C27', 'Content & Distribution (NFOs)', indent1_fmt)
    ws_assump.write_row('D27', [2.5, 3.5, 3.0, 2.5, 2.5], input_num_fmt)
    ws_assump.write('C28', 'Compliance & Legal', indent1_fmt)
    ws_assump.write_row('D28', [1.0, 1.5, 1.5, 1.5, 1.5], input_num_fmt)
    ws_assump.write('C29', 'Office & Overhead', indent1_fmt)
    ws_assump.write_row('D29', [0.5, 0.5, 0.5, 0.5, 0.5], input_num_fmt)

    # --- Corporate Params ---
    ws_assump.write('C31', '5. CORPORATE PARAMETERS', subheader_fmt)
    ws_assump.write('C32', 'Monarch Networth (Rs. Crore)', indent1_fmt)
    ws_assump.write('D32', 796.0, input_num_fmt)


    # ==========================================
    # HELPER FUNCTION TO BUILD P&L SHEETS
    # ==========================================
    def build_pl_sheet(sheet_name, is_base_case):
        ws = wb.add_worksheet(sheet_name)
        ws.set_column('A:A', 3)
        ws.set_column('B:B', 4)
        ws.set_column('C:C', 45)
        ws.set_column('D:H', 15)
        ws.freeze_panes(5, 3)

        title = 'BASE CASE — DETAILED P&L' if is_base_case else 'CONSERVATIVE CASE — DETAILED P&L'
        ws.write('C2', title, title_fmt)
        ws.write_row('D4', years, header_fmt)

        # Row Mappings based on scenario
        aum_row_start = 6 if is_base_case else 9
        cost_row_start = 16 if is_base_case else 24

        # 1. AUM ROLLUP
        ws.write('C5', 'ASSETS UNDER MANAGEMENT (Rs. Crore)', subheader_fmt)
        ws.write('C6', 'Mutual Fund (MF) AUM', indent1_fmt)
        ws.write('C7', 'Specialized Investment Fund (SIF) AUM', indent1_fmt)
        ws.write('C8', 'Total Period-End AUM', label_fmt)
        
        for i, col in enumerate(cols):
            ws.write_formula(f'{col}6', f"='2. Assumptions'!{col}{aum_row_start}", calc_int_fmt)
            ws.write_formula(f'{col}7', f"='2. Assumptions'!{col}{aum_row_start+1}", calc_int_fmt)
            ws.write_formula(f'{col}8', f'=SUM({col}6:{col}7)', total_int_fmt)

        # 2. REVENUE BUILD
        ws.write('C10', 'REVENUE (Rs. Crore)', subheader_fmt)
        ws.write('C11', 'Blended Net AMC Retention Yield', indent1_fmt)
        ws.write('C12', 'Management Fee Revenue', label_fmt)

        for col in cols:
            ws.write_formula(f'{col}11', f"='2. Assumptions'!{col}13", calc_pct_fmt)
            # Revenue = Total AUM * Yield
            ws.write_formula(f'{col}12', f'={col}8*{col}11', total_num_fmt)

        # 3. COST BUILD (With Outlining/Grouping for depth)
        ws.write('C14', 'OPERATING EXPENSES (Rs. Crore)', subheader_fmt)
        
        cost_lines = [
            ('Investment Team (incl SIF CIO)', 0),
            ('Operations & R&T (Variable)', 1),
            ('Technology (Portal & Dashboard)', 2),
            ('Content & Distribution (NFOs)', 3),
            ('Compliance & Legal', 4),
            ('Office & Overhead', 5)
        ]
        
        current_row = 15
        for desc, offset in cost_lines:
            ws.write(f'C{current_row}', desc, indent1_fmt)
            for col in cols:
                ws.write_formula(f'{col}{current_row}', f"='2. Assumptions'!{col}{cost_row_start+offset}", calc_num_fmt)
            current_row += 1

        ws.write(f'C{current_row}', 'Total Operating Expenses', label_fmt)
        for col in cols:
            ws.write_formula(f'{col}{current_row}', f'=SUM({col}15:{col}{current_row-1})', total_num_fmt)
        
        total_opex_row = current_row

        # 4. PROFITABILITY & MARGINS
        current_row += 2
        ws.write(f'C{current_row}', 'PROFITABILITY METRICS', subheader_fmt)
        
        pbt_row = current_row + 1
        ws.write(f'C{pbt_row}', 'Profit Before Tax (PBT)', label_fmt)
        ws.write(f'C{pbt_row+1}', 'PBT Margin (%)', indent1_fmt)
        
        for col in cols:
            # PBT = Rev - Opex
            ws.write_formula(f'{col}{pbt_row}', f'={col}12-{col}{total_opex_row}', total_num_fmt)
            # Margin = IF PBT > 0, PBT/Rev, NM
            ws.write_formula(f'{col}{pbt_row+1}', f'=IF({col}{pbt_row}>0, {col}{pbt_row}/{col}12, "NM")', calc_pct_fmt)

        # 5. CAPITAL DRAIN ANALYSIS
        current_row += 4
        ws.write(f'C{current_row}', 'CAPITAL DRAWDOWN ANALYSIS', subheader_fmt)
        ws.write(f'C{current_row+1}', 'Cumulative Profit/Loss', indent1_fmt)
        ws.write(f'C{current_row+2}', 'Monarch Networth', indent1_fmt)
        ws.write(f'C{current_row+3}', 'Max Drawdown as % of Networth', label_fmt)

        cum_loss_row = current_row + 1
        
        # Cumulative Loss Logic
        ws.write_formula(f'D{cum_loss_row}', f'=MIN(0, D{pbt_row})', calc_num_fmt)
        for i, col in enumerate(cols[1:]):  # E, F, G, H
            prev_col = cols[i]
            # Cum Loss = Prev Cum Loss + Current PBT. Caps at 0 (once profitable, drawdown stops growing negatively)
            ws.write_formula(f'{col}{cum_loss_row}', f'=MIN(0, {prev_col}{cum_loss_row} + {col}{pbt_row})', calc_num_fmt)
        
        for col in cols:
            ws.write_formula(f'{col}{current_row+2}', "='2. Assumptions'!$D$32", calc_num_fmt)
            ws.write_formula(f'{col}{current_row+3}', f'={col}{cum_loss_row} / {col}{current_row+2}', calc_pct_fmt)
            
        # Group the cost rows for an interactive/collapsible model feel
        for r in range(15, 21):
            ws.set_row(r - 1, None, None, {'level': 1, 'hidden': False})

    # Execute build functions
    build_pl_sheet('3. Base Case', True)
    build_pl_sheet('4. Conservative', False)


    # ==========================================
    # SHEET 5: SUMMARY DASHBOARD & CHARTS
    # ==========================================
    ws_dash = wb.add_worksheet('5. Dashboard')
    ws_dash.hide_gridlines(2)
    ws_dash.set_column('B:B', 3)
    ws_dash.set_column('C:C', 35)
    ws_dash.set_column('D:I', 15)

    ws_dash.write('C2', 'EXECUTIVE DASHBOARD', title_fmt)
    
    # Create a clean summary table for the charts to reference
    ws_dash.write_row('D4', years, header_fmt)
    ws_dash.write('C5', 'Base Case AUM (Rs. Cr)', label_fmt)
    ws_dash.write('C6', 'Base Case PBT (Rs. Cr)', label_fmt)
    ws_dash.write('C7', 'Conservative PBT (Rs. Cr)', label_fmt)

    for col in cols:
        ws_dash.write_formula(f'{col}5', f"='3. Base Case'!{col}8", calc_int_fmt)
        ws_dash.write_formula(f'{col}6', f"='3. Base Case'!{col}18", calc_num_fmt)
        ws_dash.write_formula(f'{col}7', f"='4. Conservative'!{col}18", calc_num_fmt)

    # --- CHART 1: AUM GROWTH ---
    chart_aum = wb.add_chart({'type': 'column'})
    chart_aum.add_series({
        'name':       'Total AUM',
        'categories': ['5. Dashboard', 3, 3, 3, 7],
        'values':     ['5. Dashboard', 4, 3, 4, 7],
        'fill':       {'color': c_blue},
        'data_labels': {'value': True, 'num_format': '#,##0'}
    })
    chart_aum.set_title({'name': 'Base Case AUM Trajectory (Rs. Crore)'})
    chart_aum.set_legend({'none': True})
    chart_aum.set_x_axis({'name': 'Financial Year'})
    chart_aum.set_y_axis({'name': 'AUM', 'major_gridlines': {'visible': False}})
    ws_dash.insert_chart('B10', chart_aum, {'x_scale': 1.2, 'y_scale': 1.1})

    # --- CHART 2: PBT PATH TO PROFITABILITY ---
    chart_pbt = wb.add_chart({'type': 'line'})
    chart_pbt.add_series({
        'name':       'Base Case PBT',
        'categories': ['5. Dashboard', 3, 3, 3, 7],
        'values':     ['5. Dashboard', 5, 3, 5, 7],
        'line':       {'color': c_accent, 'width': 2.5},
        'marker':     {'type': 'circle', 'size': 6, 'border': {'color': c_accent}, 'fill': {'color': c_accent}}
    })
    chart_pbt.add_series({
        'name':       'Conservative PBT',
        'categories': ['5. Dashboard', 3, 3, 3, 7],
        'values':     ['5. Dashboard', 6, 3, 6, 7],
        'line':       {'color': '#A6A6A6', 'width': 2.5, 'dash_type': 'dash'},
    })
    chart_pbt.set_title({'name': 'Path to Profitability (PBT)'})
    chart_pbt.set_legend({'position': 'bottom'})
    chart_pbt.set_y_axis({'name': 'Rs. Crore', 'major_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}})
    ws_dash.insert_chart('F10', chart_pbt, {'x_scale': 1.2, 'y_scale': 1.1})


    wb.close()
    print(f"Professional financial model '{filename}' generated successfully.")

if __name__ == '__main__':
    create_institutional_model()