"""Build HTML, PDF, and Excel reports from current computed results and selected charts."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from base64 import b64encode
from numbers import Number
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage, KeepTogether
from openpyxl.chart import BarChart, Reference
from .profiler import profile_dataset, descriptive_table
from .insights_engine import generate_insights


def _default_report_data(df: pd.DataFrame) -> dict:
    profile = profile_dataset(df)
    numeric = list(df.select_dtypes(include=np.number).columns)
    fallback = list(df.columns[:1]) if len(df.columns) and not numeric else []
    correlations = df[numeric].corr(method="pearson", min_periods=2) if 2 <= len(numeric) <= 200 else None
    return {"profile": profile, "descriptive": descriptive_table(df), "insights": generate_insights(df),
            "correlations": correlations, "chart_columns": (numeric[:2] if numeric else fallback)}


def _serializable(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value.to_dict()
    if hasattr(value, "item"):
        try: return value.item()
        except Exception: pass
    if isinstance(value, dict): return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_serializable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return str(value)


def _chart_pngs(df: pd.DataFrame, selected_columns: list[str] | None = None, max_charts: int = 4) -> list[tuple[str, bytes]]:
    """Return compact PNG charts for selected columns, derived only from current data."""
    selected = list(dict.fromkeys(c for c in (selected_columns or []) if c in df.columns))[:max_charts]
    charts: list[tuple[str, bytes]] = []
    for column in selected:
        series = df[column]
        fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
        try:
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if values.empty:
                    plt.close(fig); continue
                bins = min(30, max(5, int(np.sqrt(len(values)))))
                ax.hist(values.to_numpy(), bins=bins, edgecolor="white", linewidth=.6)
                ax.set_xlabel(str(column)); ax.set_ylabel("Observations")
                ax.set_title(f"Distribution: {str(column)[:80]}")
            else:
                values = series.astype("string").fillna("(missing)").value_counts().head(12).sort_values()
                if values.empty:
                    plt.close(fig); continue
                labels = [str(x)[:36] for x in values.index]
                ax.barh(labels, values.to_numpy())
                ax.set_xlabel("Observations"); ax.set_ylabel(str(column))
                ax.set_title(f"Most frequent values: {str(column)[:65]}")
            ax.grid(axis="x", alpha=.2)
            buffer = BytesIO(); fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            charts.append((str(column), buffer.getvalue()))
        finally:
            plt.close(fig)
    numeric_selected = [c for c in selected if c in df.select_dtypes(include=np.number).columns]
    if len(numeric_selected) >= 2 and len(charts) < max_charts:
        corr = df[numeric_selected].corr(method="pearson", min_periods=2)
        if corr.shape[0] >= 2 and corr.notna().any().any():
            fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
            try:
                image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
                ax.set_xticks(range(len(corr.columns)), labels=[str(x)[:24] for x in corr.columns], rotation=35, ha="right")
                ax.set_yticks(range(len(corr.index)), labels=[str(x)[:24] for x in corr.index])
                for i in range(len(corr.index)):
                    for j in range(len(corr.columns)):
                        value = corr.iloc[i,j]
                        if pd.notna(value): ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
                ax.set_title("Pearson correlation of selected numeric features")
                fig.colorbar(image, ax=ax, fraction=.045, pad=.04)
                buffer = BytesIO(); fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
                charts.append(("Selected-feature correlation", buffer.getvalue()))
            finally:
                plt.close(fig)
    return charts


def _summary_html(df: pd.DataFrame, data: dict, transformation_log: list[dict]) -> str:
    profile = data.get("profile") or profile_dataset(df)
    descriptive = data.get("descriptive")
    if not isinstance(descriptive, pd.DataFrame): descriptive = descriptive_table(df)
    insights = data.get("insights") or generate_insights(df)
    charts = _chart_pngs(df, data.get("chart_columns"))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric_cards = f"<div class='metrics'><div><b>{len(df):,}</b><span>Rows</span></div><div><b>{len(df.columns):,}</b><span>Columns</span></div><div><b>{int(df.isna().sum().sum()):,}</b><span>Missing cells</span></div><div><b>{int(df.duplicated().sum()):,}</b><span>Duplicate rows</span></div></div>"
    warning_html = "".join(f"<li>{escape(str(w))}</li>" for w in profile.get("quality_warnings", [])) or "<li>No quality warnings were generated by the configured checks.</li>"
    insight_html = "".join(f"<article><h3>{escape(str(i.get('title','Insight')))}</h3><p>{escape(str(i.get('finding','')))}</p><small>Category: {escape(str(i.get('category','')))} · Sample size: {escape(str(i.get('sample_size','n/a')))} · Limitation: {escape(str(i.get('limitation','Not specified')))}</small><pre>{escape(json.dumps(_serializable(i.get('evidence',{})), ensure_ascii=False, indent=2))}</pre></article>" for i in insights) or "<p>No rule-based insights met the current criteria.</p>"
    log_html = "".join(f"<tr><td>{escape(str(e.get('timestamp_utc','')))}</td><td>{escape(str(e.get('operation','')))}</td><td>{escape(str(e.get('description','')))}</td><td>{escape(str(e.get('affected_rows','')))}</td></tr>" for e in transformation_log) or "<tr><td colspan='4'>No transformations recorded.</td></tr>"
    charts_html = "".join(f"<figure><img src='data:image/png;base64,{b64encode(png).decode('ascii')}' alt='{escape(title)}' style='width:100%;max-width:760px;height:auto'><figcaption>{escape(title)} — computed from the current dataset</figcaption></figure>" for title, png in charts)
    if not charts_html: charts_html = "<p>No chart columns were selected or the selected columns had no plottable values.</p>"
    descriptive_html = descriptive.to_html(index=False, border=0, classes="table", na_rep="—", float_format=lambda v: f"{v:.4g}")
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Data Science Analytics Report</title><style>
body{{font-family:Arial,Helvetica,sans-serif;color:#172033;max-width:1120px;margin:36px auto;padding:0 22px;line-height:1.55}}h1{{font-size:30px;margin-bottom:4px}}h2{{border-bottom:1px solid #d8dee9;padding-bottom:7px;margin-top:32px}}h3{{margin-bottom:5px}}.muted,small,figcaption{{color:#5d6678}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}}.metrics div{{border:1px solid #d8dee9;border-radius:10px;padding:16px}}.metrics b{{display:block;font-size:24px}}.metrics span{{font-size:12px;color:#687386}}article{{border-bottom:1px solid #e1e6ed;padding:10px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:10px;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}}th,td{{border:1px solid #dce2eb;padding:7px;text-align:left;vertical-align:top}}th{{background:#f0f3f8}}.table-wrap{{overflow-x:auto}}figure{{margin:18px 0;padding:12px;border:1px solid #e1e6ed;border-radius:8px}}figcaption{{font-size:12px}}@media(max-width:650px){{.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}</style></head><body>
<h1>Data Science Analytics Report</h1><p class='muted'>Generated {generated} · Values and findings are calculated from the current dataset.</p>{metric_cards}
<h2>Data quality</h2><ul>{warning_html}</ul><p>Overall missing-cell rate: {profile.get('missing_pct', 0)}%. No composite quality score is reported.</p>
<h2>Descriptive statistics</h2><div class='table-wrap'>{descriptive_html}</div>
<h2>Selected charts</h2>{charts_html}
<h2>Evidence-based insights</h2>{insight_html}
<h2>Transformation log</h2><table><thead><tr><th>UTC timestamp</th><th>Operation</th><th>Description</th><th>Affected rows</th></tr></thead><tbody>{log_html}</tbody></table>
<h2>Methodology and limitations</h2><p>Descriptive summaries use non-missing observations for each statistic. Associations are not causal evidence. Automated insights are deterministic screening rules, not domain-specific decisions. Statistical tests and model evaluations should be interpreted against their stated assumptions, sampling design, and practical context. Review transformations before using processed values downstream.</p>
</body></html>"""


def generate_html_report(df: pd.DataFrame, report_data: dict | None = None, transformation_log: list[dict] | None = None) -> bytes:
    if report_data is None: report_data = _default_report_data(df)
    return _summary_html(df, report_data, transformation_log or []).encode("utf-8")


def _pdf_table_value(value, max_length: int = 22) -> str:
    if value is None: return ""
    try:
        if pd.isna(value): return ""
    except (TypeError, ValueError): pass
    if isinstance(value, Number) and not isinstance(value, bool):
        try: return f"{float(value):.4g}"
        except (ValueError, TypeError): pass
    return str(value).replace("\n", " ")[:max_length]


def generate_pdf_report(df: pd.DataFrame, report_data: dict | None = None, transformation_log: list[dict] | None = None) -> bytes:
    """Create a compact PDF with bounded-width descriptive tables and selected chart images."""
    data = report_data or _default_report_data(df)
    profile = data.get("profile") or profile_dataset(df)
    descriptive = data.get("descriptive")
    if not isinstance(descriptive, pd.DataFrame): descriptive = descriptive_table(df)
    insights = data.get("insights") or generate_insights(df)
    charts = _chart_pngs(df, data.get("chart_columns"))
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(A4), rightMargin=12*mm, leftMargin=12*mm, topMargin=12*mm, bottomMargin=14*mm, title="Data Science Analytics Report", author="Data Science Analytics")
    styles = getSampleStyleSheet(); story = []
    story.append(Paragraph("Data Science Analytics Report", styles["Title"]))
    story.append(Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Source rows: {len(df):,} · Columns: {len(df.columns):,}", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Dataset overview", styles["Heading2"]))
    overview = [["Rows", "Columns", "Missing cells", "Duplicate rows", "Missing-cell rate"],
                [f"{len(df):,}", f"{len(df.columns):,}", f"{int(df.isna().sum().sum()):,}", f"{int(df.duplicated().sum()):,}", f"{profile.get('missing_pct',0)}%"]]
    table = Table(overview, colWidths=[30*mm, 30*mm, 38*mm, 38*mm, 44*mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e9eef5")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd4df")),("PADDING",(0,0),(-1,-1),6)])); story.append(table)
    story.append(Paragraph("Data quality notes", styles["Heading2"]))
    for warning in profile.get("quality_warnings", []) or ["No warnings were generated by the configured checks."]:
        story.append(Paragraph("• " + escape(str(warning)), styles["BodyText"]))
    story.append(Spacer(1, 5)); story.append(Paragraph("Descriptive statistics", styles["Heading2"]))
    preferred = ["column", "dtype", "count", "missing", "unique", "mean", "median", "std", "min", "max", "mode"]
    shown_columns = [c for c in preferred if c in descriptive.columns]
    compact = descriptive[shown_columns].copy() if shown_columns else pd.DataFrame(columns=["column", "dtype", "count", "missing", "unique"])
    headers = [str(c).replace("_", " ").title()[:18] for c in compact.columns]
    body = [headers]
    for _, row in compact.head(100).iterrows():
        vals = []
        for col, value in row.items():
            max_len = 25 if col in {"column", "mode"} else (14 if col == "dtype" else 12)
            vals.append(_pdf_table_value(value, max_len))
        body.append(vals)
    widths_mm = {"column":35, "dtype":18, "count":14, "missing":16, "unique":17, "mean":23, "median":23, "std":23, "min":20, "max":20, "mode":40}
    col_widths = [widths_mm.get(col, 20)*mm for col in compact.columns]
    tbl = Table(body, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e9eef5")),("GRID",(0,0),(-1,-1),.25,colors.HexColor("#cbd4df")),("FONTSIZE",(0,0),(-1,-1),7), ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),3)]))
    story.append(tbl)
    if len(descriptive) > 100: story.append(Paragraph("Only the first 100 columns are shown in this PDF table; export Excel for the complete summary.", styles["Italic"]))
    if charts:
        chart_cells = []
        for title, png in charts:
            image = ReportLabImage(BytesIO(png), width=124*mm, height=67*mm, kind="proportional")
            chart_cells.append((Paragraph(escape(title), styles["Italic"]), image))
        chart_tables = []
        for i in range(0, len(chart_cells), 2):
            left_cell = [chart_cells[i][0], chart_cells[i][1]]
            right_cell = [chart_cells[i+1][0], chart_cells[i+1][1]] if i+1 < len(chart_cells) else ""
            chart_table = Table([[left_cell, right_cell]], colWidths=[132*mm,132*mm], hAlign="LEFT")
            chart_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            chart_tables.append(chart_table)
        # Keep the chart section heading with its first pair of chart images.
        story.append(KeepTogether([Spacer(1, 10), Paragraph("Selected charts", styles["Heading2"]), chart_tables[0]]))
        story.extend(chart_tables[1:])
    story.append(Spacer(1, 8))
    insight_flowables = []
    if not insights:
        insight_flowables.append(Paragraph("No rule-based insights met the current criteria.", styles["BodyText"]))
    for item in insights[:30]:
        insight_flowables.extend([
            Paragraph(escape(str(item.get("title", "Insight"))), styles["Heading3"]),
            Paragraph(escape(str(item.get("finding", ""))), styles["BodyText"]),
            Paragraph(f"Category: {escape(str(item.get('category','')))} | Sample size: {escape(str(item.get('sample_size','n/a')))}", styles["Italic"]),
            Paragraph("Limitation: " + escape(str(item.get("limitation", "Not specified"))), styles["BodyText"]),
            Spacer(1, 3),
        ])
    if insight_flowables:
        # Keep the section heading and the first insight together to avoid orphan headings.
        story.append(KeepTogether([Paragraph("Evidence-based insights", styles["Heading2"]), *insight_flowables[:5]]))
        story.extend(insight_flowables[5:])
    else:
        story.append(Paragraph("Evidence-based insights", styles["Heading2"]))
    story.append(Paragraph("Transformation log", styles["Heading2"]))
    log = transformation_log or []
    log_data = [["Operation", "Description", "Affected rows"]] + [[str(e.get("operation", ""))[:35], str(e.get("description", ""))[:120], str(e.get("affected_rows", ""))] for e in log[:100]]
    log_tbl = Table(log_data, repeatRows=1, colWidths=[35*mm, 190*mm, 25*mm])
    log_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e9eef5")),("GRID",(0,0),(-1,-1),.25,colors.HexColor("#cbd4df")),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),4)])); story.append(log_tbl)
    story.append(Spacer(1, 8)); story.append(Paragraph("Methodology and limitations", styles["Heading2"]))
    story.append(Paragraph("Descriptive summaries are based on non-missing observations. Rule-based flags are prompts for investigation, not confirmed errors. Associations do not establish causation. Test assumptions, sampling design, effect sizes, uncertainty, and practical relevance should be reviewed before decisions are made.", styles["BodyText"]))
    doc.build(story)
    return out.getvalue()


def _safe_excel_text(value):
    """Escape untrusted text that Excel could otherwise interpret as a formula."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def _safe_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    safe.columns = [_safe_excel_text(str(c)[:200]) for c in safe.columns]
    if isinstance(safe.index, pd.RangeIndex):
        pass
    else:
        safe.index = [_safe_excel_text(str(idx)) for idx in safe.index]
        if safe.index.name is not None: safe.index.name = _safe_excel_text(str(safe.index.name))
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(_safe_excel_text)
    return safe


def _chart_data_for_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = df[column]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty: return pd.DataFrame(columns=["Category / bin", "Count"])
        if values.nunique() <= 12:
            counts = values.value_counts().sort_index()
            labels = [str(v)[:50] for v in counts.index]
            counts = counts.to_numpy()
        else:
            bins = pd.cut(values, bins=min(10, max(3, int(np.sqrt(len(values))))), duplicates="drop")
            grouped = bins.value_counts().sort_index()
            labels = [str(v)[:50] for v in grouped.index]
            counts = grouped.to_numpy()
    else:
        value_counts = series.astype("string").fillna("(missing)").value_counts().head(15)
        labels = [str(v)[:50] for v in value_counts.index]
        counts = value_counts.to_numpy()
    return pd.DataFrame({"Category / bin": labels, "Count": counts})


def generate_excel_report(df: pd.DataFrame, report_data: dict | None = None, transformation_log: list[dict] | None = None) -> bytes:
    data = report_data or _default_report_data(df)
    profile = data.get("profile") or profile_dataset(df)
    descriptive = data.get("descriptive")
    if not isinstance(descriptive, pd.DataFrame): descriptive = descriptive_table(df)
    insights = data.get("insights") or generate_insights(df)
    selected_chart_columns = list(dict.fromkeys(c for c in data.get("chart_columns", []) if c in df.columns))[:4]
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _safe_excel_frame(pd.DataFrame([{"rows":len(df), "columns":len(df.columns), "missing_cells":int(df.isna().sum().sum()),
                       "missing_percent":profile.get("missing_pct",0), "duplicate_rows":int(df.duplicated().sum()),
                       "generated_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds")}])).to_excel(writer, sheet_name="Overview", index=False)
        _safe_excel_frame(profile["column_summary"]).to_excel(writer, sheet_name="Column_Quality", index=False)
        _safe_excel_frame(descriptive).to_excel(writer, sheet_name="Descriptive_Stats", index=False)
        numeric_columns = list(df.select_dtypes(include="number").columns)
        if 2 <= len(numeric_columns) <= 200:
            _safe_excel_frame(df[numeric_columns].corr().rename_axis("column")).to_excel(writer, sheet_name="Correlations")
        elif len(numeric_columns) > 200:
            _safe_excel_frame(pd.DataFrame([{"note":"Correlation worksheet omitted because more than 200 numeric columns were detected. Select a smaller feature subset in the app."}])).to_excel(writer, sheet_name="Report_Notes", index=False)
        insight_frame = pd.DataFrame([{**{k:v for k,v in i.items() if k not in {"evidence"}}, "evidence_json":json.dumps(_serializable(i.get("evidence",{})), ensure_ascii=False)} for i in insights])
        _safe_excel_frame(insight_frame).to_excel(writer, sheet_name="Insights", index=False)
        log = transformation_log or []
        log_frame = pd.DataFrame([{**entry, "columns":", ".join(map(str, entry.get("columns",[]))), "affected_columns":", ".join(map(str, entry.get("affected_columns",[]))), "parameters":json.dumps(_serializable(entry.get("parameters",{})), ensure_ascii=False)} for entry in log])
        _safe_excel_frame(log_frame).to_excel(writer, sheet_name="Transformations", index=False)
        # Add a summary of the actual analyzed values. Do not permit formula-like text to become Excel formulas.
        _safe_excel_frame(df).to_excel(writer, sheet_name="Analyzed_Data", index=False, na_rep="")
        if selected_chart_columns:
            chart_sheet = writer.book.create_sheet("Charts")
            chart_sheet["A1"] = "Selected charts generated from the analyzed dataset"
            chart_sheet["A2"] = "Counts are plotted from the selected columns; numeric columns use bins when cardinality is high."
            chart_anchor_rows = [4, 22, 40, 58]
            for index, column in enumerate(selected_chart_columns):
                chart_data = _chart_data_for_column(df, column)
                if chart_data.empty: continue
                data_sheet_name = f"ChartData{index+1}"
                _safe_excel_frame(chart_data).to_excel(writer, sheet_name=data_sheet_name, index=False)
                data_ws = writer.book[data_sheet_name]
                chart = BarChart()
                chart.type = "bar"
                chart.style = 10
                chart.title = f"Distribution: {str(column)[:50]}"
                chart.y_axis.title = "Category / bin"
                chart.x_axis.title = "Count"
                chart.height = 7.0; chart.width = 13.0
                chart.add_data(Reference(data_ws, min_col=2, min_row=1, max_row=len(chart_data)+1), titles_from_data=True)
                chart.set_categories(Reference(data_ws, min_col=1, min_row=2, max_row=len(chart_data)+1))
                chart.legend = None
                anchor_row = chart_anchor_rows[index]
                anchor_col = "A" if index % 2 == 0 else "J"
                chart_sheet.add_chart(chart, f"{anchor_col}{anchor_row}")
        for ws in writer.book.worksheets:
            if ws.max_row >= 1:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
            # Data/summary worksheet widths; drawing charts are unaffected.
            for col_cells in ws.iter_cols(min_row=1, max_row=min(max(ws.max_row, 1), 200)):
                values = [len(str(c.value)) for c in col_cells if c.value is not None]
                width = min(max(max(values, default=10) + 2, 10), 48)
                ws.column_dimensions[col_cells[0].column_letter].width = width
    return buffer.getvalue()
