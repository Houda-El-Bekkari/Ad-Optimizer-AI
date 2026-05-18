from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.audit_report import build_audit_report, save_audit_report


BASE_DIR = Path("app")
OUTPUT_DIR = BASE_DIR / "outputs"
OUT_AUDIT_PDF = OUTPUT_DIR / "audit_report.pdf"


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def _paragraph(text: Any, style: Any) -> Any:
    safe = escape(_text(text)).replace("\n", "<br/>")
    from reportlab.platypus import Paragraph

    return Paragraph(safe, style)


def _metric(value: Any, suffix: str = "", fallback: str = "--") -> str:
    if value is None:
        return fallback

    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value, fallback)

    if abs(number) >= 10:
        text = f"{number:.1f}".rstrip("0").rstrip(".")
    else:
        text = f"{number:.2f}".rstrip("0").rstrip(".")

    return f"{text}{suffix}"


def _money(value: Any) -> str:
    try:
        return f"{float(value):.2f} EUR"
    except (TypeError, ValueError):
        return "--"


def _build_table(rows: list[list[Any]], col_widths: list[float] | None = None) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f8f3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10231d")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe8ed")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _add_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.35, 0.43, 0.46)
    canvas.drawString(doc.leftMargin, 18, "AdOptimizer AI - Rapport d'audit")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 18, f"Page {doc.page}")
    canvas.restoreState()


def run_audit_report_pdf(output_path: Path = OUT_AUDIT_PDF) -> dict[str, Any]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer
    except ImportError as exc:
        return {
            "status": "error",
            "message": (
                "ReportLab n'est pas installe. Lancez: "
                ".\\venv\\Scripts\\python.exe -m pip install reportlab"
            ),
            "details": str(exc),
        }

    report = build_audit_report()
    save_audit_report(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AuditTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#10231d"),
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "AuditSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#10231d"),
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "AuditBody",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#31525a"),
        fontSize=9.5,
        leading=13,
        spaceAfter=6,
    )
    strong_style = ParagraphStyle(
        "AuditStrong",
        parent=body_style,
        textColor=colors.HexColor("#10231d"),
        fontName="Helvetica-Bold",
    )
    small_style = ParagraphStyle(
        "AuditSmall",
        parent=body_style,
        textColor=colors.HexColor("#63737b"),
        fontSize=8,
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.4 * cm,
        title="AdOptimizer AI - Rapport d'audit",
    )

    story: list[Any] = []
    portfolio = report.get("portfolio_summary", {})
    executive = report.get("executive_summary", {})
    campaigns = report.get("campaigns", [])

    story.append(_paragraph("Rapport d'audit campagnes", title_style))
    story.append(_paragraph("AdOptimizer AI - Optimisation campagne existante", strong_style))
    generated_at = report.get("generated_at") or datetime.now().isoformat()
    story.append(_paragraph(f"Genere le : {generated_at}", small_style))
    story.append(Spacer(1, 10))

    story.append(_paragraph("Resume executif", section_style))
    story.append(_paragraph(executive.get("headline", "Audit genere avec succes."), body_style))
    for finding in executive.get("key_findings", []):
        story.append(_paragraph(f"- {finding}", body_style))

    story.append(_paragraph("Vue portfolio", section_style))
    metric_rows = [
        ["Indicateur", "Valeur", "Indicateur", "Valeur"],
        ["Campagnes auditees", _metric(portfolio.get("n_campaigns")), "Health score moyen", _metric(portfolio.get("average_health_score"))],
        ["Spend courant", _money(portfolio.get("total_current_spend")), "Actions IA", _metric(sum((portfolio.get("recommended_action_counts") or {}).values()))],
    ]
    story.append(_build_table(metric_rows, [4.1 * cm, 3.4 * cm, 4.1 * cm, 3.4 * cm]))

    priority_actions = executive.get("priority_actions", [])
    if priority_actions:
        story.append(_paragraph("Actions prioritaires", section_style))
        rows = [["Campagne", "Action", "Conseil court"]]
        for action in priority_actions:
            campaign_label = f"{action.get('platform', '')} - {action.get('campaign_id', '')}"
            rows.append([
                _paragraph(campaign_label, body_style),
                _paragraph(action.get("action", "Action a confirmer"), body_style),
                _paragraph(action.get("short_summary") or action.get("advice") or action.get("summary"), body_style),
            ])
        story.append(_build_table(rows, [4.0 * cm, 4.2 * cm, 7.2 * cm]))

    story.append(PageBreak())
    story.append(_paragraph("Audit detaille par campagne", section_style))

    for index, campaign in enumerate(campaigns):
        if index:
            story.append(Spacer(1, 12))

        platform = campaign.get("platform", "plateforme")
        campaign_id = campaign.get("campaign_id", "campagne")
        story.append(_paragraph(f"{platform} - {campaign_id}", section_style))

        current = campaign.get("current_kpis", {}) or {}
        predicted = campaign.get("predicted_kpis", {}) or {}
        trend = campaign.get("trend", {}) or {}
        root = campaign.get("root_cause", {}) or {}
        rec = campaign.get("business_recommendation", {}) or {}
        anomaly = campaign.get("anomaly_summary", {}) or {}

        rows = [
            ["KPI", "Valeur", "KPI", "Valeur"],
            ["Health score", _metric(campaign.get("health_score")), "Statut", campaign.get("status", "UNKNOWN")],
            ["ROAS actuel", _metric(current.get("roas"), "x"), "ROAS J+14", _metric(predicted.get("roas_h14"), "x")],
            ["Conversions", _metric(current.get("conversions")), "Conversions J+14", _metric(predicted.get("conversions_h14"))],
            ["Spend", _money(current.get("spend")), "Tendance ROAS", _metric(trend.get("roas_trend_pct"), "%")],
        ]
        story.append(_build_table(rows, [3.7 * cm, 3.8 * cm, 3.7 * cm, 3.8 * cm]))

        story.append(_paragraph("Recommandation IA", section_style))
        story.append(_paragraph(rec.get("title", "Action a confirmer"), strong_style))
        story.append(_paragraph(rec.get("summary", ""), body_style))
        if rec.get("advice"):
            story.append(_paragraph(f"Conseil business : {rec.get('advice')}", body_style))

        story.append(_paragraph("Anomalies", section_style))
        story.append(_paragraph(anomaly.get("business_summary", "Aucune anomalie prioritaire."), body_style))
        anomaly_rows = [
            ["Signal", "Niveau", "Periode", "ROAS observe"],
            [
                anomaly.get("main_signal", "--"),
                anomaly.get("level", "--"),
                anomaly.get("period", "--"),
                anomaly.get("roas_range", "--"),
            ],
        ]
        story.append(_build_table(anomaly_rows, [4.0 * cm, 3.0 * cm, 4.3 * cm, 4.0 * cm]))

        story.append(_paragraph("Cause principale", section_style))
        cause = root.get("label") or root.get("code") or "Cause a confirmer"
        confidence = _metric(float(root.get("confidence", 0)) * 100 if root.get("confidence") is not None else None, "%")
        story.append(_paragraph(f"{cause} - confiance {confidence}", body_style))

    next_steps = report.get("next_steps", [])
    if next_steps:
        story.append(_paragraph("Plan d'action global", section_style))
        for step in next_steps:
            story.append(_paragraph(f"- {step}", body_style))

    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)

    return {
        "status": "success",
        "message": "PDF d'audit genere avec succes",
        "output_file": str(output_path),
        "data": report,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_audit_report_pdf(), ensure_ascii=True, indent=2, default=str))
