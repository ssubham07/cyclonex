"""GET /api/v1/reports/{id}/export — export cyclone report as JSON or PDF."""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import io
import json
from datetime import datetime, timezone

from app.database import get_db
from app.models.cyclone import Cyclone, TrackPoint, Prediction

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("/{cyclone_id}/export")
async def export_report(
    cyclone_id: int,
    format: str = "json",
    db: AsyncSession = Depends(get_db),
):
    q = select(Cyclone).where(Cyclone.id == cyclone_id).options(selectinload(Cyclone.track_points))
    result = await db.execute(q)
    cyclone = result.scalar_one_or_none()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")

    # Get latest prediction
    q2 = select(Prediction).where(Prediction.cyclone_id == cyclone_id).order_by(Prediction.timestamp.desc()).limit(1)
    r2 = await db.execute(q2)
    pred = r2.scalar_one_or_none()

    report = {
        "report_generated": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "CycloNex AI Prototype — NOT an official IMD warning. Official warnings issued exclusively by IMD.",
        "cyclone": {
            "id": cyclone.id,
            "name": cyclone.name,
            "year": cyclone.year,
            "status": cyclone.status,
            "category": cyclone.category,
            "max_wind_kt": cyclone.max_wind_kt,
            "min_pressure_hpa": cyclone.min_pressure_hpa,
            "current_lat": cyclone.current_lat,
            "current_lon": cyclone.current_lon,
            "confidence": cyclone.confidence,
            "source": cyclone.source,
        },
        "track_points": [
            {
                "timestamp": tp.timestamp.isoformat() if tp.timestamp else None,
                "lat": tp.lat,
                "lon": tp.lon,
                "wind_kt": tp.max_wind_kt,
                "category": tp.category,
                "is_forecast": tp.is_forecast,
                "forecast_hour": tp.forecast_hour,
            }
            for tp in sorted(cyclone.track_points, key=lambda t: t.timestamp or datetime.min)
        ],
        "latest_prediction": {
            "detection_prob": pred.detection_prob if pred else None,
            "detected": pred.detected if pred else None,
            "category": pred.category if pred else None,
            "intensity_trend": pred.intensity_trend if pred else None,
            "confidence": pred.confidence if pred else None,
            "evidence": pred.xai_evidence if pred else None,
            "track_forecast": pred.track_forecast if pred else None,
        } if pred else None,
    }

    if format == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph(f"🌪️ CycloNex — Cyclone Report: {cyclone.name} ({cyclone.year})", styles["Title"]))
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph(f"<b>DISCLAIMER:</b> {report['disclaimer']}", styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))

            # Summary table
            data = [
                ["Field", "Value"],
                ["Category", cyclone.category or "—"],
                ["Max Wind", f"{cyclone.max_wind_kt or 0:.0f} kt"],
                ["Min Pressure", f"{cyclone.min_pressure_hpa or 0:.0f} hPa"],
                ["Last Position", f"{cyclone.current_lat or 0:.1f}°N, {cyclone.current_lon or 0:.1f}°E"],
                ["Model Confidence", f"{(cyclone.confidence or 0) * 100:.1f}%"],
                ["Status", cyclone.status],
                ["Data Source", cyclone.source],
            ]
            table = Table(data, colWidths=[6*cm, 10*cm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.lightgrey, colors.white]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.5*cm))

            if pred and pred.xai_evidence:
                story.append(Paragraph("<b>AI Evidence Summary:</b>", styles["Heading2"]))
                story.append(Paragraph(pred.xai_evidence, styles["Normal"]))

            doc.build(story)
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=cyclonex_{cyclone.name}_{cyclone.year}.pdf"}
            )
        except ImportError:
            pass  # Fall through to JSON if reportlab not available

    # Default: JSON
    return JSONResponse(content=report, headers={
        "Content-Disposition": f"attachment; filename=cyclonex_{cyclone.name}_{cyclone.year}.json"
    })
