"""GET /api/v1/alerts — tiered alert system based on active cyclones."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from app.database import get_db
from app.models.cyclone import Cyclone
from app.schemas.cyclone import AlertOut

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

TIER_MAP = [
    # (min_wind, tier, color)
    (120, "Emergency — Super Cyclone Alert", "red"),
    (90,  "Emergency — Extremely Severe Cyclone Alert", "red"),
    (64,  "High Alert — Very Severe Cyclone Warning", "orange"),
    (48,  "Track Forecast — Severe Cyclone Warning", "orange"),
    (34,  "Advisory — Cyclonic Storm Watch", "yellow"),
    (28,  "Advisory — Deep Depression Watch", "yellow"),
    (17,  "Advisory — Depression Watch", "green"),
    (0,   "Information — Sub-Cyclonic Disturbance", "green"),
]


def _get_tier(wind_kt: float) -> tuple[str, str]:
    for min_w, tier, color in TIER_MAP:
        if wind_kt >= min_w:
            return tier, color
    return "Information", "green"


@router.get("", response_model=list[AlertOut])
async def get_alerts(db: AsyncSession = Depends(get_db)):
    q = select(Cyclone).where(Cyclone.status == "active").order_by(Cyclone.max_wind_kt.desc())
    result = await db.execute(q)
    active = result.scalars().all()

    # If no active, include most recent historical for demo
    if not active:
        q2 = select(Cyclone).order_by(Cyclone.updated_at.desc()).limit(3)
        result2 = await db.execute(q2)
        active = result2.scalars().all()

    alerts = []
    for cy in active:
        wind = cy.max_wind_kt or 0.0
        tier, color = _get_tier(wind)
        alerts.append(AlertOut(
            tier=tier,
            color=color,
            cyclone_id=cy.id,
            cyclone_name=cy.name,
            category=cy.category,
            wind_kt=cy.max_wind_kt,
            message=f"[PROTOTYPE — NOT OFFICIAL] {tier}: {cy.name} ({cy.category or 'Unknown'}) — {wind:.0f} kt. Official warnings from IMD only.",
            timestamp=cy.updated_at or datetime.now(timezone.utc),
        ))
    return alerts
