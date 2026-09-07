"""
Database seed script — populates the database with real IBTrACS-based
historical cyclone data (Amphan, Fani, Biparjoy, Yaas, Asani) for demo use.
Run: python -m app.services.seed
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone

from app.database import init_db, AsyncSessionLocal
from app.models.cyclone import Cyclone, TrackPoint, Prediction
from app.data.ibtracs import BUNDLED_STORMS
from app.schemas.cyclone import wind_to_category, IMD_CATEGORIES
from app.ml.inference import _synthetic_inference
from app.data.synthetic import generate_frame

logger = logging.getLogger(__name__)


def _parse_ts(ts_str: str) -> datetime:
    ts_str = ts_str.replace(" ", "T")
    if not ts_str.endswith("Z") and "+" not in ts_str:
        ts_str += "Z"
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


async def seed_database():
    await init_db()

    async with AsyncSessionLocal() as db:
        # Check if already seeded
        from sqlalchemy import select, func
        count_q = select(func.count()).select_from(Cyclone)
        result = await db.execute(count_q)
        count = result.scalar()
        if count > 0:
            logger.info(f"Database already has {count} cyclones, skipping seed.")
            return

        logger.info("Seeding database with IBTrACS demo cyclones...")

        for storm_data in BUNDLED_STORMS:
            track = storm_data["track"]
            if not track:
                continue

            # Determine peak intensity
            peak_wind = max(pt["WIND"] for pt in track)
            cat_idx, cat_name = wind_to_category(peak_wind)
            cat_idx = max(0, cat_idx)

            last_pt = track[-1]
            first_pt = track[0]

            # Create cyclone record
            cyclone = Cyclone(
                name=storm_data["NAME"],
                year=storm_data["YEAR"],
                basin="NI",
                status="historical",
                category=cat_name,
                category_code=cat_idx,
                max_wind_kt=float(peak_wind),
                current_lat=float(last_pt["LAT"]),
                current_lon=float(last_pt["LON"]),
                min_pressure_hpa=float(last_pt.get("PRES", 970)),
                confidence=0.92,
                flagged_for_review=False,
                source="ibtracs",
                extra={"sid": storm_data["SID"], "season": storm_data["SEASON"]},
            )
            db.add(cyclone)
            await db.flush()  # get ID

            # Add historical track points
            for pt in track:
                ts = _parse_ts(pt["ISO_TIME"])
                wind = float(pt["WIND"])
                c_idx, c_name = wind_to_category(wind)
                c_idx = max(0, c_idx)
                tp = TrackPoint(
                    cyclone_id=cyclone.id,
                    timestamp=ts,
                    lat=float(pt["LAT"]),
                    lon=float(pt["LON"]),
                    max_wind_kt=wind,
                    pressure_hpa=float(pt.get("PRES", 1000)),
                    category=c_name,
                    is_forecast=False,
                )
                db.add(tp)

            # Add synthetic forecast track points
            last_ts = _parse_ts(last_pt["ISO_TIME"])
            forecast_lat, forecast_lon = float(last_pt["LAT"]), float(last_pt["LON"])
            forecast_wind = float(last_pt["WIND"])
            from datetime import timedelta
            import random
            random.seed(storm_data["YEAR"])

            for h in [6, 12, 18, 24]:
                forecast_lat += random.uniform(-0.3, 0.5)
                forecast_lon += random.uniform(0.1, 0.7)
                forecast_wind = max(20, forecast_wind - random.uniform(2, 8))
                f_ts = last_ts + timedelta(hours=h)
                f_cat_idx, f_cat_name = wind_to_category(forecast_wind)
                radius = max(50, 200 - h * 4)
                tp = TrackPoint(
                    cyclone_id=cyclone.id,
                    timestamp=f_ts,
                    lat=round(forecast_lat, 2),
                    lon=round(forecast_lon, 2),
                    max_wind_kt=round(forecast_wind, 1),
                    category=f_cat_name,
                    is_forecast=True,
                    forecast_hour=h,
                    confidence_radius_km=radius,
                )
                db.add(tp)

            # Add a synthetic prediction record
            frame = generate_frame(
                center_lat=float(first_pt["LAT"]),
                center_lon=float(first_pt["LON"]),
                max_wind_kt=float(peak_wind),
            )
            pred_data = _synthetic_inference(frame)
            pred = Prediction(
                cyclone_id=cyclone.id,
                detection_prob=pred_data["detection_prob"],
                detected=pred_data["detected"],
                center_lat=pred_data["center_lat"],
                center_lon=pred_data["center_lon"],
                category=pred_data["category"],
                category_code=pred_data["category_code"],
                intensity_trend=pred_data["intensity_trend"],
                max_wind_kt=pred_data["max_wind_kt"],
                confidence=pred_data["confidence"],
                flagged_for_review=pred_data["flagged_for_review"],
                xai_evidence=pred_data["xai_evidence"],
                track_forecast=[
                    {k: v for k, v in tp.items()} for tp in pred_data["track_forecast"]
                ],
            )
            db.add(pred)

        await db.commit()
        logger.info(f"✅ Seeded {len(BUNDLED_STORMS)} cyclones with track points and predictions.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_database())
