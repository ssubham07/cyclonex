"""POST /api/v1/review — human-in-the-loop prediction review."""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime, timezone

from app.database import get_db
from app.models.cyclone import Review, Cyclone, Prediction
from app.schemas.cyclone import ReviewRequest, ReviewOut

router = APIRouter(prefix="/api/v1/review", tags=["review"])


@router.post("", response_model=ReviewOut, status_code=201)
async def submit_review(req: ReviewRequest, db: AsyncSession = Depends(get_db)):
    # Validate cyclone/prediction exist
    if req.cyclone_id:
        q = select(Cyclone).where(Cyclone.id == req.cyclone_id)
        r = await db.execute(q)
        if not r.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Cyclone {req.cyclone_id} not found")

    review = Review(
        cyclone_id=req.cyclone_id,
        prediction_id=req.prediction_id,
        action=req.action,
        analyst_note=req.analyst_note,
        overridden_category=req.overridden_category,
        overridden_confidence=req.overridden_confidence,
        analyst_id=req.analyst_id,
    )
    db.add(review)

    # If overriding, update the cyclone record
    if req.action == "override" and req.cyclone_id:
        updates = {}
        if req.overridden_category:
            updates["category"] = req.overridden_category
        if req.overridden_confidence is not None:
            updates["confidence"] = req.overridden_confidence
        if updates:
            updates["flagged_for_review"] = False
            await db.execute(
                update(Cyclone).where(Cyclone.id == req.cyclone_id).values(**updates)
            )

    # Accept: clear flag
    if req.action == "accept" and req.cyclone_id:
        await db.execute(
            update(Cyclone).where(Cyclone.id == req.cyclone_id).values(flagged_for_review=False)
        )

    await db.commit()
    await db.refresh(review)
    return review


@router.get("/{cyclone_id}", response_model=list[ReviewOut])
async def get_reviews(cyclone_id: int, db: AsyncSession = Depends(get_db)):
    q = select(Review).where(Review.cyclone_id == cyclone_id).order_by(Review.timestamp.desc())
    r = await db.execute(q)
    return r.scalars().all()
