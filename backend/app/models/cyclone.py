from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class Cyclone(Base):
    __tablename__ = "cyclones"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)
    year = Column(Integer)
    basin = Column(String(20), default="NI")  # North Indian Ocean
    status = Column(String(50), default="active")  # active | historical | dissipated
    category = Column(String(50), nullable=True)  # IMD category
    category_code = Column(Integer, nullable=True)  # 0-6 for 7 IMD categories
    max_wind_kt = Column(Float, nullable=True)  # max sustained wind knots
    current_lat = Column(Float, nullable=True)
    current_lon = Column(Float, nullable=True)
    min_pressure_hpa = Column(Float, nullable=True)
    confidence = Column(Float, default=0.85)  # 0-1 model confidence
    flagged_for_review = Column(Boolean, default=False)
    source = Column(String(50), default="ibtracs")  # ibtracs | synthetic | mosdac
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    extra = Column(JSON, nullable=True)

    track_points = relationship("TrackPoint", back_populates="cyclone", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="cyclone", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="cyclone", cascade="all, delete-orphan")


class TrackPoint(Base):
    __tablename__ = "track_points"

    id = Column(Integer, primary_key=True, index=True)
    cyclone_id = Column(Integer, ForeignKey("cyclones.id"), index=True)
    timestamp = Column(DateTime)
    lat = Column(Float)
    lon = Column(Float)
    max_wind_kt = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    category = Column(String(50), nullable=True)
    is_forecast = Column(Boolean, default=False)
    forecast_hour = Column(Integer, nullable=True)  # +6, +12, +18, +24
    confidence_radius_km = Column(Float, nullable=True)  # uncertainty cone radius

    cyclone = relationship("Cyclone", back_populates="track_points")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    cyclone_id = Column(Integer, ForeignKey("cyclones.id"), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    detection_prob = Column(Float)  # 0-1
    detected = Column(Boolean)
    center_lat = Column(Float, nullable=True)
    center_lon = Column(Float, nullable=True)
    category = Column(String(50), nullable=True)
    category_code = Column(Integer, nullable=True)
    intensity_trend = Column(String(20), nullable=True)  # Intensifying | Steady | Weakening
    max_wind_kt = Column(Float, nullable=True)
    confidence = Column(Float)
    flagged_for_review = Column(Boolean, default=False)
    xai_image_path = Column(String(255), nullable=True)
    xai_evidence = Column(Text, nullable=True)
    track_forecast = Column(JSON, nullable=True)  # [{hour: 6, lat, lon, wind_kt}, ...]
    raw_output = Column(JSON, nullable=True)

    cyclone = relationship("Cyclone", back_populates="predictions")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    cyclone_id = Column(Integer, ForeignKey("cyclones.id"), nullable=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    action = Column(String(20))  # accept | override | annotate
    analyst_note = Column(Text, nullable=True)
    overridden_category = Column(String(50), nullable=True)
    overridden_confidence = Column(Float, nullable=True)
    analyst_id = Column(String(100), default="analyst@imd.gov.in")

    cyclone = relationship("Cyclone", back_populates="reviews")
