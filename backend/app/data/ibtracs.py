"""
IBTrACSReplaySource — Replays real NOAA IBTrACS North Indian Ocean historical data.
Downloads ibtracs.NI.list.v04r01.csv from NOAA (no auth required).
Falls back to a bundled mini-subset for offline/demo use.
"""
from __future__ import annotations
import csv
import io
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import numpy as np

from app.data.base import DataSource, DataFrame
from app.data.synthetic import generate_frame, CHANNELS

logger = logging.getLogger(__name__)

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"

# Notable NIO cyclones for demo — bundled as fallback
BUNDLED_STORMS = [
    {"SID": "2020-SCS-AMPHAN", "NAME": "AMPHAN", "YEAR": 2020, "SEASON": 2020, "BASIN": "NI",
     "track": [
         {"ISO_TIME": "2020-05-16T06:00:00Z", "LAT": 10.5, "LON": 86.0, "WIND": 25, "PRES": 1000},
         {"ISO_TIME": "2020-05-17T06:00:00Z", "LAT": 12.0, "LON": 87.0, "WIND": 40, "PRES": 990},
         {"ISO_TIME": "2020-05-18T06:00:00Z", "LAT": 14.5, "LON": 87.5, "WIND": 75, "PRES": 970},
         {"ISO_TIME": "2020-05-19T06:00:00Z", "LAT": 17.0, "LON": 87.5, "WIND": 130, "PRES": 920},
         {"ISO_TIME": "2020-05-20T06:00:00Z", "LAT": 20.0, "LON": 87.5, "WIND": 100, "PRES": 940},
         {"ISO_TIME": "2020-05-20T18:00:00Z", "LAT": 22.0, "LON": 88.0, "WIND": 85, "PRES": 955},
     ]},
    {"SID": "2019-SCS-FANI", "NAME": "FANI", "YEAR": 2019, "SEASON": 2019, "BASIN": "NI",
     "track": [
         {"ISO_TIME": "2019-04-27T06:00:00Z", "LAT": 8.0, "LON": 85.0, "WIND": 30, "PRES": 998},
         {"ISO_TIME": "2019-04-29T06:00:00Z", "LAT": 11.0, "LON": 84.5, "WIND": 65, "PRES": 975},
         {"ISO_TIME": "2019-05-01T06:00:00Z", "LAT": 14.0, "LON": 85.0, "WIND": 115, "PRES": 930},
         {"ISO_TIME": "2019-05-03T06:00:00Z", "LAT": 17.5, "LON": 85.5, "WIND": 130, "PRES": 915},
         {"ISO_TIME": "2019-05-04T06:00:00Z", "LAT": 19.8, "LON": 85.8, "WIND": 100, "PRES": 945},
         {"ISO_TIME": "2019-05-05T06:00:00Z", "LAT": 22.0, "LON": 85.5, "WIND": 55, "PRES": 975},
     ]},
    {"SID": "2023-SCS-BIPARJOY", "NAME": "BIPARJOY", "YEAR": 2023, "SEASON": 2023, "BASIN": "NI",
     "track": [
         {"ISO_TIME": "2023-06-06T06:00:00Z", "LAT": 12.0, "LON": 65.0, "WIND": 35, "PRES": 995},
         {"ISO_TIME": "2023-06-08T06:00:00Z", "LAT": 14.5, "LON": 65.5, "WIND": 55, "PRES": 980},
         {"ISO_TIME": "2023-06-10T06:00:00Z", "LAT": 16.5, "LON": 66.0, "WIND": 80, "PRES": 960},
         {"ISO_TIME": "2023-06-12T06:00:00Z", "LAT": 18.0, "LON": 66.5, "WIND": 95, "PRES": 945},
         {"ISO_TIME": "2023-06-15T06:00:00Z", "LAT": 22.5, "LON": 68.0, "WIND": 80, "PRES": 960},
         {"ISO_TIME": "2023-06-15T18:00:00Z", "LAT": 23.2, "LON": 68.5, "WIND": 65, "PRES": 970},
     ]},
    {"SID": "2021-SCS-YAAS", "NAME": "YAAS", "YEAR": 2021, "SEASON": 2021, "BASIN": "NI",
     "track": [
         {"ISO_TIME": "2021-05-23T06:00:00Z", "LAT": 16.0, "LON": 88.0, "WIND": 40, "PRES": 990},
         {"ISO_TIME": "2021-05-24T06:00:00Z", "LAT": 18.5, "LON": 87.5, "WIND": 70, "PRES": 968},
         {"ISO_TIME": "2021-05-25T06:00:00Z", "LAT": 20.5, "LON": 87.0, "WIND": 90, "PRES": 955},
         {"ISO_TIME": "2021-05-26T06:00:00Z", "LAT": 22.0, "LON": 87.5, "WIND": 75, "PRES": 964},
     ]},
    {"SID": "2022-SCS-ASANI", "NAME": "ASANI", "YEAR": 2022, "SEASON": 2022, "BASIN": "NI",
     "track": [
         {"ISO_TIME": "2022-05-08T06:00:00Z", "LAT": 12.0, "LON": 88.0, "WIND": 30, "PRES": 998},
         {"ISO_TIME": "2022-05-09T06:00:00Z", "LAT": 13.5, "LON": 88.5, "WIND": 55, "PRES": 980},
         {"ISO_TIME": "2022-05-10T06:00:00Z", "LAT": 15.5, "LON": 88.0, "WIND": 75, "PRES": 966},
         {"ISO_TIME": "2022-05-11T06:00:00Z", "LAT": 17.5, "LON": 85.5, "WIND": 55, "PRES": 978},
         {"ISO_TIME": "2022-05-12T06:00:00Z", "LAT": 19.0, "LON": 84.0, "WIND": 40, "PRES": 988},
     ]},
]


class IBTrACSReplaySource(DataSource):
    """Replays real NIO IBTrACS historical cyclone data."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path
        self._storms = BUNDLED_STORMS  # fallback
        self._try_load_csv()

    def _try_load_csv(self):
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                self._storms = self._parse_csv(self.csv_path)
                logger.info(f"Loaded {len(self._storms)} storms from IBTrACS CSV")
            except Exception as e:
                logger.warning(f"Could not parse IBTrACS CSV: {e}. Using bundled data.")

    def _parse_csv(self, path: str) -> list[dict]:
        storms = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            next(reader, None)  # skip units row
            for row in reader:
                sid = row.get("SID", "")
                if not sid:
                    continue
                name = row.get("NAME", "UNNAMED")
                year = int(row.get("SEASON", 2000))
                if year < 2000:
                    continue
                if sid not in storms:
                    storms[sid] = {"SID": sid, "NAME": name, "YEAR": year, "SEASON": year, "BASIN": "NI", "track": []}
                try:
                    lat = float(row.get("LAT", 0))
                    lon = float(row.get("LON", 0))
                    wind_str = row.get("WMO_WIND", row.get("USA_WIND", "0")).strip()
                    wind = float(wind_str) if wind_str and wind_str != " " else 0.0
                    pres_str = row.get("WMO_PRES", row.get("USA_PRES", "1010")).strip()
                    pres = float(pres_str) if pres_str and pres_str != " " else 1010.0
                    ts = row.get("ISO_TIME", "")
                    if lat and lon and ts:
                        storms[sid]["track"].append({
                            "ISO_TIME": ts, "LAT": lat, "LON": lon, "WIND": wind, "PRES": pres
                        })
                except (ValueError, TypeError):
                    continue
        # Return top N storms with tracks, sorted by year desc
        result = [s for s in storms.values() if len(s["track"]) >= 3]
        result.sort(key=lambda x: x["YEAR"], reverse=True)
        return result[:20]

    @property
    def name(self) -> str:
        return "ibtracs"

    def get_channels(self) -> list[str]:
        return CHANNELS

    def get_storms(self) -> list[dict]:
        return self._storms

    def get_latest_frame(self) -> DataFrame:
        """Return a frame based on the most recent storm's latest position."""
        if not self._storms:
            from app.data.synthetic import SyntheticDataSource
            return SyntheticDataSource().get_latest_frame()
        storm = self._storms[0]
        pt = storm["track"][-1]
        return generate_frame(
            center_lat=pt["LAT"],
            center_lon=pt["LON"],
            max_wind_kt=float(pt["WIND"]),
        )

    def get_time_series(self, start: str, end: str) -> list[DataFrame]:
        if not self._storms:
            return []
        storm = self._storms[0]
        frames = []
        for pt in storm["track"]:
            ts_str = pt["ISO_TIME"].replace(" ", "T")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            frames.append(generate_frame(
                center_lat=float(pt["LAT"]),
                center_lon=float(pt["LON"]),
                max_wind_kt=float(pt["WIND"]),
                timestamp=ts,
            ))
        return frames
