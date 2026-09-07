"""Stub data sources for MOSDAC and GPM — real integration TODO."""
from app.data.base import DataSource, DataFrame
from app.data.synthetic import SyntheticDataSource, CHANNELS


class MOSDACDataSource(DataSource):
    """
    Stub for ISRO MOSDAC API (INSAT-3D/3DR products).
    TODO: Implement with real MOSDAC credentials after ISRO registration.
    Endpoint: https://mosdac.gov.in/live/
    Products: SST, QPE, OLR, CMV, WVW, UTH
    """
    def __init__(self, api_key: str = "", base_url: str = "https://mosdac.gov.in"):
        self.api_key = api_key
        self.base_url = base_url
        self._fallback = SyntheticDataSource()

    @property
    def name(self) -> str:
        return "mosdac"

    def get_channels(self) -> list[str]:
        return CHANNELS

    def get_latest_frame(self) -> DataFrame:
        # TODO: Replace with real MOSDAC API call
        # response = requests.get(f"{self.base_url}/api/latest", headers={"X-API-Key": self.api_key})
        # return self._parse_mosdac_response(response.json())
        return self._fallback.get_latest_frame()

    def get_time_series(self, start: str, end: str) -> list[DataFrame]:
        # TODO: Replace with real MOSDAC time series API
        return self._fallback.get_time_series(start, end)


class GPMDataSource(DataSource):
    """
    Stub for NASA GPM IMERG rainfall data.
    TODO: Implement with NASA Earthdata login.
    Dataset: GPM_3IMERGHH v07
    """
    def __init__(self, earthdata_token: str = ""):
        self.earthdata_token = earthdata_token
        self._fallback = SyntheticDataSource()

    @property
    def name(self) -> str:
        return "gpm"

    def get_channels(self) -> list[str]:
        return ["qpe"]  # GPM provides precipitation only

    def get_latest_frame(self) -> DataFrame:
        # TODO: NASA GESDISC API call
        return self._fallback.get_latest_frame()

    def get_time_series(self, start: str, end: str) -> list[DataFrame]:
        return self._fallback.get_time_series(start, end)
