import sys
sys.path.insert(0, '.')

from app.config import settings
from app.database import Base
from app.models.cyclone import Cyclone, TrackPoint, Prediction, Review
from app.schemas.cyclone import IMD_CATEGORIES, wind_to_category
from app.data.synthetic import SyntheticDataSource
from app.data.ibtracs import IBTrACSReplaySource, BUNDLED_STORMS
from app.data.mosdac import MOSDACDataSource
from app.ml.model import build_model
from app.ml.inference import _synthetic_inference, CHANNELS_ORDER
from app.ml.gradcam import generate_evidence_caption

print('All imports OK')
print('Channels:', CHANNELS_ORDER)
print('IMD categories:', len(IMD_CATEGORIES))
print('Bundled storms:', len(BUNDLED_STORMS))

m = build_model()
params = sum(p.numel() for p in m.parameters())
print('Model params:', params)

ds = SyntheticDataSource()
frame = ds.get_latest_frame()
print('Frame channels:', list(frame.channels.keys()))
sst_shape = frame.channels['sst'].shape
print('SST shape:', sst_shape)

pred = _synthetic_inference(frame)
print('Synthetic inference:', pred['detected'], round(pred['confidence'], 2))

ibt = IBTrACSReplaySource()
storms = ibt.get_storms()
names = [s['NAME'] for s in storms]
print('IBTrACS storms:', names)

cat, name = wind_to_category(75.0)
print('Category for 75kt:', cat, name)

print()
print('ALL BACKEND MODULES VERIFIED')
