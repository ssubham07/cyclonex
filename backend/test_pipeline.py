import sys, os
sys.path.insert(0, '.')
import numpy as np

print("Testing preprocess...")
from app.ml.preprocess import preprocess
f = preprocess(15.0, 82.0, 0.72)
print(f"  patches={len(f.patches)}, channels={list(f.raw_channels.keys())}")

print("Testing CNN...")
from app.ml.cnn_classifier import get_cnn_model
cnn = get_cnn_model()
patch = np.random.randn(6,64,64).astype('float32')
pred = cnn.predict(patch)
print(f"  cat={pred.category_name} wind={pred.max_wind_kt:.0f}kt conf={pred.confidence:.2f}")

print("Testing LSTM...")
from app.ml.lstm_predictor import get_lstm_predictor, CycloneLSTMPredictor
lstm = get_lstm_predictor()
seq = CycloneLSTMPredictor.synthetic_sequence(90.0, 15.0, 82.0, 1.04)
lp = lstm.predict(seq)
wf = lp.wind_forecasts[-1]
print(f"  trend={lp.intensity_tendency} +24h={wf['wind_kt']}kt")

print("Testing full pipeline...")
from app.ml.pipeline import run_pipeline
r = run_pipeline(15.0, 82.0, 0.72)
print(f"  detected={r.detected} cat={r.category} wind={r.max_wind_kt:.0f}kt")
print(f"  track +24h: {r.track_forecast[-1]}")
print(f"  steps: {len(r.pipeline_steps)} log lines")
print("ALL OK")
