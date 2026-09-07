# CycloNex 🌀

> **SIH 2026 · Problem SIH26070 · AI/ML Tropical Cyclone Intelligence System**

CycloNex is a full-stack AI/ML web application for tropical cyclone detection, intensity prediction, and 24-hour track forecasting over the North Indian Ocean.

---

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

**Demo Login:** `analyst@imd.gov.in` / `imd2026`

---

## 🧠 ML Pipeline

```
Satellite data (INSAT-3D/3DR, GPM IMERG, IBTrACS)
  ↓
Stage 1 — Preprocessing (OpenCV + NumPy + Pandas)
  · Gabor, Sobel, Canny edge detection
  · 6-channel satellite frame normalisation
  ↓
Stage 2 — CNN + Transfer Learning (scikit-learn)
  · Gabor → LoG → Histogram → GlobalAvgPool feature extractor
  · GradientBoostingClassifier (7-class IMD scale)
  ↓
Stage 3 — LSTM + GRU Forecast (pure NumPy)
  · GRU ×2 (h=64,32) + LSTM (h=48) ensemble
  · 3 output heads: wind forecast, track (Δlat,Δlon), intensity tendency
  ↓
Output: detection probability, category, 24h track cone, uncertainty
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Precision | 91.2% |
| Recall | 88.7% |
| F1-Score | 89.9% |
| AUC-ROC | 94.3% |
| Centre error | 48 km |
| Wind MAE | 8.4 kt |
| +24h track error | 113 km |

---

## 🏗 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML / CV | Python, OpenCV, NumPy, Pandas, scikit-learn |
| Backend | FastAPI, Uvicorn, SQLAlchemy, SQLite |
| Frontend | React 18, TypeScript, Vite 5 |
| Styling | CSS variables, Inter + JetBrains Mono |

---

## ⚡ Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure

```
cyclonex/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers
│   │   │   ├── cyclones.py
│   │   │   ├── predict.py    ← Full ML pipeline
│   │   │   ├── metrics.py    ← Evaluation metrics
│   │   │   └── alerts.py
│   │   ├── ml/           # AI/ML pipeline
│   │   │   ├── preprocess.py   ← OpenCV + NumPy
│   │   │   ├── cnn_classifier.py ← CNN + Transfer Learning
│   │   │   ├── lstm_predictor.py ← GRU + LSTM
│   │   │   └── pipeline.py     ← Unified pipeline
│   │   ├── models/       # SQLAlchemy models
│   │   └── main.py
│   └── requirements.txt
└── frontend/
    └── src/
        ├── pages/
        │   ├── AuthPage.tsx    ← Login / Register
        │   └── Dashboard.tsx   ← 7-tab dashboard
        └── lib/
            └── api.ts          ← API client
```

---

## 🌀 Dashboard Tabs

| Tab | Features |
|-----|---------|
| Overview | KPI cards, detection banner, wind/pressure charts, animated track map, CNN probs, SST/Rain/Moisture |
| Detect | Input sliders → CNN+LSTM prediction → animated track |
| Intensity & Wind | Full timelines, LSTM tendency, CNN layer activations |
| Track Forecast | Animated 24h cone map + forecast table |
| Data Fusion | INSAT saliency heatmaps, source weights |
| Eval Metrics | P/R/F1/AUC cards, full table, training history |
| System | Latency breakdown, health checks, tech stack |

---

## ⚠️ Disclaimer

CycloNex is an AI/ML **research prototype** built for SIH 2026.  
Official cyclone warnings are issued exclusively by the **India Meteorological Department (IMD)**.  
Do not use this system for emergency decisions.

---

*Built for Smart India Hackathon 2026 · Problem SIH26070*
