import axios from 'axios';

// Production: reads VITE_API_URL from Vercel environment variable
// Development: proxied through Vite to localhost:8000
const VITE_API = import.meta.env.VITE_API_URL as string | undefined;
const BASE = VITE_API ? `${VITE_API}/api/v1` : '/api/v1';

const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});


// ── Types ──────────────────────────────────────────────────────────────────
export interface Cyclone { id:number; name:string; year:number; source:string; category?:string; category_code?:number; max_wind_kt?:number; min_pressure_hpa?:number; current_lat?:number; current_lon?:number; confidence:number; flagged_for_review?:boolean; }
export interface TrackPoint { id:number; cyclone_id:number; timestamp:string; lat:number; lon:number; wind_kt?:number; pressure_hpa?:number; is_forecast:boolean; }
export interface IntensityPoint { timestamp:string; wind_kt?:number; pressure_hpa?:number; is_forecast:boolean; }
export interface Alert { tier:string; color:string; cyclone_name?:string; wind_kt?:number; category?:string; message:string; }
export interface HealthStatus { status:string; db_connected:boolean; model_loaded:boolean; }
export interface Prediction {
  detected:boolean; detection_prob:number; category?:string; category_code?:number;
  max_wind_kt?:number; confidence:number; center_lat?:number; center_lon?:number;
  intensity_trend?:string; intensity_probs?:number[]; wind_24h_kt?:number;
  track_forecast?:Array<{hour:number;lat:number;lon:number;wind_kt?:number;confidence_radius_km?:number}>;
  wind_forecast?:Array<{hour:number;wind_kt:number;lower:number;upper:number}>;
  cnn_class_probs?:number[]; cnn_layer_acts?:Array<{layer:string;mean_act:number;max_act?:number;n_maps?:number}>;
  cnn_feature_snippet?:number[]; preprocess_stats?:Record<string,{min:number;max:number;mean:number;std:number}>;
  xai_evidence?:string; flagged_for_review:boolean;
  pipeline_steps?:string[]; model_stack?:any; timestamp?:string;
}
export interface Metrics {
  detection:{ precision:number; recall:number; f1_score:number; auc_roc:number; true_positives:number; false_positives:number; false_negatives:number; true_negatives:number; };
  center_location:{ mean_position_error_km:number; median_position_error_km:number; within_50km_pct:number; within_110km_pct:number; };
  intensity:{ wind_mae_kt:number; wind_rmse_kt:number; pressure_mae_hpa:number; category_accuracy:number; category_macro_f1:number; };
  track_forecast:{ error_6h_km:number; error_12h_km:number; error_24h_km:number; error_48h_km:number; error_72h_km:number; };
  uncertainty:{ calibration_error:number; interval_coverage_90:number; sharpness:number; };
  operational:{ avg_latency_ms:number; p95_latency_ms:number; throughput_fps:number; data_delay_tolerance_min:number; uptime_pct:number; uptime_seconds:number; };
  system:Record<string,string>;
  training_history:Array<{epoch:number;loss:number;accuracy:number;val_loss:number;val_accuracy:number}>;
}

// ── API calls ──────────────────────────────────────────────────────────────
export const fetchHealth    = () => api.get<HealthStatus>('/health').then(r => r.data);
export const fetchCyclones  = () => api.get<Cyclone[]>('/cyclones').then(r => r.data);
export const fetchCyclone   = (id:number) => api.get<Cyclone>(`/cyclones/${id}`).then(r => r.data);
export const fetchTrack     = (id:number) => api.get<TrackPoint[]>(`/cyclones/${id}/track`).then(r => r.data);
export const fetchIntensity = (id:number) => api.get<IntensityPoint[]>(`/cyclones/${id}/intensity`).then(r => r.data);
export const fetchAlerts    = () => api.get<Alert[]>('/alerts').then(r => r.data);
export const fetchMetrics   = () => api.get<Metrics>('/metrics').then(r => r.data);
export const submitReview   = (body:any) => api.post('/review', body).then(r => r.data);
export const getReportUrl   = (id:number, fmt:string) => `/api/v1/reports/${id}/export?format=${fmt}`;

export const runPredict = (
  data_source = 'synthetic',
  cyclone_id?: number,
  use_xai = false,
  overrides?: Record<string, number>
) => api.post<Prediction>('/predict', { data_source, cyclone_id, use_xai, ...overrides }).then(r => r.data);
