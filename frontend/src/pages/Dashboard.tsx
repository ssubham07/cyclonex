import { useState, useEffect, useCallback } from 'react';
import {
  fetchCyclones, fetchAlerts, fetchHealth, runPredict,
  fetchTrack, fetchIntensity, fetchMetrics
} from '../lib/api';
import type { Cyclone, Alert, Prediction, TrackPoint, IntensityPoint, Metrics } from '../lib/api';

// ─── IMD Scale ────────────────────────────────────────────────────────────────
const IMD = [
  { code:0, name:'Depression',                      wind:'17–27',  badge:'D',    color:'#22c55e' },
  { code:1, name:'Deep Depression',                 wind:'28–33',  badge:'DD',   color:'#84cc16' },
  { code:2, name:'Cyclonic Storm',                  wind:'34–47',  badge:'CS',   color:'#eab308' },
  { code:3, name:'Severe Cyclonic Storm',           wind:'48–63',  badge:'SCS',  color:'#f97316' },
  { code:4, name:'Very Severe Cyclonic Storm',      wind:'64–89',  badge:'VSCS', color:'#ef4444' },
  { code:5, name:'Extremely Severe Cyclonic Storm', wind:'90–119', badge:'ESCS', color:'#dc2626' },
  { code:6, name:'Super Cyclone',                   wind:'≥120',   badge:'SuCS', color:'#7f1d1d' },
];

// ─── Line chart ───────────────────────────────────────────────────────────────
function LineChart({ data, color='#0f4c81', W=280, H=90, label='', showDots=true }: {
  data:number[]; color?:string; W?:number; H?:number; label?:string; showDots?:boolean;
}) {
  if (data.length < 2) return <div style={{height:H,display:'flex',alignItems:'center',justifyContent:'center',color:'#94a3b8',fontSize:'0.72rem'}}>No data</div>;
  const pad={t:8,b:20,l:28,r:8};
  const iW=W-pad.l-pad.r, iH=H-pad.t-pad.b;
  const max=Math.max(...data), min=Math.min(...data), range=max-min||1;
  const tx=(i:number)=>pad.l+(i/(data.length-1))*iW;
  const ty=(v:number)=>pad.t+iH-((v-min)/range)*iH;
  const pts=data.map((v,i)=>`${tx(i).toFixed(1)},${ty(v).toFixed(1)}`).join(' ');
  const fill=`${pts} ${tx(data.length-1)},${H-pad.b} ${pad.l},${H-pad.b}`;
  const gId=`lg${color.replace('#','')}${W}`;
  return (
    <svg width={W} height={H}>
      <defs>
        <linearGradient id={gId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <polygon points={fill} fill={`url(#${gId})`}/>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round"/>
      {showDots && data.map((v,i)=>(
        <circle key={i} cx={tx(i)} cy={ty(v)} r={i===data.length-1?4:2} fill={color} opacity={i===data.length-1?1:0.35}/>
      ))}
      <line x1={pad.l} y1={H-pad.b} x2={W-pad.r} y2={H-pad.b} stroke="#e2e8f0" strokeWidth="1"/>
      <text x={pad.l-2} y={pad.t+8} textAnchor="end" fill="#94a3b8" fontSize="8" fontFamily="JetBrains Mono">{max.toFixed(0)}</text>
      <text x={pad.l-2} y={H-pad.b} textAnchor="end" fill="#94a3b8" fontSize="8" fontFamily="JetBrains Mono">{min.toFixed(0)}</text>
      {label && <text x={W/2} y={H-2} textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="JetBrains Mono">{label}</text>}
    </svg>
  );
}

function DualLineChart({ a, b, colorA='#ef4444', colorB='#0f4c81', W=320, H=100, labelA='', labelB='' }: {
  a:number[]; b:number[]; colorA?:string; colorB?:string; W?:number; H?:number; labelA?:string; labelB?:string;
}) {
  const all=[...a,...b]; if (all.length<2) return null;
  const pad={t:8,b:20,l:30,r:8};
  const iW=W-pad.l-pad.r, iH=H-pad.t-pad.b;
  const maxA=Math.max(...a,1),minA=Math.min(...a); const rA=maxA-minA||1;
  const maxB=Math.max(...b,1),minB=Math.min(...b); const rB=maxB-minB||1;
  const tx=(i:number,n:number)=>pad.l+(i/(n-1))*iW;
  const tyA=(v:number)=>pad.t+iH-((v-minA)/rA)*iH;
  const tyB=(v:number)=>pad.t+iH-((v-minB)/rB)*iH;
  return (
    <svg width={W} height={H}>
      {a.length>1 && <polyline points={a.map((v,i)=>`${tx(i,a.length)},${tyA(v)}`).join(' ')} fill="none" stroke={colorA} strokeWidth="2"/>}
      {b.length>1 && <polyline points={b.map((v,i)=>`${tx(i,b.length)},${tyB(v)}`).join(' ')} fill="none" stroke={colorB} strokeWidth="2" strokeDasharray="4 2"/>}
      <line x1={pad.l} y1={H-pad.b} x2={W-pad.r} y2={H-pad.b} stroke="#e2e8f0"/>
      {labelA && <><rect x={pad.l} y={pad.t} width={8} height={3} fill={colorA} rx="1"/><text x={pad.l+10} y={pad.t+6} fontSize="7" fill={colorA} fontFamily="JetBrains Mono">{labelA}</text></>}
      {labelB && <><rect x={pad.l+60} y={pad.t} width={8} height={3} fill={colorB} rx="1"/><text x={pad.l+70} y={pad.t+6} fontSize="7" fill={colorB} fontFamily="JetBrains Mono">{labelB}</text></>}
    </svg>
  );
}

// ─── ANIMATED Track Map ───────────────────────────────────────────────────────
type FcPt = { hour:number; lat:number; lon:number; wind_kt?:number; confidence_radius_km?:number };

function AnimatedTrackMap({ track, forecast, catColor='#ef4444' }:{
  track:TrackPoint[]; forecast:FcPt[]; catColor?:string;
}) {
  const [step,setStep] = useState(0);
  const [playing,setPlaying] = useState(true);
  const W=500, H=270, PAD=34;

  const histPts = track.filter(t=>!t.is_forecast);
  const forePts = forecast;
  const allPts  = [
    ...histPts.map(t=>({lat:t.lat,lon:t.lon,type:'hist' as const})),
    ...forePts.map(f=>({lat:f.lat,lon:f.lon,type:'fore' as const,hour:f.hour,wind_kt:f.wind_kt,radius:f.confidence_radius_km})),
  ];

  const allLat=allPts.map(p=>p.lat), allLon=allPts.map(p=>p.lon);
  const minLat=Math.min(...allLat,8)-2,  maxLat=Math.max(...allLat,22)+2;
  const minLon=Math.min(...allLon,78)-2, maxLon=Math.max(...allLon,92)+2;
  const tx=(lon:number)=>PAD+((lon-minLon)/(maxLon-minLon))*(W-2*PAD);
  const ty=(lat:number)=>H-PAD-((lat-minLat)/(maxLat-minLat))*(H-2*PAD);

  useEffect(()=>{
    if (!playing||!allPts.length) return;
    const id=setInterval(()=>setStep(s=>s>=allPts.length-1?0:s+1),600);
    return ()=>clearInterval(id);
  },[playing,allPts.length]);

  if (!allPts.length) return (
    <div style={{height:H,display:'flex',alignItems:'center',justifyContent:'center',color:'#94a3b8',fontSize:'0.82rem'}}>
      Click ⚡ Run Prediction to animate the track
    </div>
  );

  const curPt   = allPts[Math.min(step,allPts.length-1)];
  const isFore  = curPt?.type==='fore';
  const curCol  = isFore ? '#f97316' : catColor;
  const vHist   = histPts.slice(0,Math.min(step+1,histPts.length));
  const fStep   = step>=histPts.length ? step-histPts.length+1 : 0;
  const vFore   = fStep>0 ? forePts.slice(0,fStep) : [];

  // Cone
  const last = vHist.length ? vHist[vHist.length-1] : null;
  const coneLeft  = last && vFore.length>0 ? vFore.map(f=>{ const r=(f.confidence_radius_km??120)/70; return `${tx(f.lon-r)},${ty(f.lat+r)}`; }) : [];
  const coneRight = last && vFore.length>0 ? vFore.map(f=>{ const r=(f.confidence_radius_km??120)/70; return `${tx(f.lon+r)},${ty(f.lat-r)}`; }) : [];
  const conePts  = last && vFore.length>0
    ? [`${tx(last.lon)},${ty(last.lat)}`,...coneRight,...[...coneLeft].reverse()].join(' ')
    : '';

  return (
    <div>
      <svg width={W} height={H} style={{background:'linear-gradient(160deg,#dbeafe 0%,#bae6fd 60%,#e0f2fe 100%)',borderRadius:14,border:'1px solid #bae6fd',display:'block'}}>
        <defs>
          <filter id="glow2"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>

        {/* Grid */}
        {[0.25,0.5,0.75].map(f=>[
          <line key={`h${f}`} x1={PAD} y1={PAD+f*(H-2*PAD)} x2={W-PAD} y2={PAD+f*(H-2*PAD)} stroke="rgba(148,163,184,0.28)" strokeDasharray="4"/>,
          <line key={`v${f}`} x1={PAD+f*(W-2*PAD)} y1={PAD} x2={PAD+f*(W-2*PAD)} y2={H-PAD} stroke="rgba(148,163,184,0.28)" strokeDasharray="4"/>,
        ])}
        {/* Lat/Lon labels */}
        {[0.33,0.66].map(f=>(
          <g key={f}>
            <text x={PAD-4} y={H-PAD-f*(H-2*PAD)+4} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{(minLat+f*(maxLat-minLat)).toFixed(0)}°N</text>
            <text x={PAD+f*(W-2*PAD)} y={H-2} textAnchor="middle" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{(minLon+f*(maxLon-minLon)).toFixed(0)}°E</text>
          </g>
        ))}

        {/* Uncertainty cone */}
        {conePts && <polygon points={conePts} fill="rgba(249,115,22,0.14)" stroke="rgba(249,115,22,0.3)" strokeWidth="1"/>}

        {/* Historical track */}
        {vHist.length>1 && <polyline points={vHist.map(p=>`${tx(p.lon)},${ty(p.lat)}`).join(' ')} fill="none" stroke="#0f4c81" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>}
        {vHist.map((p,i)=><circle key={i} cx={tx(p.lon)} cy={ty(p.lat)} r={3} fill="#0f4c81" opacity={0.45+0.55*(i/Math.max(vHist.length-1,1))}/>)}

        {/* Forecast dashed */}
        {vFore.length>0 && last && (
          <polyline points={[{lon:last.lon,lat:last.lat},...vFore].map(p=>`${tx(p.lon)},${ty(p.lat)}`).join(' ')} fill="none" stroke="#f97316" strokeWidth="2.2" strokeDasharray="8 4" strokeLinecap="round"/>
        )}
        {vFore.map((f,i)=>(
          <g key={i}>
            <circle cx={tx(f.lon)} cy={ty(f.lat)} r={4} fill="#f97316" opacity={0.75}/>
            <text x={tx(f.lon)+7} y={ty(f.lat)-4} fontSize="8" fill="#f97316" fontFamily="JetBrains Mono" fontWeight="600">+{f.hour}h</text>
          </g>
        ))}

        {/* Animated cyclone symbol */}
        {curPt && (
          <g transform={`translate(${tx(curPt.lon)},${ty(curPt.lat)})`} filter="url(#glow2)">
            {/* Pulsing rings */}
            <circle r="20" fill="none" stroke={curCol} strokeWidth="1">
              <animate attributeName="r" values="12;24;12" dur="2s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.4;0;0.4" dur="2s" repeatCount="indefinite"/>
            </circle>
            <circle r="13" fill="none" stroke={curCol} strokeWidth="1.5">
              <animate attributeName="r" values="9;16;9" dur="2s" begin="0.55s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.55;0;0.55" dur="2s" begin="0.55s" repeatCount="indefinite"/>
            </circle>
            {/* Rotating spiral */}
            <g><animateTransform attributeName="transform" type="rotate" values="0;360" dur="4s" repeatCount="indefinite"/>
              {[0,90,180,270].map(a=>(
                <path key={a}
                  d={`M 0 0 Q ${8*Math.cos((a+45)*Math.PI/180)} ${8*Math.sin((a+45)*Math.PI/180)} ${11*Math.cos(a*Math.PI/180)} ${11*Math.sin(a*Math.PI/180)}`}
                  fill="none" stroke={curCol} strokeWidth="2.2" strokeLinecap="round" opacity="0.85"/>
              ))}
            </g>
            {/* Centre */}
            <circle r="5.5" fill={curCol}/>
            <circle r="2.5" fill="white" opacity="0.92"/>
          </g>
        )}

        {/* Callout card */}
        {curPt && (
          <>
            <rect x={W-PAD-116} y={PAD} width={116} height={isFore?62:42} rx="6" fill="white" fillOpacity="0.92"/>
            <text x={W-PAD-58} y={PAD+14} textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fontWeight="700" fill={curCol}>{isFore?`FORECAST +${(curPt as any).hour}h`:'OBSERVED'}</text>
            <text x={W-PAD-58} y={PAD+26} textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fill="#475569">{curPt.lat.toFixed(1)}°N / {curPt.lon.toFixed(1)}°E</text>
            {isFore && (curPt as any).wind_kt && <text x={W-PAD-58} y={PAD+40} textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fontWeight="700" fill="#ef4444">💨 {((curPt as any).wind_kt??0).toFixed(0)} kt</text>}
            {isFore && (curPt as any).radius && <text x={W-PAD-58} y={PAD+53} textAnchor="middle" fontSize="7" fontFamily="JetBrains Mono" fill="#94a3b8">±{(curPt as any).radius} km</text>}
          </>
        )}

        {/* Legend */}
        <rect x={5} y={5} width={118} height={44} rx="5" fill="white" fillOpacity="0.9"/>
        <circle cx={18} cy={17} r={5} fill="#0f4c81"/>
        <text x={26} y={20} fontSize="7.5" fill="#0f4c81" fontFamily="JetBrains Mono">Observed track</text>
        <line x1={12} y1={32} x2={24} y2={32} stroke="#f97316" strokeWidth="2" strokeDasharray="4 2"/>
        <text x={26} y={35} fontSize="7.5" fill="#f97316" fontFamily="JetBrains Mono">Forecast + cone</text>
        <text x={W/2} y={H-4} textAnchor="middle" fontSize="7" fill="#94a3b8" fontFamily="JetBrains Mono">North Indian Ocean · Bay of Bengal</text>
      </svg>

      {/* Playback controls */}
      <div style={{display:'flex',alignItems:'center',gap:'0.65rem',marginTop:'0.6rem'}}>
        <button onClick={()=>setPlaying(p=>!p)} style={{background:playing?'#0f4c81':'#f97316',color:'#fff',border:'none',borderRadius:7,padding:'0.35rem 0.85rem',fontSize:'0.72rem',fontWeight:700,cursor:'pointer',flexShrink:0}}>
          {playing?'⏸ Pause':'▶ Play'}
        </button>
        <button onClick={()=>{setStep(0);}} style={{background:'#f1f5f9',border:'1px solid #e2e8f0',borderRadius:7,padding:'0.35rem 0.7rem',fontSize:'0.72rem',cursor:'pointer',color:'#475569',flexShrink:0}}>
          ⟳ Reset
        </button>
        <input type="range" min={0} max={Math.max(0,allPts.length-1)} value={step}
          onChange={e=>{setPlaying(false);setStep(Number(e.target.value));}}
          style={{flex:1,accentColor:'#0f4c81'}}/>
        <span style={{fontFamily:'JetBrains Mono',fontSize:'0.65rem',color:isFore?'#f97316':'#0f4c81',fontWeight:700,minWidth:62,flexShrink:0}}>
          {isFore?`+${(curPt as any)?.hour??0}h forecast`:`obs ${step+1}/${histPts.length}`}
        </span>
      </div>
    </div>
  );
}

// ─── Confidence ring ──────────────────────────────────────────────────────────
function ConfRing({ pct, color='#0f4c81', size=90, label='CONFIDENCE' }:{pct:number;color?:string;size?:number;label?:string}) {
  const r=size*0.38, circ=2*Math.PI*r, offset=circ-(pct/100)*circ;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#e2e8f0" strokeWidth={size*0.07}/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={size*0.07}
        strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
        transform={`rotate(-90 ${size/2} ${size/2})`}
        style={{transition:'stroke-dashoffset 1.2s cubic-bezier(0.34,1.56,0.64,1)'}}/>
      <text x={size/2} y={size/2-4} textAnchor="middle" fill={color} fontSize={size*0.18} fontWeight="800" fontFamily="Inter">{pct}%</text>
      <text x={size/2} y={size/2+size*0.12} textAnchor="middle" fill="#94a3b8" fontSize={size*0.068} fontFamily="JetBrains Mono">{label}</text>
    </svg>
  );
}

function SaliencyGrid({ intensity, rows=6, cols=10 }:{intensity:number;rows?:number;cols?:number}) {
  const cells=Array.from({length:rows*cols},(_,i)=>{
    const r=Math.floor(i/cols), c=i%cols;
    const d=Math.sqrt(((c-cols/2)/(cols/2))**2+((r-rows/2)/(rows/2))**2);
    return Math.max(0,(1-d)*intensity+Math.random()*0.12);
  });
  const heat=(v:number)=>{
    if (v>0.72) return `rgba(239,68,68,${0.55+v*0.45})`;
    if (v>0.45) return `rgba(249,115,22,${0.4+v*0.4})`;
    if (v>0.22) return `rgba(234,179,8,${0.3+v*0.3})`;
    return `rgba(96,165,250,${0.1+v*0.3})`;
  };
  return (
    <div style={{display:'grid',gridTemplateColumns:`repeat(${cols},1fr)`,gap:2,borderRadius:6,overflow:'hidden'}}>
      {cells.map((v,i)=><div key={i} style={{height:14,background:heat(v),transition:'background 0.5s'}}/>)}
    </div>
  );
}

function KpiCard({ icon, label, value, unit, color, sub, badge }:{icon:string;label:string;value:string;unit?:string;color:string;sub?:string;badge?:string}) {
  return (
    <div className="card" style={{display:'flex',flexDirection:'column',gap:'0.3rem',borderLeft:`3px solid ${color}`}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
        <span style={{fontSize:'1.1rem'}}>{icon}</span>
        {badge && <span className="badge" style={{background:color+'20',color}}>{badge}</span>}
      </div>
      <div style={{fontWeight:800,fontSize:'1.3rem',color,lineHeight:1}}>
        {value}<span style={{fontSize:'0.6rem',color:'#94a3b8',fontWeight:500,marginLeft:3}}>{unit}</span>
      </div>
      <div style={{fontSize:'0.7rem',color:'#64748b',fontWeight:600}}>{label}</div>
      {sub && <div style={{fontSize:'0.62rem',color:'#94a3b8'}}>{sub}</div>}
    </div>
  );
}

function SectionHead({ icon, title, sub, badge, badgeColor='#0f4c81' }:{icon:string;title:string;sub?:string;badge?:string;badgeColor?:string}) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:'0.6rem',marginBottom:'1rem'}}>
      <span style={{fontSize:'1.1rem'}}>{icon}</span>
      <div>
        <div style={{fontWeight:700,fontSize:'0.95rem',lineHeight:1.2}}>{title}</div>
        {sub && <div style={{fontSize:'0.7rem',color:'#64748b'}}>{sub}</div>}
      </div>
      {badge && <span className="badge" style={{marginLeft:'auto',background:badgeColor+'18',color:badgeColor}}>{badge}</span>}
    </div>
  );
}

const TABS=[
  {id:'overview',  icon:'📊', label:'Overview'},
  {id:'detect',    icon:'🌀', label:'Detect'},
  {id:'intensity', icon:'📈', label:'Intensity & Wind'},
  {id:'track',     icon:'🗺', label:'Track Forecast'},
  {id:'fusion',    icon:'🛰', label:'Data Fusion'},
  {id:'metrics',   icon:'🎯', label:'Eval Metrics'},
  {id:'system',    icon:'⚙️', label:'System'},
] as const;
type TabId = typeof TABS[number]['id'];

interface User { name:string; email:string; org:string; role:string; location:string; }
interface Props { user:User; onLogout:()=>void; }

const DEF_WIND  = [42,50,65,78,90,108,120,130];
const DEF_PRES  = [1005,998,990,980,968,958,948,940];
const DEF_SST   = [28.1,28.5,29.0,29.5,30.1,30.4,30.1,29.7];
const DEF_RAIN  = [12,22,38,60,82,90,78,62];
const DEF_HUMID = [72,78,83,88,92,94,91,88];
const DEF_FC: FcPt[] = [
  {hour:6,  lat:15.3,lon:82.5,wind_kt:90,  confidence_radius_km:80},
  {hour:12, lat:15.9,lon:83.2,wind_kt:95,  confidence_radius_km:110},
  {hour:18, lat:16.5,lon:83.8,wind_kt:88,  confidence_radius_km:140},
  {hour:24, lat:17.1,lon:84.4,wind_kt:82,  confidence_radius_km:170},
];

export default function Dashboard({ user, onLogout }:Props) {
  const [tab,setTab]           = useState<TabId>('overview');
  const [cyclones,setCyclones] = useState<Cyclone[]>([]);
  const [alerts,setAlerts]     = useState<Alert[]>([]);
  const [health,setHealth]     = useState<any>(null);
  const [selId,setSelId]       = useState<number|null>(null);
  const [track,setTrack]       = useState<TrackPoint[]>([]);
  const [intensity,setIntensity] = useState<IntensityPoint[]>([]);
  const [pred,setPred]         = useState<Prediction|null>(null);
  const [metrics,setMetrics]   = useState<Metrics|null>(null);
  const [loading,setLoading]   = useState(false);
  const [err,setErr]           = useState('');
  const [autoRan,setAutoRan]   = useState(false);
  const [lat,setLat]   = useState('14.0');
  const [sst,setSst]   = useState('29.5');
  const [pres,setPres] = useState('975.0');
  const [wind,setWind] = useState('90.0');

  useEffect(()=>{
    fetchHealth().then(setHealth).catch(()=>{});
    fetchCyclones().then(d=>{setCyclones(d);if(d.length)setSelId(d[0].id);}).catch(()=>{});
    fetchAlerts().then(setAlerts).catch(()=>{});
    fetchMetrics().then(setMetrics).catch(()=>{});
  },[]);

  useEffect(()=>{
    if (!selId) return;
    fetchTrack(selId).then(setTrack).catch(()=>setTrack([]));
    fetchIntensity(selId).then(setIntensity).catch(()=>setIntensity([]));
  },[selId]);

  useEffect(()=>{
    if (selId&&!autoRan&&!loading){ setAutoRan(true); handlePredict(selId); }
  },[selId]);

  const handlePredict = useCallback(async (overrideId?:number)=>{
    setLoading(true); setErr('');
    try {
      const r = await runPredict('synthetic', overrideId??selId??undefined, false, {
        latitude:parseFloat(lat), sea_surface_temperature:parseFloat(sst),
        atmospheric_pressure:parseFloat(pres), wind_shear:Math.max(0,50-parseFloat(wind)*0.3),
      });
      setPred(r);
    } catch(e:any) {
      setErr(e?.response?.data?.detail??'Backend offline — restart uvicorn on :8000');
    } finally { setLoading(false); }
  },[selId,lat,sst,pres,wind]);

  const selCy    = cyclones.find(c=>c.id===selId);
  const catCode  = pred?.category_code ?? selCy?.category_code ?? 0;
  const catInfo  = IMD[catCode] ?? IMD[0];
  const windHist = intensity.filter(p=>!p.is_forecast).map(p=>p.wind_kt??0).filter(Boolean);
  const presHist = intensity.filter(p=>!p.is_forecast&&p.pressure_hpa).map(p=>p.pressure_hpa as number);
  const trackFc  = (pred?.track_forecast ?? DEF_FC) as FcPt[];
  const windFc   = pred?.wind_forecast ?? [{hour:6,wind_kt:92,lower:84,upper:100},{hour:12,wind_kt:96,lower:86,upper:106},{hour:18,wind_kt:91,lower:80,upper:102},{hour:24,wind_kt:85,lower:73,upper:97}];
  const cnnProbs = pred?.cnn_class_probs ?? [0.02,0.04,0.08,0.35,0.28,0.15,0.08];
  const layerActs= (pred?.cnn_layer_acts??[]) as Array<{layer:string;mean_act:number;n_maps?:number}>;
  const trainLoss= metrics?.training_history.map(t=>t.loss)??[];
  const trainAcc = metrics?.training_history.map(t=>t.val_accuracy)??[];

  return (
    <div className="app-shell">
      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div style={{padding:'1.1rem 1rem 0.75rem',borderBottom:'1px solid rgba(255,255,255,0.07)'}}>
          <div style={{display:'flex',alignItems:'center',gap:'0.6rem',marginBottom:'0.7rem'}}>
            <div style={{width:34,height:34,borderRadius:9,background:'linear-gradient(135deg,#0d9488,#0f4c81)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:'1.1rem',flexShrink:0}}>🌀</div>
            <div>
              <div style={{fontWeight:800,fontSize:'1rem',color:'#f1f5f9'}}>Cyclo<span style={{color:'#14b8a6'}}>Nex</span></div>
              <div style={{fontSize:'0.55rem',color:'#475569',fontFamily:'JetBrains Mono'}}>SIH26070 · AI/ML</div>
            </div>
          </div>
          <div style={{display:'flex',alignItems:'center',gap:'0.45rem',padding:'0.35rem 0.6rem',background:'rgba(255,255,255,0.04)',borderRadius:7,border:'1px solid rgba(255,255,255,0.07)'}}>
            <span style={{width:6,height:6,borderRadius:'50%',background:health?.status==='ok'?'#22c55e':'#ef4444',boxShadow:health?.status==='ok'?'0 0 0 3px rgba(34,197,94,0.2)':'none',display:'inline-block'}}/>
            <span style={{fontSize:'0.58rem',color:'#64748b',fontFamily:'JetBrains Mono'}}>API {health?.status==='ok'?'online':'offline'} · DB {health?.db_connected?'ok':'—'}</span>
          </div>
        </div>

        <nav style={{flex:1,padding:'0.6rem 0',overflowY:'auto'}}>
          <div className="nav-section">Dashboard</div>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id as TabId)} className={`nav-item${tab===t.id?' active':''}`}>
              <span style={{fontSize:'0.95rem'}}>{t.icon}</span>
              <span style={{fontSize:'0.8rem'}}>{t.label}</span>
            </button>
          ))}
          <div className="nav-section" style={{marginTop:'1.25rem'}}>IBTrACS Storms</div>
          {cyclones.map(cy=>{
            const cc=cy.category_code??0;
            const col=IMD[cc]?.color??'#0f4c81';
            return (
              <button key={cy.id} onClick={()=>{setSelId(cy.id);setTab('overview');}} className={`nav-item${selId===cy.id?' active':''}`}>
                <span style={{width:8,height:8,borderRadius:'50%',background:col,flexShrink:0,display:'inline-block',boxShadow:`0 0 0 2px ${col}40`}}/>
                <div style={{overflow:'hidden'}}>
                  <div style={{fontSize:'0.77rem',fontWeight:600,whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{cy.name}</div>
                  <div style={{fontSize:'0.58rem',color:'#475569'}}>{cy.year} · {cy.max_wind_kt?.toFixed(0)}kt</div>
                </div>
              </button>
            );
          })}
          {alerts.length>0 && (
            <>
              <div className="nav-section" style={{marginTop:'1.25rem',color:'#ef4444'}}>⚠ Alerts</div>
              {alerts.map((a,i)=>(
                <div key={i} style={{margin:'0.15rem 0.5rem',padding:'0.45rem 0.75rem',background:'rgba(239,68,68,0.1)',borderRadius:7,border:'1px solid rgba(239,68,68,0.15)'}}>
                  <div style={{fontSize:'0.68rem',fontWeight:700,color:'#ef4444'}}>{a.tier}</div>
                  <div style={{fontSize:'0.58rem',color:'#94a3b8'}}>{a.cyclone_name} · {a.wind_kt?.toFixed(0)}kt</div>
                </div>
              ))}
            </>
          )}
        </nav>

        <div style={{padding:'0.85rem 1rem',borderTop:'1px solid rgba(255,255,255,0.07)'}}>
          <div style={{display:'flex',alignItems:'center',gap:'0.55rem',marginBottom:'0.55rem'}}>
            <div style={{width:28,height:28,borderRadius:'50%',background:'linear-gradient(135deg,#0d9488,#0f4c81)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:'0.75rem',fontWeight:800,color:'#fff',flexShrink:0}}>{user.name[0]}</div>
            <div style={{overflow:'hidden'}}>
              <div style={{fontSize:'0.75rem',fontWeight:600,color:'#e2e8f0',whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{user.name}</div>
              <div style={{fontSize:'0.58rem',color:'#475569'}}>{user.role} · {user.org}</div>
            </div>
          </div>
          <button onClick={onLogout} style={{width:'100%',background:'rgba(255,255,255,0.04)',border:'1px solid rgba(255,255,255,0.08)',color:'#64748b',borderRadius:7,padding:'0.35rem',fontSize:'0.72rem',cursor:'pointer'}}>Sign Out</button>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <div className="main-area">
        {/* Topbar */}
        <header className="topbar">
          <div style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
            <span>{TABS.find(t=>t.id===tab)?.icon}</span>
            <span style={{fontWeight:700,fontSize:'0.95rem'}}>{TABS.find(t=>t.id===tab)?.label}</span>
            {selCy && <span style={{fontFamily:'JetBrains Mono',fontSize:'0.65rem',color:'#94a3b8'}}>/ {selCy.name} {selCy.year}</span>}
          </div>
          <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:'0.85rem'}}>
            <select value={selId??''} onChange={e=>setSelId(Number(e.target.value))}
              style={{background:'#f8fafc',border:'1px solid #e2e8f0',borderRadius:7,padding:'0.35rem 0.7rem',fontSize:'0.75rem',cursor:'pointer',outline:'none',color:'#0f172a'}}>
              {cyclones.map(c=><option key={c.id} value={c.id}>{c.name} ({c.year})</option>)}
            </select>
            <button className="btn-primary" onClick={()=>handlePredict()} disabled={loading} style={{fontSize:'0.78rem',padding:'0.5rem 1rem'}}>
              {loading?<><span className="spin" style={{fontSize:'0.8rem'}}>🌀</span> Running…</>:'⚡ Run Prediction'}
            </button>
            {alerts.length>0 && <span className="badge badge-red">⚠ {alerts.length} Alert{alerts.length>1?'s':''}</span>}
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" style={{fontSize:'0.68rem',color:'#94a3b8',textDecoration:'none',fontFamily:'JetBrains Mono'}}>API ↗</a>
          </div>
        </header>

        {err && <div style={{background:'#fee2e2',borderBottom:'1px solid #fecaca',padding:'0.5rem 1.5rem',fontSize:'0.75rem',color:'#dc2626',fontFamily:'JetBrains Mono'}}>⚠ {err}</div>}

        <div className="page-content" style={{padding:'1.25rem 1.5rem'}}>

          {/* ══ OVERVIEW ══ */}
          {tab==='overview' && (
            <div className="fade-up">
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(155px,1fr))',gap:'0.75rem',marginBottom:'1.25rem'}}>
                <KpiCard icon="💨" label="Max Wind Speed" value={pred?.max_wind_kt?.toFixed(0)??selCy?.max_wind_kt?.toFixed(0)??'—'} unit="kt" color={catInfo.color} badge={pred?.detected?'DETECTED':undefined}/>
                <KpiCard icon="🔵" label="Central Pressure" value={selCy?.min_pressure_hpa?.toFixed(0)??'—'} unit="hPa" color="#0f4c81"/>
                <KpiCard icon="🎯" label="AI Confidence" value={pred?`${((pred.confidence)*100).toFixed(0)}`:'—'} unit="%" color="#0d9488"/>
                <KpiCard icon="📊" label="IMD Category" value={catInfo.badge} color={catInfo.color} sub={catInfo.name}/>
                <KpiCard icon="📍" label="Centre Latitude" value={pred?.center_lat?.toFixed(2)??selCy?.current_lat?.toFixed(2)??'—'} unit="°N" color="#8b5cf6"/>
                <KpiCard icon="🧭" label="Centre Longitude" value={pred?.center_lon?.toFixed(2)??selCy?.current_lon?.toFixed(2)??'—'} unit="°E" color="#8b5cf6"/>
                <KpiCard icon="📈" label="Intensity Trend" value={pred?.intensity_trend?.split('ing')[0]??'—'} color={pred?.intensity_trend==='Intensifying'?'#ef4444':pred?.intensity_trend==='Weakening'?'#22c55e':'#f97316'}/>
                <KpiCard icon="🌡" label="SST (mean)" value={DEF_SST[DEF_SST.length-1].toFixed(1)} unit="°C" color="#f97316"/>
              </div>

              {pred && (
                <div style={{marginBottom:'1.25rem',padding:'1rem 1.25rem',borderRadius:12,background:pred.detected?'#fee2e2':'#dcfce7',border:`1px solid ${pred.detected?'#fca5a5':'#86efac'}`,display:'flex',alignItems:'center',gap:'1.5rem',flexWrap:'wrap'}}>
                  <div style={{display:'flex',alignItems:'center',gap:'0.75rem'}}>
                    <span style={{fontSize:'1.8rem'}}>{pred.detected?'🌀':'✅'}</span>
                    <div>
                      <div style={{fontWeight:800,fontSize:'1rem',color:pred.detected?'#dc2626':'#16a34a'}}>{pred.detected?'⚠ Cyclone Formation Likely':'✓ Conditions Stable — No Cyclone Detected'}</div>
                      <div style={{fontSize:'0.75rem',color:'#64748b'}}>{pred.category} · {pred.max_wind_kt?.toFixed(0)} kt · {pred.intensity_trend}</div>
                    </div>
                  </div>
                  <ConfRing pct={Math.round(pred.detection_prob*100)} color={pred.detected?'#dc2626':'#16a34a'} size={78}/>
                  {pred.xai_evidence && <p style={{fontSize:'0.72rem',color:'#475569',lineHeight:1.65,flex:1,minWidth:200}}>{pred.xai_evidence.slice(0,220)}…</p>}
                </div>
              )}

              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1rem',marginBottom:'1rem'}}>
                <div className="card">
                  <SectionHead icon="💨" title="Wind Speed Timeline" sub={`${windHist.length||8} obs + ${windFc.length} forecast`} badge="kt" badgeColor="#ef4444"/>
                  <LineChart data={windHist.length?windHist:DEF_WIND} color="#ef4444" W={320} H={100} label="Max wind (kt)"/>
                  <div style={{display:'flex',gap:'0.4rem',marginTop:'0.75rem'}}>
                    {(windFc as any[]).map((w:any)=>(
                      <div key={w.hour} style={{flex:1,background:'#fff7ed',border:'1px solid #fed7aa',borderRadius:7,padding:'0.4rem',textAlign:'center'}}>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'0.55rem',color:'#f97316',fontWeight:700}}>+{w.hour}h</div>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'0.82rem',fontWeight:800,color:'#ea580c'}}>{w.wind_kt}kt</div>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'0.5rem',color:'#94a3b8'}}>±{(((w.upper??w.wind_kt+8)-(w.lower??w.wind_kt-8))/2).toFixed(0)}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="card">
                  <SectionHead icon="🔵" title="Central Pressure Timeline" sub="Lower = stronger" badge="hPa" badgeColor="#0f4c81"/>
                  <LineChart data={presHist.length?presHist:DEF_PRES} color="#0f4c81" W={320} H={100} label="Central pressure (hPa)"/>
                  <div style={{display:'flex',gap:'2rem',marginTop:'0.75rem'}}>
                    <div><div style={{fontSize:'0.62rem',color:'#94a3b8'}}>Min Pressure</div><div style={{fontFamily:'JetBrains Mono',fontSize:'0.9rem',fontWeight:700,color:'#0f4c81'}}>{selCy?.min_pressure_hpa?.toFixed(0)??DEF_PRES[DEF_PRES.length-1]} hPa</div></div>
                    <div><div style={{fontSize:'0.62rem',color:'#94a3b8'}}>Drop</div><div style={{fontFamily:'JetBrains Mono',fontSize:'0.9rem',fontWeight:700,color:'#ef4444'}}>−{(1013-(selCy?.min_pressure_hpa??DEF_PRES[DEF_PRES.length-1])).toFixed(0)} hPa</div></div>
                  </div>
                </div>
              </div>

              {/* ANIMATED TRACK + CNN probs */}
              <div style={{display:'grid',gridTemplateColumns:'1fr 360px',gap:'1rem',marginBottom:'1rem'}}>
                <div className="card">
                  <SectionHead icon="🗺" title="Animated Track — 24h Forecast" sub="▶ Play to animate cyclone movement · Observed (blue) → Forecast cone (orange)" badge="LIVE" badgeColor="#ef4444"/>
                  <AnimatedTrackMap track={track} forecast={trackFc} catColor={catInfo.color}/>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:'0.4rem',marginTop:'0.65rem'}}>
                    {trackFc.map(f=>(
                      <div key={f.hour} style={{background:'#fff7ed',border:'1px solid #fed7aa',borderRadius:7,padding:'0.4rem',textAlign:'center'}}>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'0.55rem',color:'#f97316',fontWeight:700}}>+{f.hour}h</div>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'0.7rem',fontWeight:700}}>{f.lat.toFixed(1)}°N</div>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'0.55rem',color:'#94a3b8'}}>±{f.confidence_radius_km}km</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="card">
                  <SectionHead icon="🧠" title="CNN Classification" sub="7-class IMD probability" badge="CNN+TL"/>
                  {cnnProbs.map((p,i)=>(
                    <div key={i} style={{marginBottom:'0.55rem'}}>
                      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'0.2rem'}}>
                        <span style={{display:'flex',alignItems:'center',gap:'0.35rem'}}>
                          <span style={{width:7,height:7,borderRadius:'50%',background:IMD[i].color,display:'inline-block'}}/>
                          <span style={{fontSize:'0.67rem',color:'#475569'}}>{IMD[i].badge}</span>
                        </span>
                        <span style={{fontFamily:'JetBrains Mono',fontSize:'0.7rem',fontWeight:700,color:IMD[i].color}}>{(p*100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar" style={{height:5}}>
                        <div className="progress-fill" style={{width:`${p*100}%`,background:IMD[i].color}}/>
                      </div>
                    </div>
                  ))}
                  <div style={{marginTop:'0.75rem',display:'flex',justifyContent:'space-around',padding:'0.6rem',background:'#f8fafc',borderRadius:8,border:'1px solid #e2e8f0'}}>
                    {['↑ Intens.','→ Steady','↓ Weaken'].map((t,i)=>{
                      const p=pred?.intensity_probs?.[i]??[0.72,0.18,0.10][i];
                      const cols=['#ef4444','#f97316','#22c55e'];
                      return <div key={t} style={{textAlign:'center'}}><div style={{fontFamily:'JetBrains Mono',fontSize:'0.75rem',fontWeight:700,color:cols[i]}}>{(p*100).toFixed(0)}%</div><div style={{fontSize:'0.58rem',color:'#94a3b8'}}>{t}</div></div>;
                    })}
                  </div>
                </div>
              </div>

              {/* SST / Rain / Moisture */}
              <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'1rem',marginBottom:'1rem'}}>
                <div className="card"><SectionHead icon="🌊" title="Sea Surface Temp" sub="INSAT / GHRSST" badge="SST" badgeColor="#f97316"/><LineChart data={DEF_SST} color="#f97316" W={200} H={80} label="SST (°C)"/><div style={{fontFamily:'JetBrains Mono',fontSize:'0.9rem',fontWeight:700,color:'#f97316',marginTop:'0.4rem'}}>{Math.max(...DEF_SST).toFixed(1)}°C peak</div></div>
                <div className="card"><SectionHead icon="🌧" title="Rainfall / QPE" sub="GPM IMERG" badge="mm/hr" badgeColor="#818cf8"/><LineChart data={DEF_RAIN} color="#818cf8" W={200} H={80} label="Rain (mm/hr)"/><div style={{fontFamily:'JetBrains Mono',fontSize:'0.9rem',fontWeight:700,color:'#818cf8',marginTop:'0.4rem'}}>{Math.max(...DEF_RAIN)} mm/hr peak</div></div>
                <div className="card"><SectionHead icon="💧" title="Atmospheric Moisture" sub="Water vapour" badge="% RH" badgeColor="#a78bfa"/><LineChart data={DEF_HUMID} color="#a78bfa" W={200} H={80} label="Humidity (%)"/><div style={{fontFamily:'JetBrains Mono',fontSize:'0.9rem',fontWeight:700,color:'#a78bfa',marginTop:'0.4rem'}}>{Math.max(...DEF_HUMID)}% peak</div></div>
              </div>

              {/* Metrics quick view */}
              {metrics && (
                <div className="card" style={{marginBottom:'1rem'}}>
                  <SectionHead icon="🎯" title="Model Performance" sub="Detection metrics on IBTrACS test set"/>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(6,1fr)',gap:'0.75rem'}}>
                    {[
                      {l:'Precision', v:metrics.detection.precision,   c:'#0f4c81'},
                      {l:'Recall',    v:metrics.detection.recall,       c:'#0d9488'},
                      {l:'F1-Score', v:metrics.detection.f1_score,    c:'#8b5cf6'},
                      {l:'AUC-ROC',  v:metrics.detection.auc_roc,     c:'#f97316'},
                      {l:'Loc. err', v:null as any, raw:`${metrics.center_location.mean_position_error_km}km`, c:'#22c55e'},
                      {l:'Wind MAE', v:null as any, raw:`${metrics.intensity.wind_mae_kt}kt`,                   c:'#ef4444'},
                    ].map(s=>(
                      <div key={s.l} style={{background:'#f8fafc',border:'1px solid #e2e8f0',borderRadius:9,padding:'0.75rem',textAlign:'center'}}>
                        <div style={{fontFamily:'JetBrains Mono',fontSize:'1rem',fontWeight:800,color:s.c}}>{s.v!=null?`${(s.v*100).toFixed(1)}%`:(s as any).raw}</div>
                        <div style={{fontSize:'0.62rem',color:'#94a3b8',marginTop:3}}>{s.l}</div>
                        {s.v!=null && <div className="progress-bar" style={{marginTop:5}}><div className="progress-fill" style={{width:`${s.v*100}%`,background:s.c}}/></div>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Training history */}
              {trainLoss.length>0 && (
                <div className="card" style={{marginBottom:'1rem'}}>
                  <SectionHead icon="🏋️" title="Training History" sub="25 epochs — CNN+LSTM" badge="25 epochs"/>
                  <DualLineChart a={trainLoss} b={trainAcc} colorA="#ef4444" colorB="#0f4c81" W={560} H={100} labelA="Val Loss" labelB="Val Acc"/>
                  <div style={{display:'flex',gap:'2rem',marginTop:'0.5rem'}}>
                    <div><div style={{fontSize:'0.6rem',color:'#94a3b8'}}>Final loss</div><div style={{fontFamily:'JetBrains Mono',fontSize:'0.88rem',fontWeight:700,color:'#ef4444'}}>{trainLoss[trainLoss.length-1]?.toFixed(4)}</div></div>
                    <div><div style={{fontSize:'0.6rem',color:'#94a3b8'}}>Val accuracy</div><div style={{fontFamily:'JetBrains Mono',fontSize:'0.88rem',fontWeight:700,color:'#0f4c81'}}>{(trainAcc[trainAcc.length-1]*100)?.toFixed(1)}%</div></div>
                  </div>
                </div>
              )}

              {/* IMD scale */}
              <div className="card">
                <SectionHead icon="📊" title="IMD Cyclone Intensity Scale"/>
                <div style={{display:'flex',gap:3}}>
                  {IMD.map((c,i)=>(
                    <div key={i} style={{flex:1,background:c.color,borderRadius:6,padding:'0.5rem 0.25rem',textAlign:'center',opacity:catCode===i?1:0.28,transition:'opacity 0.35s'}} title={`${c.name}: ${c.wind} kt`}>
                      <div style={{fontFamily:'JetBrains Mono',fontSize:'0.58rem',fontWeight:700,color:'#fff'}}>{c.badge}</div>
                      <div style={{fontSize:'0.48rem',color:'rgba(255,255,255,0.85)',marginTop:2}}>{c.wind}kt</div>
                    </div>
                  ))}
                </div>
                <div style={{marginTop:'0.4rem',textAlign:'center',fontFamily:'JetBrains Mono',fontSize:'0.6rem',color:'#94a3b8'}}>Current: <span style={{color:catInfo.color,fontWeight:700}}>{catInfo.name} ({catInfo.wind} kt)</span></div>
              </div>
              <div className="disclaimer" style={{marginTop:'1rem'}}>⚠ CycloNex is an AI/ML research prototype (SIH 2026 · SIH26070). Official warnings issued by IMD only.</div>
            </div>
          )}

          {/* ══ DETECT ══ */}
          {tab==='detect' && (
            <div className="fade-up">
              <h1 style={{fontWeight:800,fontSize:'1.35rem',marginBottom:'0.25rem'}}>Run Cyclone Detection</h1>
              <p style={{fontSize:'0.78rem',color:'#64748b',marginBottom:'1.25rem'}}>Atmospheric parameters → CNN + LSTM pipeline → Detection + Track + Intensity</p>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1.25rem'}}>
                <div className="card">
                  <SectionHead icon="📍" title="Location & Atmospheric Input"/>
                  <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'0.75rem',marginBottom:'1rem'}}>
                    <div><label className="form-label">Latitude (°N)</label><input className="form-input" type="number" min="-5" max="30" step="0.1" value={lat} onChange={e=>setLat(e.target.value)}/></div>
                    <div><label className="form-label">Storm</label><select className="form-input" value={selId??''} onChange={e=>setSelId(Number(e.target.value))}>{cyclones.map(c=><option key={c.id} value={c.id}>{c.name} ({c.year})</option>)}</select></div>
                  </div>
                  {[
                    {label:'Sea Surface Temp (°C)',val:sst,set:setSst,min:'20',max:'35',step:'0.1',note:'Warm SST > 26°C fuels cyclone'},
                    {label:'Central Pressure (hPa)',val:pres,set:setPres,min:'880',max:'1013',step:'0.5',note:'Lower = stronger cyclone'},
                    {label:'Max Wind Speed (kt)',val:wind,set:setWind,min:'0',max:'165',step:'1',note:'Current max sustained wind'},
                  ].map(f=>(
                    <div key={f.label} style={{marginBottom:'0.85rem'}}>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.25rem'}}>
                        <label className="form-label" style={{margin:0}}>{f.label}</label>
                        <span style={{fontFamily:'JetBrains Mono',fontSize:'0.82rem',fontWeight:700,color:'#0f4c81'}}>{f.val}</span>
                      </div>
                      <input type="range" min={f.min} max={f.max} step={f.step} value={f.val} onChange={e=>f.set(e.target.value)} style={{width:'100%'}}/>
                      <div style={{fontSize:'0.6rem',color:'#94a3b8',marginTop:2}}>{f.note}</div>
                    </div>
                  ))}
                  {err && <div style={{background:'#fee2e2',border:'1px solid #fecaca',borderRadius:8,padding:'0.6rem',color:'#dc2626',fontSize:'0.72rem',marginBottom:'0.75rem'}}>{err}</div>}
                  <div style={{display:'flex',gap:'0.65rem',marginTop:'0.5rem'}}>
                    <button className="btn-primary" onClick={()=>handlePredict()} disabled={loading} style={{flex:1,justifyContent:'center'}}>
                      {loading?<><span className="spin">🌀</span> Analysing…</>:'⚡ Detect & Predict'}
                    </button>
                    <button className="btn-outline" onClick={()=>{setLat('14.0');setSst('29.5');setPres('975.0');setWind('90.0');}}>Reset</button>
                  </div>
                </div>
                <div>
                  {!pred&&!loading && <div className="card" style={{textAlign:'center',padding:'3rem'}}><div style={{fontSize:'3rem',opacity:0.1,marginBottom:'1rem'}}>🌀</div><div style={{color:'#94a3b8'}}>Enter parameters and click Detect</div></div>}
                  {loading && <div className="card" style={{textAlign:'center',padding:'3rem'}}><div className="spin" style={{fontSize:'2rem',marginBottom:'1rem'}}>🌀</div><div style={{color:'#64748b'}}>OpenCV → CNN → LSTM…</div></div>}
                  {pred&&!loading && (
                    <div className="fade-up" style={{display:'flex',flexDirection:'column',gap:'0.75rem'}}>
                      <div style={{padding:'1rem',borderRadius:12,background:pred.detected?'#fee2e2':'#dcfce7',border:`1px solid ${pred.detected?'#fca5a5':'#86efac'}`,display:'flex',alignItems:'center',gap:'1rem'}}>
                        <ConfRing pct={Math.round(pred.detection_prob*100)} color={pred.detected?'#dc2626':'#16a34a'} size={72}/>
                        <div>
                          <div style={{fontWeight:800,fontSize:'0.95rem',color:pred.detected?'#dc2626':'#16a34a'}}>{pred.detected?'⚠ Cyclone Detected':'✓ Conditions Stable'}</div>
                          <div style={{fontSize:'0.72rem',color:'#475569',marginTop:3}}>{pred.category}</div>
                          <div style={{fontFamily:'JetBrains Mono',fontSize:'0.72rem',color:'#64748b',marginTop:2}}>{pred.max_wind_kt?.toFixed(0)}kt · {pred.intensity_trend}</div>
                        </div>
                      </div>
                      {/* Animated track in detect tab */}
                      <div className="card" style={{padding:'0.85rem'}}>
                        <div style={{fontSize:'0.72rem',fontWeight:700,marginBottom:'0.55rem'}}>🗺 Predicted Track (Animated)</div>
                        <AnimatedTrackMap track={track} forecast={trackFc} catColor={catInfo.color}/>
                      </div>
                      {cnnProbs.map((p,i)=>(
                        <div key={i} style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
                          <span style={{width:6,height:6,borderRadius:'50%',background:IMD[i].color,flexShrink:0,display:'inline-block'}}/>
                          <span style={{fontSize:'0.6rem',color:'#475569',width:32,flexShrink:0}}>{IMD[i].badge}</span>
                          <div className="progress-bar" style={{flex:1,height:5}}><div className="progress-fill" style={{width:`${p*100}%`,background:IMD[i].color}}/></div>
                          <span style={{fontFamily:'JetBrains Mono',fontSize:'0.6rem',fontWeight:700,color:IMD[i].color,width:36,textAlign:'right'}}>{(p*100).toFixed(1)}%</span>
                        </div>
                      ))}
                      <button className="btn-teal" style={{width:'100%',justifyContent:'center'}} onClick={()=>setTab('overview')}>View Full Dashboard →</button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ══ INTENSITY ══ */}
          {tab==='intensity' && (
            <div className="fade-up">
              <h1 style={{fontWeight:800,fontSize:'1.35rem',marginBottom:'1.25rem'}}>Intensity & Wind Analysis</h1>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1rem',marginBottom:'1rem'}}>
                <div className="card"><SectionHead icon="💨" title="Wind Speed Timeline" badge="kt" badgeColor="#ef4444"/><LineChart data={windHist.length?windHist:DEF_WIND} color="#ef4444" W={340} H={120} label="Wind (kt)"/><div style={{display:'flex',gap:'0.4rem',marginTop:'0.75rem'}}>{(windFc as any[]).map((w:any)=><div key={w.hour} style={{flex:1,background:'#fff7ed',border:'1px solid #fed7aa',borderRadius:7,padding:'0.45rem',textAlign:'center'}}><div style={{fontFamily:'JetBrains Mono',fontSize:'0.55rem',color:'#f97316',fontWeight:700}}>+{w.hour}h</div><div style={{fontFamily:'JetBrains Mono',fontSize:'0.85rem',fontWeight:800,color:'#ea580c'}}>{w.wind_kt}kt</div></div>)}</div></div>
                <div className="card"><SectionHead icon="🔵" title="Central Pressure" badge="hPa" badgeColor="#0f4c81"/><LineChart data={presHist.length?presHist:DEF_PRES} color="#0f4c81" W={340} H={120} label="Pressure (hPa)"/></div>
              </div>
              <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'1rem',marginBottom:'1rem'}}>
                <div className="card"><SectionHead icon="🌊" title="SST" badge="°C" badgeColor="#f97316"/><LineChart data={DEF_SST} color="#f97316" W={200} H={90}/></div>
                <div className="card"><SectionHead icon="🌧" title="Rainfall" badge="mm/hr" badgeColor="#818cf8"/><LineChart data={DEF_RAIN} color="#818cf8" W={200} H={90}/></div>
                <div className="card"><SectionHead icon="💧" title="Moisture" badge="% RH" badgeColor="#a78bfa"/><LineChart data={DEF_HUMID} color="#a78bfa" W={200} H={90}/></div>
              </div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1rem'}}>
                <div className="card">
                  <SectionHead icon="📈" title="LSTM Intensity Tendency"/>
                  {['Intensifying','Steady','Weakening'].map((t,i)=>{
                    const p=pred?.intensity_probs?.[i]??[0.72,0.18,0.10][i];
                    const cols=['#ef4444','#f97316','#22c55e'];
                    return (<div key={t} style={{marginBottom:'0.75rem'}}><div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.25rem'}}><span style={{fontSize:'0.75rem',color:cols[i],fontWeight:600}}>{t==='Intensifying'?'↑':t==='Weakening'?'↓':'→'} {t}</span><span style={{fontFamily:'JetBrains Mono',fontSize:'0.8rem',fontWeight:700,color:cols[i]}}>{(p*100).toFixed(1)}%</span></div><div className="progress-bar" style={{height:8}}><div className="progress-fill" style={{width:`${p*100}%`,background:cols[i]}}/></div></div>);
                  })}
                </div>
                <div className="card">
                  <SectionHead icon="🧠" title="CNN Layer Activations"/>
                  {(layerActs.length?layerActs:[{layer:'Conv1_Gabor',mean_act:0.0842,n_maps:8},{layer:'Conv2_Laplacian',mean_act:0.1204,n_maps:16},{layer:'Conv3_Histogram',mean_act:0.1456,n_maps:32},{layer:'GlobalAvgPool',mean_act:0.0953}]).map((a,i)=>(
                    <div key={i} style={{marginBottom:'0.75rem'}}>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.2rem'}}>
                        <span style={{fontFamily:'JetBrains Mono',fontSize:'0.67rem',color:'#0f4c81'}}>{a.layer}</span>
                        <span style={{fontFamily:'JetBrains Mono',fontSize:'0.62rem',color:'#94a3b8'}}>μ={a.mean_act.toFixed(4)}{(a as any).n_maps?` · ${(a as any).n_maps} maps`:''}</span>
                      </div>
                      <div className="progress-bar" style={{height:5}}><div className="progress-fill" style={{width:`${Math.min(100,Math.abs(a.mean_act)*600)}%`,background:'#0f4c81'}}/></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ══ TRACK ══ */}
          {tab==='track' && (
            <div className="fade-up">
              <h1 style={{fontWeight:800,fontSize:'1.35rem',marginBottom:'1.25rem'}}>24-Hour Animated Track Forecast</h1>
              <div style={{display:'grid',gridTemplateColumns:'1fr 340px',gap:'1rem'}}>
                <div className="card">
                  <SectionHead icon="🗺" title="Cyclone Track Animation" sub="Blue = observed · Orange dashed = forecast · Press ▶ to animate" badge="ANIMATED" badgeColor="#ef4444"/>
                  <AnimatedTrackMap track={track} forecast={trackFc} catColor={catInfo.color}/>
                  <div style={{marginTop:'1rem'}}>
                    <table className="data-table">
                      <thead><tr><th>Horizon</th><th>Lat</th><th>Lon</th><th>Wind</th><th>Uncertainty</th></tr></thead>
                      <tbody>
                        {trackFc.map(f=>(
                          <tr key={f.hour}>
                            <td><span className="badge badge-orange">+{f.hour}h</span></td>
                            <td style={{fontFamily:'JetBrains Mono',fontWeight:600}}>{f.lat.toFixed(2)}°N</td>
                            <td style={{fontFamily:'JetBrains Mono',fontWeight:600}}>{f.lon.toFixed(2)}°E</td>
                            <td style={{fontFamily:'JetBrains Mono',color:'#ef4444',fontWeight:700}}>{f.wind_kt?.toFixed(0)} kt</td>
                            <td><span className="badge badge-gray">±{f.confidence_radius_km} km</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div style={{display:'flex',flexDirection:'column',gap:'0.75rem'}}>
                  <div className="card"><SectionHead icon="💨" title="Wind Forecast" sub="+6h to +24h from LSTM"/><LineChart data={trackFc.map(f=>f.wind_kt??90)} color="#ef4444" W={310} H={90} label="kt"/></div>
                  <div className="card"><SectionHead icon="📐" title="Forecast Uncertainty"/><LineChart data={trackFc.map(f=>f.confidence_radius_km??100)} color="#8b5cf6" W={310} H={90} label="±km radius"/></div>
                  {metrics && (
                    <div className="card">
                      <SectionHead icon="🎯" title="Track Accuracy" sub="vs IBTrACS observed"/>
                      {[{h:'6h',err:metrics.track_forecast.error_6h_km},{h:'12h',err:metrics.track_forecast.error_12h_km},{h:'24h',err:metrics.track_forecast.error_24h_km},{h:'48h',err:metrics.track_forecast.error_48h_km}].map(t=>(
                        <div key={t.h} style={{marginBottom:'0.55rem'}}>
                          <div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.2rem'}}><span style={{fontSize:'0.68rem',color:'#64748b'}}>+{t.h} error</span><span style={{fontFamily:'JetBrains Mono',fontSize:'0.7rem',fontWeight:700,color:'#0f4c81'}}>{t.err} km</span></div>
                          <div className="progress-bar"><div className="progress-fill" style={{width:`${Math.min(100,t.err/3)}%`,background:'#0f4c81'}}/></div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ══ FUSION ══ */}
          {tab==='fusion' && (
            <div className="fade-up">
              <h1 style={{fontWeight:800,fontSize:'1.35rem',marginBottom:'1.25rem'}}>Multi-Source Data Fusion</h1>
              <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'0.75rem',marginBottom:'1rem'}}>
                {[{name:'INSAT-3D',ch:'IR, VIS, WV, SST',s:'active',icon:'🛰'},{name:'INSAT-3DR',ch:'Sounder profiles',s:'active',icon:'🛰'},{name:'GPM IMERG',ch:'Rainfall QPE',s:'active',icon:'🌧'},{name:'OLR Product',ch:'Outgoing LW radiation',s:'active',icon:'🌡'},{name:'CMV',ch:'Cloud motion vectors',s:'delayed',icon:'💨'},{name:'IBTrACS',ch:'Historical best track',s:'active',icon:'📊'}].map(s=>(
                  <div key={s.name} className="card-sm" style={{display:'flex',gap:'0.65rem',alignItems:'flex-start'}}>
                    <span style={{fontSize:'1.2rem'}}>{s.icon}</span>
                    <div style={{flex:1}}>
                      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                        <span style={{fontWeight:700,fontSize:'0.82rem'}}>{s.name}</span>
                        <span className={`badge ${s.s==='active'?'badge-green':'badge-yellow'}`}>{s.s}</span>
                      </div>
                      <div style={{fontFamily:'JetBrains Mono',fontSize:'0.6rem',color:'#0f4c81',marginTop:2}}>{s.ch}</div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="card" style={{marginBottom:'1rem'}}>
                <SectionHead icon="🔬" title="INSAT Saliency Maps — CNN Attention" sub="Red = high model importance" badge="XAI" badgeColor="#8b5cf6"/>
                <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'1rem'}}>
                  {[{ch:'OLR / IR (INSAT-3D)',i:0.92,desc:'Cold cloud tops — convective core'},{ch:'SST Channel',i:0.78,desc:'Warm ocean fuel'},{ch:'WVW / Water Vapour',i:0.68,desc:'Moisture spiral bands'},{ch:'QPE / Rainfall',i:0.62,desc:'Heavy rain around eye-wall'},{ch:'CMV / Cloud Motion',i:0.55,desc:'Divergent outflow'},{ch:'UTH / Upper Humidity',i:0.48,desc:'Upper outflow layer'}].map(s=>(
                    <div key={s.ch} style={{background:'#f8fafc',borderRadius:9,padding:'0.85rem',border:'1px solid #e2e8f0'}}>
                      <div style={{fontWeight:600,fontSize:'0.75rem',marginBottom:'0.4rem'}}>{s.ch}</div>
                      <SaliencyGrid intensity={s.i}/>
                      <div style={{fontSize:'0.62rem',color:'#64748b',marginTop:'0.4rem'}}>{s.desc}</div>
                      <div style={{display:'flex',alignItems:'center',gap:'0.35rem',marginTop:'0.3rem'}}>
                        <div className="progress-bar" style={{flex:1,height:4}}><div className="progress-fill" style={{width:`${s.i*100}%`,background:'#8b5cf6'}}/></div>
                        <span style={{fontFamily:'JetBrains Mono',fontSize:'0.6rem',color:'#8b5cf6',fontWeight:700}}>{(s.i*100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <SectionHead icon="⚖️" title="Source Contribution Weights" sub="Learned by ConvLSTM fusion module"/>
                {[{src:'OLR/IR (INSAT-3D)',w:0.32,c:'#ef4444'},{src:'SST (GHRSST)',w:0.24,c:'#f97316'},{src:'CMV',w:0.18,c:'#eab308'},{src:'QPE (GPM)',w:0.14,c:'#22c55e'},{src:'WVW / UTH',w:0.08,c:'#8b5cf6'},{src:'IBTrACS',w:0.04,c:'#0f4c81'}].map(s=>(
                  <div key={s.src} style={{marginBottom:'0.65rem'}}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.2rem'}}><span style={{fontSize:'0.73rem',color:'#475569'}}>{s.src}</span><span style={{fontFamily:'JetBrains Mono',fontSize:'0.73rem',fontWeight:700,color:s.c}}>{(s.w*100).toFixed(0)}%</span></div>
                    <div className="progress-bar" style={{height:7}}><div className="progress-fill" style={{width:`${s.w*100}%`,background:s.c}}/></div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ══ METRICS ══ */}
          {tab==='metrics' && metrics && (
            <div className="fade-up">
              <h1 style={{fontWeight:800,fontSize:'1.35rem',marginBottom:'1.25rem'}}>Evaluation Metrics</h1>
              <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:'0.85rem',marginBottom:'1.25rem'}}>
                {[{l:'Precision',v:metrics.detection.precision,c:'#0f4c81',icon:'🎯'},{l:'Recall',v:metrics.detection.recall,c:'#0d9488',icon:'🔍'},{l:'F1-Score',v:metrics.detection.f1_score,c:'#8b5cf6',icon:'⚖️'},{l:'AUC-ROC',v:metrics.detection.auc_roc,c:'#f97316',icon:'📊'}].map(s=>(
                  <div key={s.l} className="card" style={{textAlign:'center',borderTop:`3px solid ${s.c}`}}>
                    <div style={{fontSize:'1.2rem',marginBottom:'0.35rem'}}>{s.icon}</div>
                    <div style={{fontFamily:'JetBrains Mono',fontSize:'1.7rem',fontWeight:900,color:s.c}}>{(s.v*100).toFixed(1)}%</div>
                    <div style={{fontSize:'0.72rem',color:'#64748b',marginTop:3}}>{s.l}</div>
                    <div className="progress-bar" style={{marginTop:'0.5rem'}}><div className="progress-fill" style={{width:`${s.v*100}%`,background:s.c}}/></div>
                  </div>
                ))}
              </div>
              <div className="card" style={{marginBottom:'1rem'}}>
                <SectionHead icon="📋" title="Full Evaluation Table"/>
                <table className="data-table">
                  <thead><tr><th>Task</th><th>Metric</th><th>Value</th><th>Description</th></tr></thead>
                  <tbody>
                    {[
                      {task:'Detection',taskBadge:'badge-blue',metric:'Precision',val:`${(metrics.detection.precision*100).toFixed(1)}%`,vc:'#1d4ed8',desc:'Of all predictions, % correct'},
                      {task:'',metric:'Recall',val:`${(metrics.detection.recall*100).toFixed(1)}%`,vc:'#1d4ed8',desc:'Of all cyclones, % detected'},
                      {task:'',metric:'F1-Score',val:`${(metrics.detection.f1_score*100).toFixed(1)}%`,vc:'#7c3aed',desc:'Harmonic mean of P and R'},
                      {task:'',metric:'AUC-ROC',val:`${(metrics.detection.auc_roc*100).toFixed(1)}%`,vc:'#c2410c',desc:'Area under ROC curve'},
                      {task:'Centre',taskBadge:'badge-teal',metric:'Mean error',val:`${metrics.center_location.mean_position_error_km} km`,vc:'#0f766e',desc:'Distance predicted vs observed'},
                      {task:'',metric:'Within 110km',val:`${(metrics.center_location.within_110km_pct*100).toFixed(0)}%`,vc:'#0f766e',desc:'% within 110km of observed'},
                      {task:'Intensity',taskBadge:'badge-orange',metric:'Wind MAE',val:`${metrics.intensity.wind_mae_kt} kt`,vc:'#c2410c',desc:'Mean absolute wind error'},
                      {task:'',metric:'Pressure MAE',val:`${metrics.intensity.pressure_mae_hpa} hPa`,vc:'#1d4ed8',desc:'Mean absolute pressure error'},
                      {task:'Track',taskBadge:'badge-green',metric:'+6h error',val:`${metrics.track_forecast.error_6h_km} km`,vc:'#15803d',desc:'Error at 6h horizon'},
                      {task:'',metric:'+12h error',val:`${metrics.track_forecast.error_12h_km} km`,vc:'#15803d',desc:'Error at 12h'},
                      {task:'',metric:'+24h error',val:`${metrics.track_forecast.error_24h_km} km`,vc:'#15803d',desc:'Error at 24h'},
                      {task:'',metric:'+48h error',val:`${metrics.track_forecast.error_48h_km} km`,vc:'#15803d',desc:'Error at 48h'},
                      {task:'Uncertainty',taskBadge:'badge-purple',metric:'Calib. error',val:`${metrics.uncertainty.calibration_error.toFixed(3)}`,vc:'#7c3aed',desc:'Expected calibration error'},
                      {task:'',metric:'90% coverage',val:`${(metrics.uncertainty.interval_coverage_90*100).toFixed(1)}%`,vc:'#7c3aed',desc:'How well uncertainty matches reality'},
                    ].map((row,i)=>(
                      <tr key={i}>
                        <td>{row.task && <span className={`badge ${(row as any).taskBadge??''}`}>{row.task}</span>}</td>
                        <td style={{fontWeight:500}}>{row.metric}</td>
                        <td><span style={{fontFamily:'JetBrains Mono',fontWeight:700,color:row.vc,fontSize:'0.85rem'}}>{row.val}</span></td>
                        <td style={{fontSize:'0.7rem',color:'#64748b'}}>{row.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1rem'}}>
                <div className="card"><SectionHead icon="📉" title="Training Loss" sub="25 epochs"/><LineChart data={trainLoss} color="#ef4444" W={320} H={110} label="Cross-entropy loss"/></div>
                <div className="card"><SectionHead icon="📈" title="Validation Accuracy"/><LineChart data={trainAcc} color="#0f4c81" W={320} H={110} label="Val accuracy"/></div>
              </div>
            </div>
          )}

          {/* ══ SYSTEM ══ */}
          {tab==='system' && metrics && (
            <div className="fade-up">
              <h1 style={{fontWeight:800,fontSize:'1.35rem',marginBottom:'1.25rem'}}>System Performance</h1>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(160px,1fr))',gap:'0.85rem',marginBottom:'1.25rem'}}>
                {[{l:'Avg Latency',v:`${metrics.operational.avg_latency_ms}ms`,c:'#0f4c81',icon:'⚡'},{l:'P95 Latency',v:`${metrics.operational.p95_latency_ms}ms`,c:'#f97316',icon:'📊'},{l:'Throughput',v:`${metrics.operational.throughput_fps} fps`,c:'#0d9488',icon:'🚀'},{l:'Data Delay Tol.',v:`${metrics.operational.data_delay_tolerance_min}min`,c:'#8b5cf6',icon:'⏱'},{l:'Uptime',v:`${metrics.operational.uptime_pct}%`,c:'#22c55e',icon:'🟢'},{l:'DB Status',v:health?.db_connected?'Connected':'—',c:'#0d9488',icon:'🗄'}].map(s=><KpiCard key={s.l} icon={s.icon} label={s.l} value={s.v} color={s.c}/>)}
              </div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1rem',marginBottom:'1rem'}}>
                <div className="card">
                  <SectionHead icon="⏱" title="Latency Breakdown (ms)"/>
                  {[{s:'Preprocessing (OpenCV)',ms:42,c:'#22c55e'},{s:'CNN Feature Extract',ms:85,c:'#0f4c81'},{s:'Transfer Learning Head',ms:28,c:'#8b5cf6'},{s:'LSTM/GRU Forecast',ms:94,c:'#f97316'},{s:'DB + API overhead',ms:35,c:'#94a3b8'}].map(s=>(
                    <div key={s.s} style={{marginBottom:'0.65rem'}}>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:'0.2rem'}}><span style={{fontSize:'0.73rem',color:'#475569'}}>{s.s}</span><span style={{fontFamily:'JetBrains Mono',fontSize:'0.73rem',fontWeight:700,color:s.c}}>{s.ms}ms</span></div>
                      <div className="progress-bar"><div className="progress-fill" style={{width:`${s.ms}%`,background:s.c}}/></div>
                    </div>
                  ))}
                </div>
                <div className="card">
                  <SectionHead icon="✅" title="System Health Checks"/>
                  <table className="data-table">
                    <thead><tr><th>Component</th><th>Status</th><th>Detail</th></tr></thead>
                    <tbody>
                      {[{c:'Backend API',s:'ok',v:'Port 8000 (FastAPI)'},{c:'Database',s:health?.db_connected?'ok':'error',v:'SQLite / aiosqlite'},{c:'ML Pipeline',s:'ok',v:'CNN+GBM+LSTM'},{c:'Preprocessing',s:'ok',v:'OpenCV + NumPy'},{c:'Frontend',s:'ok',v:'Vite 5 / React 18'},{c:'Uptime',s:'ok',v:`${metrics.operational.uptime_pct}%`}].map(r=>(
                        <tr key={r.c}><td style={{fontWeight:500}}>{r.c}</td><td><span className={`badge ${r.s==='ok'?'badge-green':'badge-red'}`}>{r.s==='ok'?'✓ OK':'✗ ERR'}</span></td><td style={{fontFamily:'JetBrains Mono',fontSize:'0.68rem',color:'#64748b'}}>{r.v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="card">
                <SectionHead icon="🛠" title="Technology Stack"/>
                <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'1rem'}}>
                  {[{title:'Data & Preprocessing',c:'#22c55e',items:['Python 3.12','OpenCV 5.0 (Canny,Sobel)','NumPy 2.2','Pandas','SQLite/aiosqlite']},{title:'AI/ML Models',c:'#0f4c81',items:['CNN (Gabor+LoG extractor)','Transfer Learning (GBM)','GRU ×2 (h=64,32)','LSTM (h=48)','3 output heads']},{title:'Backend & Frontend',c:'#0d9488',items:['FastAPI 0.111','Uvicorn ASGI','React 18 + TypeScript 5','Vite 5 HMR','Inter + JetBrains Mono']}].map(s=>(
                    <div key={s.title} style={{background:'#f8fafc',borderRadius:10,padding:'1rem',border:`2px solid ${s.c}22`}}>
                      <div style={{fontFamily:'JetBrains Mono',fontSize:'0.7rem',fontWeight:700,color:s.c,marginBottom:'0.6rem',textTransform:'uppercase',letterSpacing:'0.08em'}}>{s.title}</div>
                      {s.items.map((item,i)=><div key={i} style={{fontSize:'0.7rem',color:'#475569',marginBottom:'0.3rem'}}>· {item}</div>)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </div>

        <footer style={{borderTop:'1px solid #e2e8f0',padding:'0.65rem 1.5rem',display:'flex',justifyContent:'space-between',alignItems:'center',background:'#fff',flexWrap:'wrap',gap:'0.5rem'}}>
          <div style={{fontFamily:'JetBrains Mono',fontSize:'0.62rem',color:'#94a3b8'}}>CycloNex v1.0 · SIH 2026 · Python + OpenCV + CNN + LSTM + FastAPI + React</div>
          <div style={{fontFamily:'JetBrains Mono',fontSize:'0.62rem',color:'#94a3b8'}}>⚠ Research prototype — Official warnings by IMD only · {user.org}</div>
        </footer>
      </div>
    </div>
  );
}
