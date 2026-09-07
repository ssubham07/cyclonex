import { useState } from 'react';

const USERS_KEY = 'cyclonex_users';
const SESSION_KEY = 'cyclonex_session';

interface User { name: string; email: string; org: string; role: string; location: string; password: string; }
interface Props { onAuth: (u: User) => void; }

function getUsers(): User[] {
  try { return JSON.parse(localStorage.getItem(USERS_KEY) || '[]'); } catch { return []; }
}
function saveUsers(u: User[]) { localStorage.setItem(USERS_KEY, JSON.stringify(u)); }
export function saveSession(u: User) { localStorage.setItem(SESSION_KEY, JSON.stringify(u)); }
export function loadSession(): User | null {
  try { const s = localStorage.getItem(SESSION_KEY); return s ? JSON.parse(s) : null; } catch { return null; }
}
export function clearSession() { localStorage.removeItem(SESSION_KEY); }

// ── Help & Support Widget ─────────────────────────────────────────────────────
export function HelpWidget() {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 9999 }}>
      {open && (
        <div style={{
          position: 'absolute', bottom: 56, right: 0,
          width: 300, background: '#fff', borderRadius: 14,
          boxShadow: '0 8px 40px rgba(0,0,0,0.18)',
          border: '1px solid #e2e8f0', overflow: 'hidden',
          animation: 'fadeUp 0.2s ease',
        }}>
          {/* Header */}
          <div style={{ background: 'linear-gradient(135deg,#0f4c81,#0d9488)', padding: '1rem 1.1rem', color: '#fff' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.25rem' }}>
              <span style={{ fontSize: '1.1rem' }}>🌀</span>
              <span style={{ fontWeight: 800, fontSize: '0.95rem' }}>Help & Support</span>
              <button onClick={() => setOpen(false)} style={{ marginLeft: 'auto', background: 'rgba(255,255,255,0.15)', border: 'none', color: '#fff', borderRadius: '50%', width: 22, height: 22, cursor: 'pointer', fontSize: '0.75rem' }}>✕</button>
            </div>
            <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.75)', fontFamily: 'JetBrains Mono' }}>CycloNex · SIH 2026 · SIH26070</div>
          </div>

          <div style={{ padding: '1rem' }}>
            {/* Contact options */}
            {[
              { icon: '📞', label: 'Emergency Helpline', val: '+91-044-2856-4444', sub: 'IMD Cyclone Warning Centre', href: 'tel:+914428564444', color: '#ef4444' },
              { icon: '📧', label: 'Technical Support', val: 'support@cyclonex.in', sub: 'Response within 24 hours', href: 'mailto:support@cyclonex.in', color: '#0f4c81' },
              { icon: '🌐', label: 'IMD Official Portal', val: 'mausam.imd.gov.in', sub: 'Official cyclone bulletins', href: 'https://mausam.imd.gov.in', color: '#0d9488' },
              { icon: '📡', label: 'MOSDAC Data', val: 'mosdac.gov.in', sub: 'INSAT-3D/3DR satellite data', href: 'https://mosdac.gov.in', color: '#8b5cf6' },
            ].map(c => (
              <a key={c.label} href={c.href} target={c.href.startsWith('http') ? '_blank' : undefined} rel="noreferrer"
                style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem', padding: '0.65rem 0.5rem', borderBottom: '1px solid #f1f5f9', textDecoration: 'none', borderRadius: 8, transition: 'background 0.15s' }}
                onMouseEnter={e => (e.currentTarget.style.background = '#f8fafc')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                <span style={{ fontSize: '1rem', flexShrink: 0 }}>{c.icon}</span>
                <div>
                  <div style={{ fontSize: '0.72rem', fontWeight: 600, color: '#0f172a', marginBottom: 1 }}>{c.label}</div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.68rem', fontWeight: 700, color: c.color }}>{c.val}</div>
                  <div style={{ fontSize: '0.6rem', color: '#94a3b8' }}>{c.sub}</div>
                </div>
              </a>
            ))}

            {/* Regional offices */}
            <div style={{ marginTop: '0.75rem', background: '#f8fafc', borderRadius: 8, padding: '0.65rem', border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: '0.65rem', fontWeight: 700, color: '#64748b', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.07em' }}>IMD Regional Cyclone Warning Centres</div>
              {[
                { name: 'Mumbai', phone: '022-2265-1667' },
                { name: 'Chennai', phone: '044-2856-4444' },
                { name: 'Kolkata', phone: '033-2337-0111' },
                { name: 'Bhubaneswar', phone: '0674-2540-392' },
              ].map(o => (
                <div key={o.name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span style={{ fontSize: '0.67rem', color: '#475569' }}>{o.name}</span>
                  <a href={`tel:${o.phone.replace(/-/g, '')}`} style={{ fontFamily: 'JetBrains Mono', fontSize: '0.65rem', color: '#0f4c81', textDecoration: 'none', fontWeight: 600 }}>{o.phone}</a>
                </div>
              ))}
            </div>

            <div style={{ marginTop: '0.65rem', fontSize: '0.6rem', color: '#94a3b8', textAlign: 'center', lineHeight: 1.5 }}>
              ⚠ CycloNex is a research prototype.<br />For official warnings, contact IMD only.
            </div>
          </div>
        </div>
      )}

      {/* FAB button */}
      <button onClick={() => setOpen(!open)} style={{
        width: 46, height: 46, borderRadius: '50%',
        background: 'linear-gradient(135deg,#0f4c81,#0d9488)',
        border: 'none', cursor: 'pointer', color: '#fff',
        fontSize: '1.1rem', boxShadow: '0 4px 16px rgba(15,76,129,0.4)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'transform 0.2s',
      }}
        onMouseEnter={e => (e.currentTarget.style.transform = 'scale(1.1)')}
        onMouseLeave={e => (e.currentTarget.style.transform = 'scale(1)')}>
        {open ? '✕' : '❓'}
      </button>
    </div>
  );
}

// ── Auth Page ─────────────────────────────────────────────────────────────────
export function RegisterPage({ onAuth }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [form, setForm] = useState({ name: '', email: '', org: '', role: 'Meteorologist', location: '', password: '', confirm: '' });
  const [err, setErr]   = useState('');

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setForm(f => ({ ...f, [k]: e.target.value }));

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault(); setErr('');
    if (form.password !== form.confirm) return setErr('Passwords do not match.');
    if (form.password.length < 6) return setErr('Password must be at least 6 characters.');
    const users = getUsers();
    if (users.find(u => u.email === form.email)) return setErr('Email already registered — please sign in.');
    const user: User = { name: form.name, email: form.email, org: form.org, role: form.role, location: form.location, password: form.password };
    users.push(user); saveUsers(users); saveSession(user); onAuth(user);
  };

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault(); setErr('');
    const users = getUsers();
    if (!users.find(u => u.email === 'analyst@imd.gov.in'))
      users.push({ name: 'IMD Analyst', email: 'analyst@imd.gov.in', org: 'IMD', role: 'Meteorologist', location: 'New Delhi', password: 'imd2026' });
    if (!users.find(u => u.email === 'demo@cyclonex.in'))
      users.push({ name: 'Demo User', email: 'demo@cyclonex.in', org: 'SIH', role: 'Observer', location: 'Mumbai', password: 'demo123' });
    saveUsers(users);
    const user = users.find(u => u.email === form.email && u.password === form.password);
    if (!user) return setErr('Invalid email or password.');
    saveSession(user); onAuth(user);
  };

  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      {/* ── BACKGROUND matching uploaded image ── */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0,
        background: 'linear-gradient(135deg, #061212 0%, #091818 40%, #0a1f1f 70%, #061212 100%)',
      }} />
      {/* Subtle radial glow */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0, background: 'radial-gradient(ellipse 70% 60% at 30% 50%, rgba(13,148,136,0.12) 0%, transparent 70%)' }} />
      <div style={{ position: 'fixed', inset: 0, zIndex: 0, background: 'radial-gradient(ellipse 40% 40% at 80% 30%, rgba(15,76,129,0.10) 0%, transparent 70%)' }} />

      {/* ── TOPBAR ── */}
      <header style={{ position: 'relative', zIndex: 10, padding: '1rem 2.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem', borderBottom: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(8px)' }}>
        <div style={{ width: 34, height: 34, borderRadius: 9, background: 'linear-gradient(135deg,#0d9488,#0ea5e9)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.1rem' }}>🌀</div>
        <span style={{ color: '#f1f5f9', fontWeight: 900, fontSize: '1.15rem', letterSpacing: '-0.02em' }}>Cyclo<span style={{ color: '#2dd4bf' }}>Nex</span></span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e', display: 'inline-block' }} />
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: '0.62rem', color: '#64748b' }}>Model service <span style={{ color: '#22c55e', fontWeight: 700 }}>operational</span></span>
          </div>
          <a href="https://mausam.imd.gov.in" target="_blank" rel="noreferrer"
            style={{ fontFamily: 'JetBrains Mono', fontSize: '0.62rem', color: '#94a3b8', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            Official weather desk ↗
          </a>
        </div>
      </header>

      {/* ── HERO SECTION ── */}
      <div style={{ position: 'relative', zIndex: 5, flex: 1, display: 'flex', alignItems: 'center' }}>
        <div style={{ padding: '0 2.5rem', maxWidth: 1400, margin: '0 auto', width: '100%', display: 'grid', gridTemplateColumns: '1fr 420px', gap: '4rem', alignItems: 'center' }}>

          {/* Left — hero text matching uploaded image */}
          <div>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.65rem', color: '#2dd4bf', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: '1.25rem' }}>
              WEATHER INTELLIGENCE / RF-01
            </div>
            <h1 style={{ fontWeight: 900, fontSize: 'clamp(2.4rem, 5vw, 4.2rem)', lineHeight: 1.06, letterSpacing: '-0.035em', color: '#f1f5f9', marginBottom: '1.25rem' }}>
              Read the<br />atmosphere<br /><span style={{ color: '#2dd4bf' }}>before it turns.</span>
            </h1>
            <p style={{ fontSize: '0.85rem', color: '#2dd4bf', lineHeight: 1.75, maxWidth: 440, marginBottom: '2rem' }}>
              Translate seven atmospheric signals into a clear cyclone-formation risk estimate, powered by CNN + LSTM models.
            </p>

            {/* Stats row */}
            <div style={{ display: 'flex', gap: '3rem', marginBottom: '2.5rem' }}>
              {[
                { l: 'COVERAGE', v: 'Regional conditions' },
                { l: 'MODEL', v: 'CNN + LSTM' },
                { l: 'OUTPUT', v: 'Binary risk + probability' },
              ].map(s => (
                <div key={s.l}>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.55rem', color: '#475569', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '0.3rem' }}>{s.l}</div>
                  <div style={{ fontWeight: 700, fontSize: '0.82rem', color: '#f1f5f9' }}>{s.v}</div>
                </div>
              ))}
            </div>

            {/* Feature chips */}
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {['OpenCV Preprocessing', 'CNN + Transfer Learning', 'LSTM + GRU Forecast', 'INSAT-3D/3DR', 'GPM IMERG', 'IBTrACS'].map(t => (
                <span key={t} style={{ padding: '0.25rem 0.65rem', border: '1px solid rgba(45,212,191,0.25)', borderRadius: 20, fontFamily: 'JetBrains Mono', fontSize: '0.58rem', color: '#5eead4' }}>
                  {t}
                </span>
              ))}
            </div>
          </div>

          {/* Right — auth card */}
          <div style={{ background: 'rgba(255,255,255,0.04)', backdropFilter: 'blur(20px)', borderRadius: 18, border: '1px solid rgba(255,255,255,0.10)', padding: '2rem', boxShadow: '0 24px 60px rgba(0,0,0,0.35)' }}>
            {/* Tabs */}
            <div style={{ display: 'flex', marginBottom: '1.5rem', background: 'rgba(255,255,255,0.04)', borderRadius: 10, padding: '3px' }}>
              {(['login', 'register'] as const).map(t => (
                <button key={t} onClick={() => { setMode(t); setErr(''); }}
                  style={{ flex: 1, padding: '0.55rem', fontWeight: 700, fontSize: '0.78rem', background: mode === t ? 'rgba(45,212,191,0.15)' : 'transparent', border: mode === t ? '1px solid rgba(45,212,191,0.3)' : '1px solid transparent', borderRadius: 8, cursor: 'pointer', color: mode === t ? '#2dd4bf' : '#64748b', transition: 'all 0.2s' }}>
                  {t === 'login' ? '🔑 Sign In' : '✍️ Register'}
                </button>
              ))}
            </div>

            {mode === 'login' ? (
              <form onSubmit={handleLogin}>
                <div style={{ textAlign: 'center', marginBottom: '1.25rem' }}>
                  <div style={{ fontWeight: 800, fontSize: '1rem', color: '#f1f5f9', marginBottom: 3 }}>Welcome back</div>
                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Sign in to the CycloNex dashboard</div>
                </div>
                {[{ k: 'email', label: 'Email', type: 'email', ph: 'analyst@imd.gov.in' }, { k: 'password', label: 'Password', type: 'password', ph: '••••••••' }].map(f => (
                  <div key={f.k} style={{ marginBottom: '0.85rem' }}>
                    <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.3rem' }}>{f.label}</label>
                    <input style={{ width: '100%', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: '0.85rem', padding: '0.65rem 0.85rem', outline: 'none', boxSizing: 'border-box' }}
                      type={f.type} placeholder={f.ph} value={(form as any)[f.k]} onChange={set(f.k)} required
                      onFocus={e => e.target.style.borderColor = '#2dd4bf'}
                      onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.12)'} />
                  </div>
                ))}
                {err && <div style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: '0.55rem 0.75rem', color: '#f87171', fontSize: '0.72rem', marginBottom: '0.75rem' }}>{err}</div>}
                <button type="submit" style={{ width: '100%', background: 'linear-gradient(135deg,#0d9488,#0f4c81)', color: '#fff', fontWeight: 700, fontSize: '0.85rem', padding: '0.7rem', borderRadius: 9, border: 'none', cursor: 'pointer', marginBottom: '1rem' }}>
                  Sign In →
                </button>
                {/* Demo accounts */}
                <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 9, padding: '0.75rem', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div style={{ fontSize: '0.6rem', color: '#475569', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'JetBrains Mono' }}>Quick Access</div>
                  {[{ e: 'analyst@imd.gov.in', p: 'imd2026', r: 'IMD Analyst' }, { e: 'demo@cyclonex.in', p: 'demo123', r: 'Observer' }].map(u => (
                    <button key={u.e} type="button" onClick={() => setForm(f => ({ ...f, email: u.e, password: u.p }))}
                      style={{ display: 'flex', justifyContent: 'space-between', width: '100%', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 7, padding: '0.4rem 0.6rem', cursor: 'pointer', marginBottom: 4, color: '#94a3b8', fontSize: '0.68rem' }}>
                      <span style={{ fontFamily: 'JetBrains Mono', color: '#2dd4bf', fontSize: '0.65rem' }}>{u.e}</span>
                      <span>{u.r}</span>
                    </button>
                  ))}
                </div>
              </form>
            ) : (
              <form onSubmit={handleRegister}>
                <div style={{ textAlign: 'center', marginBottom: '1.1rem' }}>
                  <div style={{ fontWeight: 800, fontSize: '1rem', color: '#f1f5f9', marginBottom: 3 }}>Create Account</div>
                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Join CycloNex for dashboard access</div>
                </div>
                {[
                  [{ k: 'name', label: 'Full Name', type: 'text', ph: 'Dr. Rao' }, { k: 'email', label: 'Email', type: 'email', ph: 'you@imd.gov.in' }],
                  [{ k: 'org', label: 'Organisation', type: 'text', ph: 'IMD / ISRO / TERI' }, { k: 'location', label: 'Station', type: 'text', ph: 'Chennai' }],
                  [{ k: 'password', label: 'Password', type: 'password', ph: 'min 6 chars' }, { k: 'confirm', label: 'Confirm', type: 'password', ph: 'repeat' }],
                ].map((row, ri) => (
                  <div key={ri} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem', marginBottom: '0.6rem' }}>
                    {row.map(f => (
                      <div key={f.k}>
                        <label style={{ display: 'block', fontSize: '0.68rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>{f.label}</label>
                        <input style={{ width: '100%', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: '0.78rem', padding: '0.55rem 0.7rem', outline: 'none', boxSizing: 'border-box' }}
                          type={f.type} placeholder={f.ph} value={(form as any)[f.k]} onChange={set(f.k)} required
                          onFocus={e => e.target.style.borderColor = '#2dd4bf'}
                          onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.12)'} />
                      </div>
                    ))}
                  </div>
                ))}
                <div style={{ marginBottom: '0.75rem' }}>
                  <label style={{ display: 'block', fontSize: '0.68rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.25rem' }}>Role</label>
                  <select style={{ width: '100%', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: '0.78rem', padding: '0.55rem 0.7rem', outline: 'none', cursor: 'pointer' }} value={form.role} onChange={set('role')}>
                    {['Meteorologist', 'Forecaster', 'Researcher', 'Disaster Manager', 'Observer'].map(r => <option key={r} style={{ background: '#0f172a' }}>{r}</option>)}
                  </select>
                </div>
                {err && <div style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: '0.55rem 0.75rem', color: '#f87171', fontSize: '0.72rem', marginBottom: '0.65rem' }}>{err}</div>}
                <button type="submit" style={{ width: '100%', background: 'linear-gradient(135deg,#0d9488,#0f4c81)', color: '#fff', fontWeight: 700, fontSize: '0.85rem', padding: '0.7rem', borderRadius: 9, border: 'none', cursor: 'pointer' }}>
                  Create Account →
                </button>
              </form>
            )}
            <div style={{ marginTop: '1rem', textAlign: 'center', fontSize: '0.6rem', color: '#334155', lineHeight: 1.6 }}>
              ⚠ Research prototype · SIH 2026 · Not an official IMD system
            </div>
          </div>
        </div>
      </div>

      {/* Help widget on auth page too */}
      <HelpWidget />
    </div>
  );
}
