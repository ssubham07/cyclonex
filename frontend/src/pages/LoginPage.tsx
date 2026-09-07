import { useState } from 'react';

interface LoginPageProps {
  onLogin: (user: { name: string; role: string }) => void;
}

const DEMO_USERS = [
  { email: 'analyst@imd.gov.in',  password: 'imd2026', name: 'IMD Analyst',      role: 'Meteorologist' },
  { email: 'admin@cyclonex.in',   password: 'admin123', name: 'System Admin',     role: 'Administrator' },
  { email: 'viewer@sih2026.in',   password: 'sih2026',  name: 'SIH Evaluator',   role: 'Observer' },
];

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [email, setEmail]     = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]     = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(''); setLoading(true);
    setTimeout(() => {
      const user = DEMO_USERS.find(u => u.email === email.trim() && u.password === password);
      if (user) {
        onLogin({ name: user.name, role: user.role });
      } else {
        setError('Invalid credentials. Use a demo account below.');
      }
      setLoading(false);
    }, 800);
  };

  const quickLogin = (u: typeof DEMO_USERS[0]) => {
    setEmail(u.email); setPassword(u.password); setError('');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
      <div className="bg-glow" />
      <div className="bg-glow-left" />

      <div style={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: 440, padding: '0 1.5rem' }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 56, height: 56, borderRadius: 14, background: 'linear-gradient(135deg, #0d9488, #2dd4bf)', marginBottom: '1rem', fontSize: '1.6rem' }}>
            🌀
          </div>
          <div style={{ fontWeight: 800, fontSize: '1.6rem', letterSpacing: '-0.02em' }}>
            Cyclo<span style={{ color: '#2dd4bf' }}>Nex</span>
          </div>
          <div className="label-mono" style={{ marginTop: '0.4rem', color: '#1a3535' }}>
            AI Cyclone Intelligence System · SIH26070
          </div>
        </div>

        {/* Card */}
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.3rem' }}>Sign in</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Access the weather intelligence dashboard</div>
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <div className="label-mono" style={{ marginBottom: '0.4rem' }}>Email address</div>
              <input className="login-input" type="email" placeholder="analyst@imd.gov.in"
                value={email} onChange={e => setEmail(e.target.value)} required />
            </div>
            <div>
              <div className="label-mono" style={{ marginBottom: '0.4rem' }}>Password</div>
              <input className="login-input" type="password" placeholder="••••••••"
                value={password} onChange={e => setPassword(e.target.value)} required />
            </div>

            {error && (
              <div style={{ padding: '0.65rem 0.9rem', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 7, color: '#fca5a5', fontSize: '0.75rem', fontFamily: 'JetBrains Mono' }}>
                {error}
              </div>
            )}

            <button className="btn-primary" type="submit" disabled={loading} style={{ width: '100%', justifyContent: 'center', marginTop: '0.5rem' }}>
              {loading ? <><span className="spin" style={{ fontSize: '0.9rem' }}>🌀</span> Authenticating…</> : '→ Sign in'}
            </button>
          </form>

          {/* Demo accounts */}
          <div style={{ marginTop: '1.75rem', paddingTop: '1.5rem', borderTop: '1px solid var(--bg-border)' }}>
            <div className="label-mono" style={{ marginBottom: '0.75rem', fontSize: '0.55rem' }}>Demo accounts — click to fill</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {DEMO_USERS.map(u => (
                <button key={u.email} onClick={() => quickLogin(u)}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-deep)', border: '1px solid var(--bg-border)', borderRadius: 7, padding: '0.6rem 0.9rem', cursor: 'pointer', transition: 'border-color 0.2s' }}
                  onMouseOver={e => (e.currentTarget.style.borderColor = '#2dd4bf')}
                  onMouseOut={e => (e.currentTarget.style.borderColor = 'var(--bg-border)')}>
                  <div>
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.72rem', color: '#f0fafa' }}>{u.email}</div>
                    <div className="label-mono" style={{ fontSize: '0.52rem', marginTop: '0.1rem' }}>{u.role}</div>
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.62rem', color: '#2dd4bf' }}>↙ fill</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.65rem', color: '#1a3535', fontFamily: 'JetBrains Mono', lineHeight: 1.6 }}>
          ⚠ Research prototype only · Not an official IMD system<br />
          Official warnings from IMD only · SIH 2026 · Problem SIH26070
        </div>
      </div>
    </div>
  );
}
