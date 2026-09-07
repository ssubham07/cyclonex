import { useState, useEffect } from 'react';
import { RegisterPage, loadSession, saveSession, clearSession, HelpWidget } from './pages/AuthPage';
import Dashboard from './pages/Dashboard';

interface User { name: string; email: string; org: string; role: string; location: string; password: string; }

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const session = loadSession();
    if (session) setUser(session);
    setReady(true);
  }, []);

  const handleAuth  = (u: User) => { saveSession(u); setUser(u); };
  const handleLogout = () => { clearSession(); setUser(null); };

  if (!ready) return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#091818' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem', animation: 'spin 1.5s linear infinite', display: 'inline-block' }}>🌀</div>
        <div style={{ fontSize: '0.82rem', color: '#2dd4bf', fontFamily: 'JetBrains Mono' }}>Loading CycloNex…</div>
      </div>
    </div>
  );

  if (!user) return <RegisterPage onAuth={handleAuth} />;

  return (
    <>
      <Dashboard user={user} onLogout={handleLogout} />
      <HelpWidget />
    </>
  );
}
