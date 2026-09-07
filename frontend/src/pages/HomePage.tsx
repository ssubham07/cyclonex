// Redirect shim — the new Dashboard contains all functionality
export default function HomePage({ onNavigate }: { onNavigate: (p: string) => void }) {
  onNavigate('home');
  return null;
}
