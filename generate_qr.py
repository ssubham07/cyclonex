"""
CycloNex QR Code + Deployment Info Generator
Generates QR codes for the public URLs and prints a deployment summary.
"""
import qrcode
import os

FRONTEND_URL = "https://cyclonex-app.loca.lt"
BACKEND_URL  = "https://cyclonex-api.loca.lt"
OUT_DIR = r"C:\Users\pradh\.gemini\antigravity\brain\6700f645-ef48-4108-b5f8-86c803482512"

os.makedirs(OUT_DIR, exist_ok=True)

def make_qr(url, filename, color="#0f4c81"):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=color, back_color="white")
    path = os.path.join(OUT_DIR, filename)
    img.save(path)
    print(f"✅ QR saved → {path}")
    return path

# Main app QR (frontend)
make_qr(FRONTEND_URL, "cyclonex_qr_app.png",   "#0f4c81")
# API QR (backend)
make_qr(BACKEND_URL,  "cyclonex_qr_api.png",   "#0d9488")

print()
print("=" * 60)
print("  🌀 CYCLONEX — LIVE DEPLOYMENT SUMMARY")
print("=" * 60)
print()
print(f"  🌐 Frontend (any device):  {FRONTEND_URL}")
print(f"  🔧 Backend API:            {BACKEND_URL}")
print(f"  📖 API Docs:               {BACKEND_URL}/docs")
print()
print("  📱 SCAN QR CODE to open on phone/tablet:")
print(f"     cyclonex_qr_app.png  ← MAIN APP")
print()
print("  🔑 Demo login:")
print("     analyst@imd.gov.in  /  imd2026")
print("     demo@cyclonex.in    /  demo123")
print()
print("  ⚠  Note: tunnels are active while this PC is on.")
print("     First visit may show a localtunnel page —")
print("     click 'Click to Continue' to reach CycloNex.")
print("=" * 60)
