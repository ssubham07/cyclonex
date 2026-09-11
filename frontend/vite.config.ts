import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  // For production build — backend tunnel URL baked in
  define: {
    'import.meta.env.VITE_API_URL': JSON.stringify('https://cyclonex-api.loca.lt'),
  }
})
