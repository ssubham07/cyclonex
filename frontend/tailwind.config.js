/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        teal: {
          DEFAULT: '#2dd4bf',
          dim: '#14b8a6',
          dark: '#0d9488',
        },
        bg: {
          deep: '#060f0f',
          base: '#091818',
          card: '#0d2020',
          border: '#1a3535',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
