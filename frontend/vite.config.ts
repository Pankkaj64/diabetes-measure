import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Backend origin. Defaults to the documented port 8000; override with
// VITE_API_TARGET when that port is already taken by another service.
const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
