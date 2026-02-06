import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:4000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // Prevent proxy from buffering SSE (Server-Sent Events) responses
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes, req, res) => {
            // If the backend sends text/event-stream, disable buffering
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              // Ensure no compression/buffering on the proxy side
              res.setHeader('Cache-Control', 'no-cache')
              res.setHeader('X-Accel-Buffering', 'no')
            }
          })
        }
      }
    }
  }
})

