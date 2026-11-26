import { defineConfig } from "vite";

// In development the viewer runs on Vite and proxies the API to zai-server.
const server = process.env.ZAI_SERVER ?? "http://localhost:8080";

export default defineConfig({
  // three.js (~530 kB) is its own chunk, loaded only when the 3D view is opened.
  build: { chunkSizeWarningLimit: 600 },
  server: {
    port: 5173,
    proxy: {
      "/api": server,
      "/healthz": server,
      "/ws": { target: server.replace(/^http/, "ws"), ws: true },
    },
  },
});
