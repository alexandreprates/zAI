import { defineConfig } from "vite";

// In development the viewer runs on Vite and proxies the API to zai-server.
const server = process.env.ZAI_SERVER ?? "http://localhost:8080";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": server,
      "/healthz": server,
      "/ws": { target: server.replace(/^http/, "ws"), ws: true },
    },
  },
});
