import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the Python API runs on :8000 (python -m linkstudio serve --no-browser).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
    },
  },
});
