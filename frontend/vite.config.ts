import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on 5173 (assistant-ui / Vite's usual default) -- matches the
// origin agents/api.py's CORSMiddleware allows out of the box
// (AGENT_API_CORS_ORIGINS). Change both together if you move this.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
