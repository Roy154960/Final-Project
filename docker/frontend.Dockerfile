# Frontend image: the React/assistant-ui chat UI (frontend/), built with
# Vite, then served by nginx -- which now does real reverse-proxy work,
# not just static file serving (see docker/nginx.conf's own comment).
#
# VITE_API_BASE_URL is baked in at BUILD time (Vite inlines
# import.meta.env.* into the bundle -- see frontend/src/api.ts), not at
# container-run time, because this is code that runs in the person's
# BROWSER, not inside the docker network. It's now a RELATIVE path
# ("/api"), not a separate host:port -- the browser calls this
# container's own origin, and nginx.conf proxies /api/* to backend
# internally. That's the whole point of this change: backend no longer
# needs its own port published to the host at all (see
# docker-compose.yml's public-net/private-net split).
#
# Build from the PROJECT ROOT:
#   docker build -f docker/frontend.Dockerfile \
#       --build-arg VITE_API_BASE_URL=/api \
#       -t inmind-frontend:latest .

# ---- Stage 1: build ---------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ---- Stage 2: serve -----------------------------------------------------
# nginx instead of the old `serve` static server -- this stage now does
# real reverse-proxy work (see docker/nginx.conf), which `serve` never
# could. COPY overwrites nginx's own default.conf outright, no need to
# rm it first.
FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD wget -qO- http://localhost:8080/ > /dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
