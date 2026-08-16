# Frontend image: the React/assistant-ui chat UI (frontend/), built with
# Vite and served as static files.
#
# VITE_API_BASE_URL is baked in at BUILD time (Vite inlines
# import.meta.env.* into the bundle -- see frontend/src/api.ts), not at
# container-run time, because this is code that runs in the person's
# BROWSER, not inside the docker network. It has to be a URL the browser
# itself can reach -- the backend's PUBLISHED host port (default
# http://localhost:8001, matching docker-compose.yml's `ports:` mapping
# for the backend service), never the backend's docker-compose SERVICE
# NAME (e.g. "backend"), which only resolves inside the compose network,
# not on the host machine's browser.
#
# Build from the PROJECT ROOT:
#   docker build -f docker/frontend.Dockerfile \
#       --build-arg VITE_API_BASE_URL=http://localhost:8001 \
#       -t inmind-frontend:latest .

# ---- Stage 1: build ---------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_API_BASE_URL=http://localhost:8001
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ---- Stage 2: serve -----------------------------------------------------
# Plain static files after the build -- no need to carry Vite, npm, or
# node_modules into the image that actually runs. `serve` is a small,
# well-known static file server; swap for nginx if you'd rather.
FROM node:22-alpine

WORKDIR /app
RUN npm install -g serve

COPY --from=build /app/frontend/dist ./dist

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD wget -qO- http://localhost:8080/ > /dev/null || exit 1

CMD ["serve", "-s", "dist", "-l", "8080"]
