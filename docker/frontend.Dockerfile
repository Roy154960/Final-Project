FROM node:22-alpine AS build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_API_BASE_URL=http://localhost:8001
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

FROM node:22-alpine

WORKDIR /app
RUN npm install -g serve

COPY --from=build /app/frontend/dist ./dist

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD wget -qO- http://localhost:8080/ > /dev/null || exit 1

CMD ["serve", "-s", "dist", "-l", "8080"]
