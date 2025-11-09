# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files first for better caching
COPY package.json package-lock.json ./

# Install dependencies
# Using npm ci for reproducible builds (requires package-lock.json)
RUN npm ci

# Copy source files (excluding node_modules, dist, etc. via .dockerignore)
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files to nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

