FROM node:20-alpine

WORKDIR /app

# Copy dependency configuration
COPY package.json ./

# Install npm modules using faster mirror registry
RUN npm install --registry=https://registry.npmmirror.com

# Copy source assets
COPY . .

EXPOSE 5173

# Vite server must bind to host interface 0.0.0.0 inside container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
