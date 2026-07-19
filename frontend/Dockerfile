FROM node:20-slim

WORKDIR /app

# Install dependencies first (for caching)
COPY package.json package-lock.json* ./

RUN npm install --no-audit --no-fund

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
