FROM node:20-alpine

# Install wget for health checks
RUN apk add --no-cache wget

WORKDIR /app

COPY package.json tsconfig.json build.js ./
RUN npm install --production=false

COPY src ./src
COPY memory ./memory
COPY scripts ./scripts

# Build using esbuild (much faster and uses less memory than tsc)
RUN npm run build

# Copy data directories after build (they're also mounted as volumes, but this ensures they exist)
COPY conversations ./conversations
COPY files ./files

ENV PORT=4000
ENV MEMORY_DIR=./memory

EXPOSE 4000

CMD ["npm", "run", "start"]


