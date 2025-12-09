FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json tsconfig.json build.js ./
RUN npm install --production=false

COPY src ./src
COPY memory ./memory
COPY scripts ./scripts

# Install Python dependencies for processing scripts
RUN pip3 install --no-cache-dir --break-system-packages -r scripts/conversation_processing/requirements.txt

# Install Python dependencies for identity model (ML libraries)
# Use CPU-only PyTorch for compatibility
RUN pip3 install --no-cache-dir --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r scripts/identity_model/requirements.txt

# Build using esbuild (much faster and uses less memory than tsc)
RUN npm run build

# Copy data directories after build (they're also mounted as volumes, but this ensures they exist)
COPY conversations ./conversations
COPY files ./files

ENV PORT=4000
ENV MEMORY_DIR=./memory

EXPOSE 4000

CMD ["npm", "run", "start"]


