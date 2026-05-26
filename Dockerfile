FROM node:20-slim

# Install system dependencies (postgresql-client-16 matches letta-postgres pg16)
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
       https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
       > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    postgresql-client-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json tsconfig.json build.js ./
RUN npm install --production=false

COPY src ./src
COPY memory ./memory
COPY scripts ./scripts
COPY letta ./letta

# Install Python dependencies for processing scripts
RUN pip3 install --no-cache-dir --break-system-packages -r scripts/conversation_processing/requirements.txt

# Letta ingest / register_tools (maintenance pipeline)
RUN pip3 install --no-cache-dir --break-system-packages -r letta/requirements.txt

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


