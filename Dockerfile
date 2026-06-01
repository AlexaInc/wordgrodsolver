FROM node:20-slim

# Install system dependencies for Tesseract OCR and Sharp
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libvips42 \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory with correct ownership
WORKDIR /home/node/app
RUN chown -R node:node /home/node/app

USER node
ENV HOME=/home/node \
    PORT=7860

# Install Node dependencies first (layer caching)
COPY --chown=node:node package*.json ./
RUN npm install --omit=dev

# Copy source files
COPY --chown=node:node . .

EXPOSE 7860

CMD ["node", "bot.js"]
