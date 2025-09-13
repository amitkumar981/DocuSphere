# Use a specific Python 3.10 slim image for reproducible builds
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

COPY .env .

# Install OS dependencies & upgrade pip in a single layer, then clean cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential poppler-utils && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for efficient caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for the app
EXPOSE 8000

# Start Uvicorn server with multiple workers
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
