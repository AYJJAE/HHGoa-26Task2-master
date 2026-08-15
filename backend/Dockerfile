# Multi-stage build for smaller final image
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download NLTK data (used by context_gate.py)
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/nltk_data')"

# Stage 2: Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only (curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /nltk_data /usr/share/nltk_data

# Copy the backend code
COPY . .

# Environment variables (overridable at runtime)
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/share/nltk_data

# Expose the port FastAPI runs on
EXPOSE 8000

# Health check — verifies the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
