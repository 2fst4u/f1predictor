# --- Stage 1: Build Stage ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install build tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install setuptools>=68.0 wheel setuptools-scm>=8.0

# Copy necessary files for the build
# We need .git to allow setuptools-scm to resolve the version
COPY .git/ .git/
COPY pyproject.toml .
COPY README.md .
COPY f1pred/ f1pred/

# Build the wheel (this also generates f1pred/_version.py automatically)
RUN python -m pip wheel . --no-deps -w dist

# --- Stage 2: Final Runtime Stage ---
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the built wheel from the builder stage and install it
COPY --from=builder /app/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy only required runtime files
COPY main.py config.yaml calibration_weights.json ./

# Create cache directories with appropriate permissions
RUN mkdir -p .cache/http_cache .cache/fastf1 && \
    chmod -R 777 .cache

# Expose the port the app runs on
EXPOSE 8000

# Run the web server by default
ENTRYPOINT ["python", "main.py"]
CMD ["--web", "--host", "0.0.0.0", "--port", "8000"]
