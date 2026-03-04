# --- Stage 1: Build Stage ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install build tools
RUN pip install --no-cache-dir setuptools>=68.0 wheel setuptools-scm>=8.0

# Copy only files needed for the build
COPY pyproject.toml .
COPY README.md .
COPY f1pred/ f1pred/
COPY .git/ .git/

# Build the wheel (this also generates f1pred/_version.py automatically)
RUN python -m pip wheel . --no-deps -w dist

# --- Stage 2: Final Runtime Stage ---
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist/*.whl .

# Install the wheel
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy remaining project files (main.py, configs, etc.)
COPY . .

# Remove unnecessary files from the image
RUN rm -rf .git f1pred/ tests/ pyproject.toml

# Create cache directories
RUN mkdir -p .cache/http_cache .cache/fastf1

# Expose the port the app runs on
EXPOSE 8000

# Run the web server by default
# Note: Since the package is installed, we can also use a script or module execution.
ENTRYPOINT ["python", "main.py"]
CMD ["--web", "--host", "0.0.0.0", "--port", "8000"]
