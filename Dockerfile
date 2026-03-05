# --- Stage 1: Build Stage ---
FROM python:3.12-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    cmake \
    git \
    libstdc++ \
    linux-headers \
    gcompat \
    openblas-dev

# Install build tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install setuptools>=68.0 wheel setuptools-scm>=8.0

# Pre-build heavy C-extension dependencies as wheels
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel -r requirements.txt -w /wheels

# Copy necessary files for the build
# We need .git to allow setuptools-scm to resolve the version
COPY .git/ .git/
COPY pyproject.toml .
COPY README.md .
COPY f1pred/ f1pred/

# Build the application wheel (this also generates f1pred/_version.py automatically)
RUN python -m pip wheel . --no-deps -w dist

# --- Stage 2: Final Runtime Stage ---
FROM python:3.12-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/app/.cache/matplotlib

WORKDIR /app

# Install runtime system dependencies
RUN apk add --no-cache \
    libgomp \
    libstdc++ \
    gcompat \
    openblas

# Install Python dependencies first (for better caching)
# Copy the built dependency wheels from the builder stage and install them
COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy the built application wheel from the builder stage and install it
COPY --from=builder /app/dist/*.whl /tmp/app_wheel/
RUN pip install --no-cache-dir /tmp/app_wheel/*.whl && rm -rf /tmp/app_wheel

# Copy only required runtime files
COPY main.py config.yaml calibration_weights.json ./

# Create cache directories with appropriate permissions
RUN mkdir -p .cache/http_cache .cache/fastf1 .cache/matplotlib && \
    chmod -R 777 .cache

# Expose the port the app runs on
EXPOSE 8000

# Run the web server by default
ENTRYPOINT ["python", "main.py"]
CMD ["--web", "--host", "0.0.0.0", "--port", "8000"]
