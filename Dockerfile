# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create cache directories
RUN mkdir -p .cache/http_cache .cache/fastf1

# Expose the port the app runs on
EXPOSE 8000

# Run the web server by default
ENTRYPOINT ["python", "main.py"]
CMD ["--web", "--host", "0.0.0.0", "--port", "8000"]
