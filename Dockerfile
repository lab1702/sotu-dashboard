# Use Python image as base
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TAIPY_PORT=5000

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY dashboard.py .
COPY presidential_speeches/ ./presidential_speeches/

# Expose the port that Taipy runs on
EXPOSE 5000

# Health check (using wget which is more commonly available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/ || exit 1

# Run the dashboard directly with Python
CMD ["python", "dashboard.py"]
