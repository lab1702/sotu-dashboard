# Use Python image as base
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY dashboard.py .
COPY presidential_speeches/ ./presidential_speeches/

# Expose the port that Streamlit runs on
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/_stcore/health || exit 1

# Run the dashboard with Streamlit
CMD ["streamlit", "run", "dashboard.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.headless=true"]
