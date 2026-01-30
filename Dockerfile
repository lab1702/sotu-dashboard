# Use Python image as base
FROM python:3.14-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (tokenizers, taggers, NER models)
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
    nltk.download('maxent_ne_chunker', quiet=True); \
    nltk.download('maxent_ne_chunker_tab', quiet=True); \
    nltk.download('words', quiet=True)"

# Copy application code and data
COPY dashboard.py .
COPY analysis/ ./analysis/
COPY views/ ./views/
COPY utils/ ./utils/
COPY presidential_speeches/ ./presidential_speeches/

# Expose the port that Streamlit runs on
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/_stcore/health || exit 1

# Run the dashboard with Streamlit
CMD ["streamlit", "run", "dashboard.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.headless=true"]
