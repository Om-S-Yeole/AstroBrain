FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
# 8000 for websocket backend
# 8501 for streamlit UI
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "app.websocket_server"]