# Use python slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies needed for your project + display libs for Tkinter
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libx11-6 libxext6 libxrender1 libxtst6 libxi6 libglu1-mesa && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire app
COPY . .

# Default command (can be overridden)
CMD ["python", "saga_app.py"]
