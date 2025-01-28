# Use an official Python image as a base
FROM python:3.9.19-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies manually (excluding torch first)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt || true

# Install torch manually from PyTorch repository
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Expose the application port (if needed)
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]
