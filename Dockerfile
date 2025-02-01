# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=32771
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 32771

# Run the application
CMD ["python", "index.py"]