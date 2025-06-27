# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and uv.lock from parent directory context
COPY pyproject.toml uv.lock ./

# Install uv package manager
RUN pip install uv

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy the app directory
COPY app/ /code/app/

# Copy other necessary directories for imports
COPY data/ /code/data/
COPY ml/ /code/ml/
COPY models/ /code/models/
COPY pipelines/ /code/pipelines/

# Expose port 80
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
