# Start from the official MLflow image
FROM ghcr.io/mlflow/mlflow:latest

# Install system dependencies (libpq-dev) needed to compile psycopg2
# Note: psycopg2-binary often works without these, but including them ensures robustness.
# Update the package list
RUN apt-get update && \
    # Install the PostgreSQL client libraries
    apt-get install -y libpq-dev gcc && \
    # AND THE S3 CLIENT LIBRARY (boto3)
    pip install --no-cache-dir psycopg2-binary boto3 && \
    # Clean up the cache to keep the image size small
    rm -rf /var/lib/apt/lists/*

# Install the PostgreSQL Python driver (psycopg2-binary)
# We use psycopg2-binary for easier installation inside the container
RUN pip install --no-cache-dir psycopg2-binary
