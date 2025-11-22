FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement file
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code into container
COPY . /app

# Expose API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
