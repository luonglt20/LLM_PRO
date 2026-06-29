FROM python:3.12-slim

WORKDIR /app

# Install system utilities if needed for numpy/sklearn compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files into container working directory
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
