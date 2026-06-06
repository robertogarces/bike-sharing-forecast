FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install project package
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Run dashboard
CMD ["streamlit", "run", "src/bike_sharing/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]