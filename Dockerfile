FROM python:3.10-slim

WORKDIR /app

# Install only required tools
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY ./src ./src
COPY ./models ./models
COPY requirements.txt requirements.txt

# Install Python libs
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]

