FROM python:3.10-slim-bullseye

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 1: Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Upgrade pip
RUN pip install --upgrade pip

# Step 3: Install Python dependencies
RUN pip install malaya==5.1.1
RUN pip install git+https://github.com/mesolitica/malaya-speech
RUN pip install git+https://github.com/mesolitica/MeloTTS-MS
RUN pip install phonemizer
RUN pip install transformers==4.30.2
RUN pip install fastapi uvicorn

# Working directory
WORKDIR /workspace

# Copy your inference script into the container
COPY melottsms.py .

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "melottsms:app", "--host", "0.0.0.0", "--port", "8000"]
