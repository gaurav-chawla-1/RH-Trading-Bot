# Use miniconda base image
FROM continuumio/miniconda3

# Set timezone to PST
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create conda environment
RUN conda create -n trading_env python=3.9 -y && \
    conda install -n trading_env -c conda-forge ta-lib pandas numpy -y

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies in conda environment
SHELL ["conda", "run", "-n", "trading_env", "/bin/bash", "-c"]
RUN pip install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p /app/data/trades /app/data/logs

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_DIR=/app/data/logs
ENV TRADES_DIR=/app/data/trades
ENV TZ=America/Los_Angeles

# Activate conda environment and set default command
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trading_env", "python", "trading_bot.py"]