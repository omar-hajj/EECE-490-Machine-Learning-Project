FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt /app/requirements-lock.txt
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY app /app/app
COPY scripts /app/scripts
COPY config /app/config
COPY artifacts /app/artifacts
COPY results/metrics.csv /app/results/metrics.csv
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
