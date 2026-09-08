# Builds a container that serves the Flask app with gunicorn.
# Works on Render, Railway, Fly.io, Google Cloud Run, or any host that can
# run a Docker image - none of them require the model to be retrained inside
# the container, since model/artifacts/ ships with the image.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY feature_extraction/ feature_extraction/
COPY model_inference/ model_inference/
COPY response/ response/
COPY client/ client/
COPY model/artifacts/ model/artifacts/

EXPOSE 8000
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} app:app
