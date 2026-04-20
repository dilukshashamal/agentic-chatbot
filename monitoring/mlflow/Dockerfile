FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow==2.16.2

WORKDIR /mlflow

EXPOSE 5000
