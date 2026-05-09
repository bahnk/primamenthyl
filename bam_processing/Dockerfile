FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md ./
COPY bam_processing ./bam_processing

RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "bam-processing"]
