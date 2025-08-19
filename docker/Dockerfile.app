FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Базовые пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install python-docx requests tqdm

# (опционально) для PDF/HTML позже: pymupdf bs4 lxml tika и пр.

# Рабочая директория
WORKDIR /app

# Копируем скрипты
COPY scripts/ /app/scripts/

# Точки монтирования
VOLUME ["/data/docs", "/data/outputs", "/data/hf-cache"]

# По умолчанию ничего не запускаем — команды будем передавать через `docker compose run`
CMD ["bash"]
