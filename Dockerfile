FROM python:3.10-slim

WORKDIR /app

# Копируем файлы проекта
COPY requirements.txt .
COPY main.py .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем папки для данных и результатов (на случай если они монтируются)
RUN mkdir -p data results

# Запускаем скрипт
CMD ["python", "main.py"]