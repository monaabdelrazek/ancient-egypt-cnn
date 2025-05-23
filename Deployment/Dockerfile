FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# نسخ المتطلبات أولاً لتسريع الكاش
COPY requirements.txt ./  
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# نسخ باقي الملفات
COPY . /app

# فك ضغط الموديل
RUN 7z x brave_pharos_detection_model256.7z -o/app

# إنشاء بيئة افتراضية وتثبيت المتطلبات
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

EXPOSE 7860

CMD ["/venv/bin/python", "app.py"]
