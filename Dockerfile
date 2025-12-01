FROM python:3.10-slim-buster AS builder
ENV PYTHONBUFFERED=1
ENV APP_HOME=/usr/src/app

WORKDIR $APP_HOME
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim-buster
WORKDIR /usr/src/app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

COPY src/ src/
COPY .env.example .
COPY assets/ assets/
COPY models/ models/
RUN chmod -R 777 models/

EXPOSE 8080

ENV FLASK_APP=app.py

CMD ["gunicorn", "src.app:app", "--bind", "0.0.0.0:8080", "--workers", "4"]