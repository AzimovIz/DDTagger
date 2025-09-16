FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deep_danbooru_model.py .
COPY main.py .
COPY tagging_utils.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "--host", "0.0.0.0", "main:app"]