FROM python:3.10-alpine

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT 'flask_file:app'
