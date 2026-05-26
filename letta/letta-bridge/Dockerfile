FROM python:3.12-slim

WORKDIR /app
COPY letta-bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY letta-bridge/app.py letta-bridge/reasoning_guard.py letta-bridge/guard_log.py .
COPY model_prefs.py .

EXPOSE 8284
# Simple liveness: the bridge answers /v1/models without needing Letta up yet.
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8284/v1/models')" || exit 1

CMD ["python", "app.py"]
