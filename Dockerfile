# Minimal CPU-only environment for running the analysis/evaluation scripts
# (Arms A, B and C inference) without a local venv.
#   docker build -t isro-ridge .
#   docker run --rm -v "$PWD":/work -w /work isro-ridge python <script>
# Data (data/tiles/<id>/, data/raw/) is gitignored and must exist in the
# mounted checkout.
FROM python:3.12-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/work
