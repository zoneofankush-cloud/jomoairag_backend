# Build once (and rebuild whenever WHITELISTED_LIBRARIES in sandbox.py changes):
# docker build -t my-python-sandbox:latest -f sandbox.Dockerfile .
# docker network create sandbox-net
# docker run -p 8000:8000 -v /aapke/server/ka/path/uploads:/app/app/uploads my-backend-image
#docker build -t my-python-sandbox:latest -f tools/Dockerfile .
FROM python:3.10-slim

# Keep this list in sync with WHITELISTED_LIBRARIES in sandbox.py
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    matplotlib \
    scipy \
    openpyxl \
    scikit-learn

RUN useradd -m sandboxuser
USER sandboxuser
WORKDIR /app