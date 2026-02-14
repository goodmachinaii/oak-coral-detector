FROM debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
 && rm -rf /var/lib/apt/lists/*

# Coral repo (python3-pycoral + tflite runtime compatible con py3.9)
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pycoral python3-tflite-runtime \
    python3-flask python3-pil python3-numpy \
    libedgetpu1-std \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/app.py /app/app.py

EXPOSE 8765
CMD ["python3", "/app/app.py"]
