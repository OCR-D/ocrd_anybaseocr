FROM ocrd/core:v2.62.0 AS base

WORKDIR /build-ocrd_anybaseocr
COPY setup.py .
COPY ocrd_anybaseocr/ocrd-tool.json .
COPY ocrd_anybaseocr ./ocrd_anybaseocr
COPY requirements.txt .
COPY README.md .
RUN pip install . \
	&& rm -rf /build-ocrd_anybaseocr

WORKDIR /data
