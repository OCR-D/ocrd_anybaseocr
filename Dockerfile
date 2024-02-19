FROM ocrd/core-cuda:v2.63.0 AS base
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/OCR-D/ocrd_anybaseocr/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_anybaseocr" \
    org.label-schema.build-date=$BUILD_DATE

WORKDIR /build
COPY setup.py .
COPY ocrd_anybaseocr/ocrd-tool.json .
COPY ocrd_anybaseocr ./ocrd_anybaseocr
COPY requirements.txt .
COPY README.md .
RUN pip install . \
	&& rm -rf /build

WORKDIR /data
VOLUME ["/data"]
