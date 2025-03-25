FROM ocrd/core-cuda-tf2:v2.70.0 AS base
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://github.com/OCR-D/ocrd_anybaseocr/issues" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_anybaseocr" \
    org.label-schema.build-date=$BUILD_DATE

WORKDIR /build/ocrd_anybaseocr
COPY setup.py .
COPY ocrd_anybaseocr/ocrd-tool.json .
COPY ocrd_anybaseocr ./ocrd_anybaseocr
COPY requirements.txt .
COPY README.md .
RUN pip install . \
	&& rm -rf /build/ocrd_anybaseocr

WORKDIR /data
VOLUME ["/data"]
