ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/kontakt" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_anybaseocr" \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_anybaseocr" \
    org.opencontainers.image.source="https://github.com/OCR-D/ocrd_anybaseocr" \
    org.opencontainers.image.documentation="https://github.com/OCR-D/ocrd_anybaseocr/blob/${VCS_REF}/README.md" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE

WORKDIR /build/ocrd_anybaseocr
COPY setup.py .
COPY ocrd_anybaseocr/ocrd-tool.json .
COPY ocrd_anybaseocr ./ocrd_anybaseocr
COPY requirements.txt .
COPY README.md .
RUN pip install . \
	&& rm -rf /build/ocrd_anybaseocr

RUN ocrd resmgr download -l system ocrd-anybaseocr-layout-analysis mapping_densenet.pickle && \
    ocrd resmgr download -l system ocrd-anybaseocr-layout-analysis structure_analysis && \
    ocrd resmgr download -l system ocrd-anybaseocr-dewarp latest_net_G.pth && \
    ocrd resmgr download -l system ocrd-anybaseocr-tiseg seg_model  && \
    ocrd resmgr download -l system ocrd-anybaseocr-block-segmentation  block_segmentation_weights.h5  && \
    # clean possibly created log-files/dirs of ocrd_network logger to prevent permission problems
    rm -rf /tmp/ocrd_*

WORKDIR /data
VOLUME ["/data"]
