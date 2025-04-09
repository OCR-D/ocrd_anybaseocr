testdir = tests

export

CUDA_VISIBLE_DEVICES=0
SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
PIP_INSTALL = $(PIP) install
LOG_LEVEL = INFO
PYTHONIOENCODING=utf8

TESTDATA = $(testdir)/assets/dfki-testdata/data

TESTS=tests

# Tag to publish docker image to
DOCKER_TAG = ocrd/anybaseocr

# BEGIN-EVAL makefile-parser --make-help Makefile

.PHONY: help
help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps                                  Install python deps via pip"
	@echo "    install                               Install"
	@echo "    repo/assets                           Clone OCR-D/assets to ./repo/assets"
	@echo "    assets-clean                          Remove assets"
	@echo "    assets                                Setup test assets"
	@echo "    test                                  Run unit tests"
	@echo "    cli-test                              Run CLI tests"
	@echo "    test-crop                             Test cropping CLI"
	@echo "    test-layout-analysis                  Test document structure analysis CLI"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    DOCKER_TAG  Tag to publish docker image to"

# END-EVAL

# Install python deps via pip
.PHONY: deps
deps:
	$(PIP_INSTALL) -r requirements.txt

# Install
install:
	$(PIP_INSTALL) .

# Install
install-dev: PIP_INSTALL = $(PIP) install -e
install-dev: install

#
# Assets
#


.PHONY: docker
docker:
	docker build \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t '$(DOCKER_TAG)' .

# Clone OCR-D/assets to ./repo/assets
repo/assets:
	mkdir -p $(dir $@)
	git clone https://github.com/OCR-D/assets "$@"

# Remove assets
.PHONY: assets-clean
assets-clean:
	rm -rf $(testdir)/assets

# Setup test assets
.PHONY: assets
assets: repo/assets
	mkdir -p $(testdir)/assets
	cp -r -t $(testdir)/assets repo/assets/data/*
#
# Tests
#

# Run unit tests
.PHONY: test
test: assets-clean assets
	$(PYTHON) -m pytest --continue-on-collection-errors $(TESTS)

# Run CLI tests
.PHONY: cli-test
cli-test: assets-clean assets
cli-test: test-crop test-layout

# Test cropping CLI
.PHONY: test-crop
test-crop:
	ocrd-anybaseocr-crop -m $(TESTDATA)/mets.xml -I DESKEW -O CROP-TEST

# Test layout-analysis CLI
.PHONY: test-layout
test-layout:
	ocrd-anybaseocr-layout-analysis -m $(TESTDATA)/mets.xml -I CROP -O LAYOUT-TEST
