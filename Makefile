export

SHELL = /bin/bash
PYTHON ?= python
PIP ?= pip
PIP_INSTALL = $(PIP) install
PYTHONIOENCODING=utf8

TESTDATA = tests/assets/dfki-testdata/data
PYTEST_ARGS ?= -vv

# Tag to publish docker image to
DOCKER_TAG = ocrd/anybaseocr
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-tf2:v3.3.0

# BEGIN-EVAL makefile-parser --make-help Makefile

.PHONY: help
help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps                      Install Python dependencies via pip"
	@echo "    deps-test                 Install Python dependencies for test"
	@echo "    install                   Install Python package via pip"
	@echo "    install-dev               Install Python package in editable mode"
	@echo "    build                     Build Python package binary and source dist"
	@echo "    docker                    Build Docker image"
	@echo "    repo/assets               Clone OCR-D/assets to ./repo/assets"
	@echo "    tests/assets              Setup test data from repo/assets"
	@echo "    assets-clean              Remove tests/assets"
	@echo "    test                      Run unit tests via Pytest"
	@echo "    cli-test                  Run CLI tests"
	@echo "    test-crop                 Test cropping CLI"
	@echo "    test-layout-analysis      Test document structure analysis CLI"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    DOCKER_TAG                Tag name to build Docker image for [$(DOCKER_TAG)]"
	@echo "    PYTEST_ARGS               Pytest options for test [$(PYTEST_ARGS)]"

# END-EVAL

# Install python deps via pip
.PHONY: deps deps-test
deps:
	$(PIP_INSTALL) -r requirements.txt
deps-test:
	$(PIP_INSTALL) -r requirements.test.txt

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
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
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
	rm -rf tests/assets

# Setup test assets
tests/assets: repo/assets
	mkdir -p $@
	cp -r -t $@ $</data/*
#
# Tests
#

# Run unit tests
.PHONY: test
test: assets-clean tests/assets
	$(PYTHON) -m pytest --continue-on-collection-errors --durations=0 tests $(PYTEST_ARGS)

# Run CLI tests
.PHONY: cli-test
cli-test: assets-clean tests/assets
cli-test: test-crop test-layout

# Test cropping CLI
.PHONY: test-crop
test-crop:
	ocrd-anybaseocr-crop -m $(TESTDATA)/mets.xml -I DESKEW -O CROP-TEST

# Test layout-analysis CLI
.PHONY: test-layout
test-layout:
	ocrd-anybaseocr-layout-analysis -m $(TESTDATA)/mets.xml -I CROP -O LAYOUT-TEST
