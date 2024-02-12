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
	@echo "    ocrd_anybaseocr/pix2pixhd             Checkout pix2pixhd submodule"
	@echo "    repo/assets                           Clone OCR-D/assets to ./repo/assets"
	@echo "    assets-clean                          Remove assets"
	@echo "    assets                                Setup test assets"
	@echo "    test                                  Run unit tests"
	@echo "    cli-test                              Run CLI tests"
	@echo "    test-binarize                         Test binarization CLI"
	@echo "    test-deskew                           Test deskewing CLI"
	@echo "    test-crop                             Test cropping CLI"
	@echo "    test-tiseg                            Test text/non-text segmentation CLI"
	@echo "    test-block-segmentation               Test block segmentation CLI"
	@echo "    test-textline                         Test textline segmentation CLI"
	@echo "    test-layout-analysis                  Test document structure analysis CLI"
	@echo "    test-dewarp                           Test page dewarping CLI"
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
install: ocrd_anybaseocr/pix2pixhd
	$(PIP_INSTALL) .

ocrd_anybaseocr/pix2pixhd:
	git submodule update --init $@

#
# Assets
#


# Download sample model TODO Add other models here
.PHONY: models
models:
	ocrd resmgr download --allow-uninstalled --location cwd ocrd-anybaseocr-dewarp '*'
	ocrd resmgr download --allow-uninstalled --location cwd ocrd-anybaseocr-block-segmentation '*'
	ocrd resmgr download --allow-uninstalled --location cwd ocrd-anybaseocr-layout-analysis '*'
	ocrd resmgr download --allow-uninstalled --location cwd ocrd-anybaseocr-tiseg '*'

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
assets: repo/assets models
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
cli-test: test-binarize test-deskew test-crop test-tiseg test-textline test-layout-analysis test-dewarp

# Test binarization CLI
.PHONY: test-binarize
test-binarize: assets
	ocrd-anybaseocr-binarize -m $(TESTDATA)/mets.xml -I MAX -O BIN-TEST

# Test deskewing CLI
.PHONY: test-deskew
test-deskew: test-binarize
	ocrd-anybaseocr-deskew -m $(TESTDATA)/mets.xml -I BIN-TEST -O DESKEW-TEST

# Test cropping CLI
.PHONY: test-crop
test-crop: test-deskew
	ocrd-anybaseocr-crop -m $(TESTDATA)/mets.xml -I DESKEW-TEST -O CROP-TEST

# Test text/non-text segmentation CLI
.PHONY: test-tiseg
test-tiseg: test-crop
	ocrd-anybaseocr-tiseg -m $(TESTDATA)/mets.xml --overwrite -I CROP-TEST -O TISEG-TEST

# Test block segmentation CLI
.PHONY: test-block-segmentation
test-block-segmentation: test-tiseg
	ocrd-anybaseocr-block-segmentation -m $(TESTDATA)/mets.xml -I TISEG-TEST -O OCR-D-BLOCK-SEGMENT

# Test textline segmentation CLI
.PHONY: test-textline
test-textline: test-tiseg
	ocrd-anybaseocr-textline -m $(TESTDATA)/mets.xml -I TISEG-TEST -O TL-TEST

# Test page dewarping CLI
.PHONY: test-dewarp
test-dewarp: test-crop
	ocrd-anybaseocr-dewarp -m $(TESTDATA)/mets.xml -I CROP-TEST -O DEWARP-TEST

# Test document structure analysis CLI
.PHONY: test-layout-analysis
test-layout-analysis: test-binarize
	ocrd-anybaseocr-layout-analysis -m $(TESTDATA)/mets.xml -I BIN-TEST -O LAYOUT
