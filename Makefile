exec_name_prefix = ocrd-anybaseocr
testdir = tests

export

SHELL = /bin/bash
PYTHON = python
PIP = pip
LOG_LEVEL = INFO
PYTHONIOENCODING=utf8

# Tag to publish docker image to
DOCKER_TAG = ocrd/anybaseocr

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps                     Install python deps via pip"
	@echo "    install                  Install"
	@echo "    model                    Download sample model TODO Add other models here"
	@echo "    repo/assets              Clone OCR-D/assets to ./repo/assets"
	@echo "    assets-clean             Remove assets"
	@echo "    assets                   Setup test assets"
	@echo "    #test                    Run all tests"
	@echo "    test                     Run minimum sample"
	@echo "    test-binarize            Test binarization"
	@echo "    test-deskew              Test deskewing"
	@echo "    test-crop                Test cropping"
	@echo "    test-tiseg               Test text/non-text segmentation"
	@echo "    test-textline            Test textline extraction"
	@echo "    test-block-segmentation  Test block segmentation"
	@echo "    test-layout-analysis     Test document structure analysis"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    DOCKER_TAG  Tag to publish docker image to"

# END-EVAL

# Install python deps via pip
deps:
	$(PIP) install -r requirements.txt

# Install
install:
	$(PIP) install .

#
# Assets
#


# Download sample model TODO Add other models here
model: models/latest_net_G.pth
models/latest_net_G.pth:
	wget -O"$@" "https://cloud.dfki.de/owncloud/index.php/s/3zKza5sRfQB3ygy/download"
	
model: models/block_segmentation_weights.h5
models/block_segmentation_weights.h5:
	wget -O"$@" "https://cloud.dfki.de/owncloud/index.php/s/dgACCYzytxnb7Ey/download"

model: models/structure_analysis.h5
models/structure_analysis.h5:
	wget -O"$@" "https://cloud.dfki.de/owncloud/index.php/s/E85PL48Cjs8ZkJL/download"

model: models/mapping_densenet.pickle
models/mapping_densenet.pickle:
	wget -O"$@" "https://cloud.dfki.de/owncloud/index.php/s/2kpMxnMSSqS8z3X/download"
	
model: models/seg_model.hdf5
models/seg_model.hdf5:
	wget -O"$@" "https://cloud.dfki.de/owncloud/index.php/s/Qxm8baqq9Zf8brQ/download"

docker:
	docker build -t '$(DOCKER_TAG)' .

# Clone OCR-D/assets to ./repo/assets
repo/assets:
	mkdir -p $(dir $@)
	git clone https://github.com/OCR-D/assets "$@"

# Remove assets
assets-clean:
	rm -rf $(testdir)/assets

# Setup test assets
assets: repo/assets
	mkdir -p $(testdir)/assets
	cp -r -t $(testdir)/assets repo/assets/data/*
	mkdir -p models
	make model
	cp -r  models/ $(testdir)/assets/dfki-testdata/data/
#
# Tests
# Run all tests
test: test-binarize test-deskew test-crop test-tiseg test-block-segmentation test-textline test-layout-analysis

# Test binarization
test-binarize: assets-clean assets
	cd $(testdir)/assets/dfki-testdata/data && $(exec_name_prefix)-binarize -m mets.xml -I OCR-D-IMG -O OCR-D-IMG-BIN-TEST

# Test deskewing
test-deskew: assets-clean assets
	cd $(testdir)/assets/dfki-testdata/data && $(exec_name_prefix)-deskew -m mets.xml -I OCR-D-IMG-BIN-TEST -O OCR-D-IMG-DESKEW-TEST

# Test cropping
test-crop: assets-clean assets
	cd $(testdir)/assets/dfki-testdata/data && $(exec_name_prefix)-crop -m mets.xml -I OCR-D-IMG-DESKEW-TEST -O OCR-D-IMG-CROP-TEST

# Test text/non-text segmentation
test-tiseg: assets-clean assets
	cd $(testdir)/assets/dfki-testdata/data && $(exec_name_prefix)-tiseg -m mets.xml -I OCR-D-IMG-CROP-TEST -O OCR-D-IMG-TISEG-TEST

# Test block segmentation
test-block-segmentation:
	cd $(testdir)/assets/dfki-testdata/data && CUDA_VISIBLE_DEVICES=0 && $(exec_name_prefix)-block-segmentation -m mets.xml -I OCR-D-IMG-TISEG-TEST -O OCR-D-BLOCK-SEGMENT

# Test textline extraction
test-textline: assets-clean assets
	cd $(testdir)/assets/dfki-testdata/data && $(exec_name_prefix)-textline -m mets.xml -I OCR-D-BLOCK-SEGMENT -O OCR-D-IMG-TL-TEST

# Test document structure analysis
test-layout-analysis:
	cd $(testdir)/assets/dfki-testdata/data && CUDA_VISIBLE_DEVICES=0 && $(exec_name_prefix)-layout-analysis -m mets.xml -I OCR-D-IMG-BIN-TEST -O OCR-D-IMG-LAYOUT
