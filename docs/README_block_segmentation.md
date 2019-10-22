# README file for Block Segmentation

Filename : ocrd_anybaseocr_block_segmentation.py

Author: Khurram Hashmi
Responsible: Khurram Hashmi
Contact Email: Khurram_Azeem.Hashmi@dfki.de

1. The parameters values are read from ocrd-tools.json file. The values can be changed in that file.
2. The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/).
3. Sample files are available at [OCR-D/assets](https://github.com/OCR-D/ocrd-assets/tree/master/data/dfki-testdata)
4. Download model from https://cloud.dfki.de/owncloud/index.php/s/tgjJQBHnzeGYqoj

# Method Behaviour 


# Usage:
```sh
CUDA_VISIBLE_DEVICES=0 ocrd-anybaseocr-block-segmentation -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename) -p params.json]
```
# Example:
```sh
CUDA_VISIBLE_DEVICES=0 ocrd-anybaseocr-block-segmentation \
	-m mets_block.xml \
	-I OCR-D-BLOCK \
	-O OCR-D-BLOCK-SEGMENT
```
