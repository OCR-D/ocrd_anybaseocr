# Document Preprocessing and Segmentation

[![CircleCI](https://circleci.com/gh/OCR-D/ocrd_anybaseocr.svg?style=svg)](https://circleci.com/gh/OCR-D/ocrd_anybaseocr)
[![PyPI](https://img.shields.io/pypi/v/ocrd_anybaseocr.svg)](https://pypi.org/project/ocrd_anybaseocr/)


> Tools to preprocess and segment scanned images for OCR-D

   * [Installing](#installing)
   * [Tools](#tools)
      * [Binarizer](#binarizer)
      * [Deskewer](#deskewer)
      * [Cropper](#cropper)
      * [Dewarper](#dewarper)
      * [Text/Non-Text Segmenter](#textnon-text-segmenter)
      * [Block Segmenter](#block-segmenter)
      * [Textline Segmenter](#textline-segmenter)
      * [Document Analyser](#document-analyser)
   * [Testing](#testing)
   * [License](#license)

# Installing

Requires Python >= 3.6.

1. Create a new `venv` unless you already have one

        python3 -m venv venv

2. Activate the `venv`

        source venv/bin/activate

3. To install from source, get GNU make and do:

        make install

   There are also prebuilds available on PyPI:

        pip install ocrd_anybaseocr

(This will install both PyTorch and TensorFlow, along with their dependents.)

# Tools

All tools, also called _processors_, abide by the [CLI specifications](https://ocr-d.de/en/spec/cli) for [OCR-D](https://ocr-d.de), which roughly looks like:

    ocrd-<processor-name> [-m <path to METs input file>] -I <input group> -O <output group> [-p <path to parameter file>]* [-P <param name> <param value>]*

## Binarizer

### Method Behaviour 
For each page (or sub-segment), this processor takes a scanned colored / gray scale document image as input and computes a binarized (black and white) image.

Implemented via rule-based methods (percentile based adaptive background estimation in Ocrolib).
 
### Example

    ocrd-anybaseocr-binarize -I OCR-D-IMG -O OCR-D-BIN -P operation_level line -P threshold 0.3


## Deskewer

### Method Behaviour 
For each page (or sub-segment), this processor takes a document image as input and computes the skew angle of that. It also annotates a deskewed image. 

The input images have to be binarized for this module to work.

Implemented via rule-based methods (binary projection profile entropy maximization in Ocrolib).
 
### Example

    ocrd-anybaseocr-deskew -I OCR-D-BIN -O OCR-D-DESKEW -P maxskew 5.0 -P skewsteps 20 -P operation_level page

## Cropper

### Method Behaviour 
For each page, this processor takes a document image as input and computes the border around the page content area (i.e. removes textual noise as well as any other noise around the page frame). It also annotates a cropped image.

The input image need not be binarized, but should be deskewed for the module to work optimally.

Implemented via rule-based methods (gradient-based line segment detection and morphology based textline detection).
 
### Example:

    ocrd-anybaseocr-crop -I OCR-D-DESKEW -O OCR-D-CROP -P rulerAreaMax 0 -P marginLeft 0.1

## Dewarper

### Method Behaviour 
For each page, this processor takes a document image as input and computes a morphed image which will make the text lines straight if they are curved.

The input image has to be binarized for the module to work, and should be cropped and deskewed for optimal quality.

Implemented via data-driven methods (neural GAN conditional image model trained with pix2pixHD/Pytorch).
 
### Models

    ocrd resmgr download ocrd-anybaseocr-dewarp '*'

### Example

    ocrd-anybaseocr-dewarp -I OCR-D-CROP -O OCR-D-DEWARP -P resize_mode none -P gpu_id -1

## Text/Non-Text Segmenter

### Method Behaviour 
For each page, this processor takes a document image as an input and computes two images, separating the text and non-text parts.

The input image has to be binarized for the module to work, and should be cropped and deskewed for optimal quality.

Implemented via data-driven methods (neural pixel classifier model trained with Tensorflow/Keras).
 
### Models

    ocrd resmgr download ocrd-anybaseocr-tiseg '*'

### Example

    ocrd-anybaseocr-tiseg -I OCR-D-DEWARP -O OCR-D-TISEG -P use_deeplr true

## Block Segmenter

### Method Behaviour 
For each page, this processor takes the raw document image as an input and computes a text region segmentation for it (distinguishing various types of text blocks).

The input image need not be binarized, but should be deskewed for the module to work optimally.

Implemented via data-driven methods (neural Mask-RCNN instance segmentation model trained with Tensorflow/Keras).
 
### Models

    ocrd resmgr download ocrd-anybaseocr-block-segmentation '*'

### Example

    ocrd-anybaseocr-block-segmentation -I OCR-D-TISEG -O OCR-D-BLOCK -P active_classes '["page-number", "paragraph", "heading", "drop-capital", "marginalia", "caption"]' -P min_confidence 0.8 -P post_process true

## Textline Segmenter

### Method Behaviour 
For each page (or region), this processor takes a cropped document image as an input and computes a textline segmentation for it.

The input image should be binarized and deskewed for the module to work. 

Implemented via rule-based methods (gradient and morphology based line estimation in Ocrolib).
 
### Example

    ocrd-anybaseocr-textline -I OCR-D-BLOCK -O OCR-D-LINE -P operation_level region

## Document Analyser

### Method Behaviour 
For the whole document, this processor takes all the cropped page images and their corresponding text regions as input and computes the logical structure (page types and sections).

The input image should be binarized and segmented for this module to work.

Implemented via data-driven methods (neural Inception-V3 image classification model trained with Tensorflow/Keras).

### Models

    ocrd resmgr download ocrd-anybaseocr-layout-analysis '*'

### Example

    ocrd-anybaseocr-layout-analysis -I OCR-D-LINE -O OCR-D-STRUCT

## Testing

To test the tools under realistic conditions (on OCR-D workspaces),
download [OCR-D/assets](https://github.com/OCR-D/assets). In particular,
the code is tested with the [dfki-testdata](https://github.com/OCR-D/assets/tree/master/data/dfki-testdata)
dataset.

To download the data:

    make assets

To run module tests:

    make test

To run processor/workflow tests:

    make cli-test

## License


```
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
```
