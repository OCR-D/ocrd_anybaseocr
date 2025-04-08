# Document Croppnig

[![CircleCI](https://circleci.com/gh/OCR-D/ocrd_anybaseocr.svg?style=svg)](https://circleci.com/gh/OCR-D/ocrd_anybaseocr)
[![PyPI](https://img.shields.io/pypi/v/ocrd_anybaseocr.svg)](https://pypi.org/project/ocrd_anybaseocr/)


> Tools to crop scanned images for OCR-D

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

# Tools

All tools, also called _processors_, abide by the [CLI specifications](https://ocr-d.de/en/spec/cli) for [OCR-D](https://ocr-d.de), which roughly looks like:

    ocrd-<processor-name> [-m <path to METs input file>] -I <input group> -O <output group> [-p <path to parameter file>]* [-P <param name> <param value>]*

## Cropper

### Method Behaviour 
For each page, this processor takes a document image as input and computes the border around the page content area (i.e. removes textual noise as well as any other noise around the page frame). It also annotates a cropped image.

The input image does not need to be binarized, but should be deskewed for the module to work optimally.

Implemented via rule-based methods (gradient-based line segment detection and morphology based textline detection).
 
### Example:

    ocrd-anybaseocr-crop -I OCR-D-DESKEW -O OCR-D-CROP -P rulerAreaMax 0 -P marginLeft 0.1

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
