# Document Preprocessing and Segmentation

[![CircleCI](https://circleci.com/gh/mjenckel/OCR-D-LAYoutERkennung.svg?style=svg)](https://circleci.com/gh/mjenckel/OCR-D-LAYoutERkennung)

> Tools for preprocessing scanned images for OCR

# Installing

To install anyBaseOCR dependencies system-wide:

    $ sudo pip install .

Alternatively, dependencies can be installed into a Virtual Environment:

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install -e .

#Tools

## Binarizer

### Method Behaviour 
 This function takes a scanned colored /gray scale document image as input and do the black and white binarize image.
 
 #### Usage:
```sh
ocrd-anybaseocr-binarize -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```

#### Example: 
```sh
ocrd-anybaseocr-binarize \
   -m mets.xml \
   -I OCR-D-IMG \
   -O OCR-D-IMG-BIN \
   -O OCR-D-PAGE-BIN
```

## Deskewer

### Method Behaviour 
 This function takes a document image as input and do the skew correction of that document.
 
 #### Usage:
```sh
ocrd-anybaseocr-deskew -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```

#### Example: 
```sh
ocrd-anybaseocr-deskew \
  -m mets.xml \
  -I OCR-D-PAGE-BIN \
  -O OCR-D-IMG-DESKEW \
  -O OCR-D-PAGE-DESKEW
```

## Cropper

### Method Behaviour 
 This function takes a document image as input and crops/selects the page content area only (that's mean remove textual noise as well as any other noise around page content area)
 
 #### Usage:
```sh
ocrd-anybaseocr-crop -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```

#### Example: 
```sh
ocrd-anybaseocr-crop \
   -m mets.xml \
   -I OCR-D-PAGE-DESKEW \
   -O OCR-D-IMG-CROP \
   -O OCR-D-PAGE-CROP
```


## Dewarper

### Method Behaviour 
 This function takes a document image as input and make the text line straight if its curved.
 
 #### Usage:
```sh
ocrd-anybaseocr-dewarp -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```


#### Example: 
```sh
CUDA_VISIBLE_DEVICES=0 ocrd-anybaseocr-dewarp \
   -m mets.xml \
   -I OCR-D-PAGE-CROP \
   -O OCR-D-IMG-DEWARP \
   -O OCR-D-PAGE-DEWARP
```

## Text/Non-Text Segmenter

### Method Behaviour 
 This function takes a document image as an input and separates the text and non-text part from the input document image.
 
 #### Usage:
```sh
ocrd-anybaseocr-tiseg -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```

#### Example: 
```sh
ocrd-anybaseocr-tiseg \
   -m mets.xml \
   -I OCR-D-PAGE-CROP \
   -O OCR-D-IMG-TISEG \
   -O OCR-D-PAGE-TISEG
```

## Textline Segmenter

### Method Behaviour 
 This function takes a cropped document image as an input and segment the image into textline images.
 
 #### Usage:
```sh
ocrd-anybaseocr-textline -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```

#### Example: 
```sh
ocrd-anybaseocr-textline \
   -m mets.xml \
   -I OCR-D-PAGE-TISEG \
   -O OCR-D-IMG-TL \
   -O OCR-D-PAGE-TL
```


## Testing

To test the tools, download [OCR-D/assets](https://github.com/OCR-D/assets). In
particular, the code is tested with the
[dfki-testdata](https://github.com/OCR-D/assets/tree/master/data/dfki-testdata)
dataset.

Run `make test` to run all tests.

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
