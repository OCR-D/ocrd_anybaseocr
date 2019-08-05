# docanalysis

[![CircleCI](https://circleci.com/gh/syedsaqibbukhari/docanalysis.svg?style=svg)](https://circleci.com/gh/syedsaqibbukhari/docanalysis)

> Tools for preprocessing scanned images for OCR

# Installing

To install anyBaseOCR dependencies system-wide:

    $ sudo pip install .

Alternatively, dependencies can be installed into a Virtual Environment:

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install -e .

## Tools included

To see how to run binarization, deskew, crop and dewarp method, please follow corresponding below files :

   * [README_binarize.md](README_binarize.md) instruction for binarization method
   * [README_deskew.md](README_deskew.md) instruction for deskew method
   * [README_cropping.md](README_cropping.md) instruction for cropping method
   * [README_dewarp.md](README_dewarp.md) instruction for dewarp method

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
