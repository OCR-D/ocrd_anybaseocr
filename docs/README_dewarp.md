# README file for Dewarping component

Filename : ocrd-anybaseocr-dewarp.py

Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
Note: 
1. this work has been done in DFKI, Kaiserslautern, Germany, as a part of the DFG research project "Scalable Methods of Text and Structure Recognition for the Full-Text Digitization of Historical Prints" Part 1.B: Image Optimization"
Link: http://gepris.dfg.de/gepris/projekt/394343055?language=en
2. The parameters values are read from ocrd-anybaseocr-parameter.json file. The values can be changed in that file.
3. The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). 
4. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).
5. Sample files are available at [OCR-D/assets](https://github.com/OCR-D/ocrd-assets/tree/master/data/dfki-testdata)

# Method Behaviour
This function takes a document image as input and make the text line straight if its curved.

# LICENSE
Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
Apache License 2.0

pix2pixHD: Copyright (C) 2017 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
All rights reserved. 
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

@inproceedings{wang2018pix2pixHD,
  title={High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs},
  author={Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Andrew Tao and Jan Kautz and Bryan Catanzaro},  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}

A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

# Environment preparation:
- python3
- scipy (i.e., pip install scipy)
- opencv-python (i.e., pip install opencv-python)
- PyTorch and torchvision for GPU support version (from http://pytorch.org)
- dominate (i.e., pip install dominate)

- Download pix2pixHD from the gitHub (https://github.com/NVIDIA/pix2pixHD). Extract zip file to the ocrd-anybaseocr script directory and rename the folder name "pix2pixHD"
- Move following files from pix2pixHD_modified folder to pix2pixHD.
	- test.py move/replace to pix2pixHD/test.py
	- visualizer.py move/replace to pix2pixHD/util/visualizer.py
- Download model from https://cloud.dfki.de/owncloud/index.php/s/3zKza5sRfQB3ygy. Copy the file into "models" folder


# Usage:
```sh
ocrd-anybaseocr-dewarp -m (path to METs input file) -I (Input group name) -O (Output group name)
	[-p (path to parameter file) -o (METs output filename)]
```

# Example: 
```sh
ocrd-anybaseocr-dewarp \
   -m mets.xml \
   -I OCR-D-IMG-CROP \
   -O OCR-D-IMG-DEWARP
```