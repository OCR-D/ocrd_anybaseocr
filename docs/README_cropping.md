# README file for Page Cropping component

Filename : ocrd-anybaseocr-cropping.py

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
This function takes a document image as input and crops/selects the page content area only (that's mean remove textual noise as well as any other noise around page content area)


# LICENSE
```sh
 Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
 Apache License 2.0

 A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.
```

# Usage:
```sh
ocrd-anybaseocr-cropping -m (path to METs input file) -I (Input group name) -O (Output group name) 
	[-p (path to parameter file) -o (METs output filename)]
```
# Example:
```sh
ocrd-anybaseocr-cropping \
   -m mets.xml \
   -I OCR-D-IMG-DESKEW \
   -O OCR-D-IMG-CROP
```
