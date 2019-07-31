# README file for Text/non-Text region segmentation

Filename : ocrd_anybaseocr_tiseg.py

Author: Syed Saqib Bukhari, Mohammad Mohsin Reza
Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza
Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de
Note: 
1. this work has been done in DFKI, Kaiserslautern, Germany, as a part of the DFG research project "Scalable Methods of Text and Structure Recognition for the Full-Text Digitization of Historical Prints" Part 2: Layout Analysis "
Link: http://gepris.dfg.de/gepris/projekt/394346204?language=en
2. The parameters values are read from ocrd-tools.json file. The values can be changed in that file.
3. The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). 
4. The sequence of operations is: binarization, deskewing, cropping, dewarping and text-nontext (or can also be: binarization, dewarping, deskewing, cropping and text-nontext; depends upon use-case).
5. Sample files are available at [OCR-D/assets](https://github.com/OCR-D/ocrd-assets/tree/master/data/dfki-testdata)

# Method Behaviour 
This function takes a document image as an input and separates the text and non-text part from the input document image.

# Usage:
```sh
ocrd-anybaseocr-tiseg -m (path to METs input file) -I (Input group name) -O (Output group name) [-p (path to parameter file) -o (METs output filename)]
```
# Example:
```sh
ocrd-anybaseocr-tiseg \
	-m mets.xml \
	-I OCR-D-IMG-CROP \
	-O OCR-D-IMG-TISEG
```
