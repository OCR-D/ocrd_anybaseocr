# README file for Test nonText area segment

Filename : ocrd-anybaseocr-tiseg.py

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
This function takes a document image as input and do the skew correction of that document.

# Usage:
```sh
python ocrd-anybaseocr-tiseg.py -m (path to met input file) -I (Input group name) -O (Output group name) -w (Working directory)
	[-p (path to parameter file) -o (METs output filename)]
```
# Example:
```sh
python ocrd-anybaseocr-tiseg.py \
	-m work_dir/mets.xml \
	-I OCR-D-IMG-CROP \
	-O OCR-D-IMG-TISEG \
	-w work_dir
```
