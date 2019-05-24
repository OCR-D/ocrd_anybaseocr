#!/usr/bin/python

# Author: Saqib Bukhari
# Email: Saqib.Bukhari@dfki.de
# Paper: Syed Saqib Bukhari, Faisal Shafait, Thomas M. Breuel, "Improved document image segmentation algorithm using multiresolution morphology," Proc. SPIE 7874, Document Recognition and Retrieval XVIII, 78740D (24 January 2011);

#The anyBaseOCR-tiseg is licensed under the "OpenContent License". While using/refering this work, always add the reference of the following paper: 
#"Syed Saqib Bukhari, Ahmad Kadi, Mohammad Ayman Jouneh, Fahim Mahmood Mir, Andreas Dengel, “anyOCR: An Open-Source OCR System for Historical Archives”, The 14th IAPR International Conference on Document Analysis and Recognition (ICDAR 2017), Kyoto, Japan, 2017.
# URL - https://www.dfki.de/fileadmin/user_upload/import/9512_ICDAR2017_anyOCR.pdf


from scipy import ones, zeros, array, where, shape, ndimage, sum, logical_or, logical_and
import copy
from pylab import unique
from ocrolib.toplevel import *
from ParserAnybaseocr import *


def textimageseg(imf):
	# I: binarized-input-image; imftext: output-text-portion.png; imfimage: output-image-portion.png
	I = ocrolib.read_image_binary(imf)
	I = 1-I/I.max()
	rows,cols = I.shape
	
	# Generate Mask and Seed Images
	Imask, Iseed = pixMorphSequence_mask_seed_fill_holes(I)
	
	# Iseedfill: Union of Mask and Seed Images
	Iseedfill = pixSeedfillBinary(Imask, Iseed)
	
	# Dilation of Iseedfill
	mask = ones((3,3))
	Iseedfill = ndimage.binary_dilation(Iseedfill,mask)
	
	# Expansion of Iseedfill to become equal in size of I
	Iseedfill = expansion(Iseedfill,(rows,cols))
	
	# Write  Text and Non-Text images
	image_part = array((1-I*Iseedfill),dtype=int)
	image_part[0,0] = 0 # only for visualisation purpose
	text_part = array((1-I*(1-Iseedfill)),dtype=int)
	text_part[0,0] = 0 # only for visualisation purpose
	
	base,_ = ocrolib.allsplitext(imf)
	ocrolib.write_image_binary(base + ".ts.png",text_part)
	
	#imf_image = imf[0:-3] + "nts.png"
	ocrolib.write_image_binary(base + ".nts.png",image_part)
	return [base + ".ts.png", base + ".nts.png"]
	
def pixMorphSequence_mask_seed_fill_holes(I):
	Imask = reduction_T_1(I)
	Imask = reduction_T_1(Imask)
	Imask = ndimage.binary_fill_holes(Imask)
	Iseed = reduction_T_4(Imask)
	Iseed = reduction_T_3(Iseed)
	mask = array(ones((5,5)),dtype=int);
	Iseed = ndimage.binary_opening(Iseed,mask)
	Iseed = expansion(Iseed,Imask.shape)	
	return Imask, Iseed

def pixSeedfillBinary(Imask, Iseed):
	Iseedfill = copy.deepcopy(Iseed)
	s=ones((3,3))
	Ijmask, k = ndimage.label(Imask,s)
	Ijmask2  = Ijmask * Iseedfill
	A = list(unique(Ijmask2))
	A.remove(0)
	for i in range(0,len(A)):
		x,y = where(Ijmask==A[i])
		Iseedfill[x,y] = 1
	return Iseedfill
		
def reduction_T_1(I):
	A = logical_or(I[0:-1:2,:],I[1::2,:])
	A = logical_or(A[:,0:-1:2],A[:,1::2])
	return A

def reduction_T_2(I):
	A = logical_or(I[0:-1:2,:],I[1::2,:])
	A = logical_and(A[:,0:-1:2],A[:,1::2])
	B = logical_and(I[0:-1:2,:],I[1::2,:])
	B = logical_or(B[:,0:-1:2],B[:,1::2])
	C = logical_or(A,B)
	return C
	
def reduction_T_3(I):
	A = logical_or(I[0:-1:2,:],I[1::2,:])
	A = logical_and(A[:,0:-1:2],A[:,1::2])
	B = logical_and(I[0:-1:2,:],I[1::2,:])
	B = logical_or(B[:,0:-1:2],B[:,1::2])
	C = logical_and(A,B)
	return C

def reduction_T_4(I):
	A = logical_and(I[0:-1:2,:],I[1::2,:])
	A = logical_and(A[:,0:-1:2],A[:,1::2])
	return A

def expansion(I,rows_cols):
	r,c = I.shape
	rows,cols = rows_cols
	A = zeros((rows,cols))
	A[0:4*r:4,0:4*c:4] = I
	A[1:4*r:4,:] = A[0:4*r:4,:]
	A[2:4*r:4,:] = A[0:4*r:4,:]
	A[3:4*r:4,:] = A[0:4*r:4,:]
	A[:,1:4*c:4] = A[:,0:4*c:4]
	A[:,2:4*c:4] = A[:,0:4*c:4]
	A[:,3:4*c:4] = A[:,0:4*c:4]
	return A


if __name__ == "__main__":
	myparser = ParserAnybaseocr()
	args = myparser.get_parameters('ocrd-anybaseocr-tiseg')

	files = myparser.parseXML()
	fname = []
	block=[]
	for i, f in enumerate(files):
		myparser.print_info("Process file: %s" % str(f))
		block.append(textimageseg(str(f)))
		fname.append(str(f))

	myparser.write_to_xml(fname, 'TS_', block)
