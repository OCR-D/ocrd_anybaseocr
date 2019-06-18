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
import ocrolib
import argparse
import sys
import os
import os.path
import sys
import json

from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults, print_error
from ..constants import OCRD_TOOL


class OcrdAnybaseocrTiseg:

	def __init__(self, param):
		self.param = param


	def textimageseg(self,imf):
		# I: binarized-input-image; imftext: output-text-portion.png; imfimage: output-image-portion.png
		I = ocrolib.read_image_binary(imf)
		I = 1-I/I.max()
		rows,cols = I.shape
		
		# Generate Mask and Seed Images
		Imask, Iseed = self.pixMorphSequence_mask_seed_fill_holes(I)
		
		# Iseedfill: Union of Mask and Seed Images
		Iseedfill = self.pixSeedfillBinary(Imask, Iseed)
		
		# Dilation of Iseedfill
		mask = ones((3,3))
		Iseedfill = ndimage.binary_dilation(Iseedfill,mask)
		
		# Expansion of Iseedfill to become equal in size of I
		Iseedfill = self.expansion(Iseedfill,(rows,cols))
		
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
		
	def pixMorphSequence_mask_seed_fill_holes(self, I):
		Imask = self.reduction_T_1(I)
		Imask = self.reduction_T_1(Imask)
		Imask = ndimage.binary_fill_holes(Imask)
		Iseed = self.reduction_T_4(Imask)
		Iseed = self.reduction_T_3(Iseed)
		mask = array(ones((5,5)),dtype=int);
		Iseed = ndimage.binary_opening(Iseed,mask)
		Iseed = self.expansion(Iseed,Imask.shape)	
		return Imask, Iseed

	def pixSeedfillBinary(self, Imask, Iseed):
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
			
	def reduction_T_1(self, I):
		A = logical_or(I[0:-1:2,:],I[1::2,:])
		A = logical_or(A[:,0:-1:2],A[:,1::2])
		return A

	def reduction_T_2(self, I):
		A = logical_or(I[0:-1:2,:],I[1::2,:])
		A = logical_and(A[:,0:-1:2],A[:,1::2])
		B = logical_and(I[0:-1:2,:],I[1::2,:])
		B = logical_or(B[:,0:-1:2],B[:,1::2])
		C = logical_or(A,B)
		return C
		
	def reduction_T_3(self, I):
		A = logical_or(I[0:-1:2,:],I[1::2,:])
		A = logical_and(A[:,0:-1:2],A[:,1::2])
		B = logical_and(I[0:-1:2,:],I[1::2,:])
		B = logical_or(B[:,0:-1:2],B[:,1::2])
		C = logical_and(A,B)
		return C

	def reduction_T_4(self, I):
		A = logical_and(I[0:-1:2,:],I[1::2,:])
		A = logical_and(A[:,0:-1:2],A[:,1::2])
		return A

	def expansion(self, I, rows_cols):
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


def main():
	parser = argparse.ArgumentParser("""
	Image Text and Non-Text detection.

			  
			ocrd-anybaseocr-tiseg -m -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

	This is a compute-intensive text/non-text segmentation method that works on degraded
	and historical book pages.
	""")

	parser.add_argument('-p', '--parameter', type=str, help="Parameter file location")
	parser.add_argument('-w', '--work', type=str, help="Working directory location", default=".")
	parser.add_argument('-I', '--Input', default=None, help="Input directory")
	parser.add_argument('-O', '--Output', default=None, help="output directory")
	parser.add_argument('-m', '--mets', default=None, help="METs input file")
	parser.add_argument('-o', '--OutputMets', default=None, help="METs output file")
	parser.add_argument('-g', '--group', default=None, help="METs image group id")
	args = parser.parse_args()

	param = {}
	if args.parameter:
		with open(args.parameter, 'r') as param_file:
			param = json.loads(param_file.read())
	param = parse_params_with_defaults(param, OCRD_TOOL['tools']['ocrd-anybaseocr-tiseg']['parameters'])

	if not args.mets or not args.Input or not args.Output or not args.work:
		parser.print_help()
		print("Example: ocrd_anyBaseOCR_tigseg -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
		sys.exit(0)

	if args.work:
		if not os.path.exists(args.work):
			os.mkdir(args.work)
	
	tiseg = OcrdAnybaseocrTiseg(param)
	files = parseXML(args.mets, args.Input)
	fnames = []
	block=[]
	for i, fname in enumerate(files):
		print_info("Process file: %s" % str(fname))
		block.append(tiseg.textimageseg(str(fname)))
		fnames.append(str(fname))
	write_to_xml(fnames, args.mets, args.Output, args.OutputMets, args.work)

	'''
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
'''