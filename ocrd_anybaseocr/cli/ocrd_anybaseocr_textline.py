import builtins as python
import random as pyrandom
import sys
import os
import re
import glob
import argparse
import codecs
from pylab import median, imread
from PIL import Image
import ocrolib
from re import split
import argparse
import os.path
import json
from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults, print_error
from ..constants import OCRD_TOOL

#
import subprocess


# limits
# parser.add_argument('--minscale',type=float,default=8.0,
#                    help='minimum scale permitted (%(default)s)') # default was 12.0, Ajraf, Mohsin and Saqib chnaged it into 8.0

from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults, print_error
from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_utils import concat_padded


class OcrdAnybaseocrTextline(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-textline']
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrTextline, self).__init__(*args, **kwargs)

    def addzeros(self, file):
        F = open(file, "r")
        D = F.read()
        D = split("\n", D)
        D = D[:-1]
        F.close()
        F = open(file, "w")
        for d in D:
            d += " 0 0 0 0\n"
            F.write(d)

    def process(self):
        for (n, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            binImg = self.workspace.resolve_image_as_pil(pcgts.get_Page().imageFilename)
            # I: binarized-input-image; imftext: output-text-portion.png; imfimage: output-image-portion.png
            fname = pcgts.get_Page().imageFilename
            image = ocrolib.read_image_binary(fname)
            height, width = image.shape
            H = height
            W = width
            base, _ = ocrolib.allsplitext(fname)
            base2 = os.path.splitext(fname)[0]

            if not os.path.exists("%s/lines" % base):
                os.system("mkdir -p %s/lines" % base)
                # if os.path.exists(base2 + ".ts.png") :
                #    f = ocrolib.read_image_binary(base2 + ".ts.png")
                #    height, width = f.shape
                #    os.system("python "+args.libpath+"/anyBaseOCR-nlbin.py %s.pf.bin.png" % base2)
                # else:
                #    os.system("python "+args.libpath+"/anyBaseOCR-nlbin.py %s" % arg)
                #print("convert %s.ts.png %s/block-000.bin.png" % (base,base))
                #os.system("convert %s.ts.png %s/block-000.bin.png" % (base,base))
                #os.system("rm %s.bin.png %s.nrm.png" % (base, base))
                file = open('%s/sorted_cuts.dat' % base, 'w')
                l = "0 0 " + str(int(width)) + " " + str(int(height)) + " 0 0 0 0\n"
                file.write(l)
                file.close()

            # if not os.path.exists("%s/lines" % base) :
            #    os.system("mkdir %s/lines" % base)

            blockarray = []
            if os.path.exists(base + "/sorted_cuts.dat"):
                blocks = open(base + "/sorted_cuts.dat", "r")
                i = 0
                for block in blocks:
                    words = block.split()
                    blockarray.append((int(words[0]), -int(words[1]), int(words[2]), int(words[3]), i))
                    i += 1
            else:
                blockarray.append((0, 0, width, height, 0))

            i = 0
            j = 0
            lines = []
            for block in blockarray:
                (x0, y0, x1, y1, i) = block
                y0 = -y0
                #blockImage = "%s/block-%03d" % (base, i)
                os.system("convert %s.ts.png %s/temp.png" % (base, base))
                img = Image.open("%s.ts.png" % base, 'r')
                img_w, img_h = img.size
                background = Image.new('RGBA', (W, H), (255, 255, 255, 255))
                bg_w, bg_h = background.size
                offX = (bg_w - img_w) // 2
                offY = (bg_h - img_h) // 2
                offset = (offX, offY)
                background.paste(img, offset)
                background.save("%s/temp.png" % base)
                command = "python "+self.parameter['libpath']+"/cli/anyBaseOCR-gpageseg.py %s/temp.png -n --minscale %f --maxlines %f --scale %f --hscale %f --vscale %f --threshold %f --noise %d --maxseps %d --sepwiden %d --maxcolseps %d --csminaspect %f --csminheight %f -p %d -e %d -Q %d" % (
                    base, self.parameter['minscale'], self.parameter['maxlines'], self.parameter['scale'], self.parameter['hscale'], self.parameter['vscale'], self.parameter['threshold'], self.parameter['noise'], self.parameter['maxseps'], self.parameter['sepwiden'], self.parameter['maxcolseps'], self.parameter['csminaspect'], self.parameter['csminheight'], self.parameter['pad'], self.parameter['expand'], self.parameter['parallel'])
                if(self.parameter['blackseps']):
                    command = command + " -b"
                if(self.parameter['usegauss']):
                    command = command + " --usegauss"
                os.system(command)
                pseg = ocrolib.read_page_segmentation("%s/temp.pseg.png" % base)
                regions = ocrolib.RegionExtractor()
                regions.setPageLines(pseg)
                file = open('%s/sorted_lines.dat' % base, 'w')
                for h in range(1, regions.length()):
                    id = regions.id(h)
                    y0, x0, y1, x1 = regions.bbox(h)
                    l = str(int(x0 - offX)) + " " + str(int(img_h - (y1 - offY))) + " " + str(int(x1 - offX)) + " " + str(int(img_h - (y0 - offY))) + " 0 0 0 0\n"
                    file.write(l)
                filelist = glob.glob("%s/temp/*" % base)
                for infile in sorted(filelist):
                    os.system("convert %s %s/lines/01%02x%02x.bin.png" % (infile, base, i + 1, j + 1))
                    lines.append("%s/lines/01%02x%02x.bin.png" % (base, i + 1, j + 1))
                    j += 1
                os.system("rm -r %s/temp/" % base)
                os.system("rm %s/temp.png %s/temp.pseg.png" % (base, base))
                i += 1
            # return lines


'''
def main():
	parser = argparse.ArgumentParser("""
	Image Text Line Segmentation.

			  
			ocrd-anybaseocr-textline -m -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

	This is a compute-intensive text line segmentation method that works on degraded
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
	param = parse_params_with_defaults(param, OCRD_TOOL['tools']['ocrd-anybaseocr-textline']['parameters'])

	if not args.mets or not args.Input or not args.Output or not args.work:
		parser.print_help()
		print("Example: ocrd_anyBaseOCR_tigseg -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
		sys.exit(0)

	if args.work:
		if not os.path.exists(args.work):
			os.mkdir(args.work)	
	
	textline = OcrdAnybaseocrTextline(param)				
	files = parseXML(args.mets, args.Input)
	fnames = []
	block=[]
	for i, fname in enumerate(files):
		print_info("Process file: %s" % str(fname))
		block.append(textline.textline(str(fname)))
		fnames.append(str(fname))
	write_to_xml(fnames, args.mets, args.Output, args.OutputMets, args.work)

'''
