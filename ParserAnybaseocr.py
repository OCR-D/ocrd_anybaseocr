#!/usr/bin/env python
import argparse,os,os.path,glob
import json
from xml.dom import minidom
import ocrolib

class ParserAnybaseocr:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.parser.add_argument('-p','--parameter',type=str,help="Parameter file location")
		self.parser.add_argument('-w','--work',type=str,help="Working directory location", default=".")
		self.parser.add_argument('-I','--Input',default=None,help="Input directory")
		self.parser.add_argument('-O','--Output',default=None,help="output directory")
		self.parser.add_argument('-m','--mets',default=None,help="METs input file")
		self.parser.add_argument('-o','--OutputMets',default=None,help="METs output file")

		args = self.parser.parse_args()
		# mendatory parameter check
		if not args.mets or not args.Input or not args.Output or not args.work:
		    print("Example: python script.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
		    sys.exit(0)

		if args.work:
		    if not os.path.exists(args.work):
		        os.mkdir(args.work)		    

		self.param_path = args.parameter
		self.work_dir = args.work
		self.input_grp = args.Input
		self.output_grp = args.Output
		self.met_path = args.mets
		self.out_met_path = args.OutputMets
		pass

	## Read parameter values from json file
	def get_parameters(self, param_block):
	    if self.param_path:
	        if not os.path.exists(self.param_path):
	            print("Error : Parameter file does not exists.")
	            sys.exit(0)
	        else:
	            with open(self.param_path) as json_file:
	                json_data = json.load(json_file)
	    else:
	        parameter_path = os.path.dirname(os.path.realpath(__file__))
	        if not os.path.exists(os.path.join(parameter_path, 'ocrd-anybaseocr-parameter.json')):
	            print("Error : Parameter file does not exists.")
	            sys.exit(0)
	        else:
	            with open(os.path.join(parameter_path, 'ocrd-anybaseocr-parameter.json')) as json_file:
	                json_data = json.load(json_file)
	    parser = self.parse_data(json_data['tools'][param_block]['parameters'])
	    parameters = parser.parse_args()
	    return parameters

	def parse_data(self, arguments):
		for key, val in arguments.items():
			self.parser.add_argument('--%s' % key,
		            type=eval(val["type"]),
		            help=val["description"],
		            default=val["default"])
		return self.parser

	def parseXML(self):
	    input_files=[]
	    xmldoc = minidom.parse(self.met_path)
	    nodes = xmldoc.getElementsByTagName('mets:fileGrp')
	    for attr in nodes:
	        if attr.attributes['USE'].value==self.input_grp:
	            childNodes = attr.getElementsByTagName('mets:FLocat')
	            for f in childNodes:
	                input_files.append(f.attributes['xlink:href'].value)
	    return input_files

	def write_to_xml(self, fpath, tag, bpath=None):
	    xmldoc = minidom.parse(self.met_path)
	    subRoot = xmldoc.createElement('mets:fileGrp')
	    subRoot.setAttribute('USE', self.output_grp)

	    if bpath is None:
	    	for f in fpath:
	    		basefile = ocrolib.allsplitext(os.path.basename(f))[0]
	    		child = xmldoc.createElement('mets:file')
	    		child.setAttribute('ID', tag + basefile)
	    		child.setAttribute('GROUPID', 'P_' + basefile)
	    		child.setAttribute('MIMETYPE', "image/png")

	    		subChild = xmldoc.createElement('mets:FLocat')
	    		subChild.setAttribute('LOCTYPE', "URL")
	    		subChild.setAttribute('xlink:href', f)

	    		subRoot.appendChild(child)
	    		child.appendChild(subChild)
	    else:
	    	for f, b in zip(fpath,bpath):
	    		basefile = ocrolib.allsplitext(os.path.basename(f))[0]
	    		child = xmldoc.createElement('mets:file')
	    		child.setAttribute('ID', tag + basefile)
	    		child.setAttribute('GROUPID', 'P_' + basefile)
	    		child.setAttribute('MIMETYPE', "image/png")

	    		for block in b:
	    			subChild = xmldoc.createElement('mets:FLocat')
	    			subChild.setAttribute('LOCTYPE', "URL")
	    			subChild.setAttribute('xlink:href', block)
	    			child.appendChild(subChild)

	    		subRoot.appendChild(child)

	    #subRoot.appendChild(child)
	    xmldoc.getElementsByTagName('mets:fileSec')[0].appendChild(subRoot);
	    
	    if not self.out_met_path:
	        metsFileSave = open(os.path.join(self.work_dir, os.path.basename(self.met_path)), "w")
	    else:
	        metsFileSave = open(os.path.join(self.work_dir, self.out_met_path if self.out_met_path.endswith(".xml") else self.out_met_path+'.xml'), "w")
	    metsFileSave.write(xmldoc.toxml())

	def pageXML_deskew():
		pass
	
	def pageXML_cropping():
		pass		


	    
	def print_info(self, msg):
	    print("INFO: %s" % msg)

	def print_error(self, msg):
	    print("ERROR: %s" % msg)
