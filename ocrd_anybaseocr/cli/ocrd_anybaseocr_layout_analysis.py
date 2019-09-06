import os
import sys
import pickle

import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) 
import tensorflow as tf
from keras.layers import Input
from keras.models import load_model
from keras.applications import inception_v3, resnet50
from keras.preprocessing.image import ImageDataGenerator

from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, parse, TextRegionType
from ocrd_utils import getLogger, concat_padded, MIMETYPE_PAGE
from ocrd_models.ocrd_page_generateds import RegionType

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from pathlib import Path
import ocrolib
from PIL import Image

from lxml import etree as ET

from ocrd_models.ocrd_mets import OcrdMets
from ocrd_models.constants import (
    NAMESPACES as NS,
    TAG_METS_AGENT,
    TAG_METS_DIV,
    TAG_METS_FILE,
    TAG_METS_FILEGRP,
    TAG_METS_FILESEC,
    TAG_METS_FPTR,
    TAG_METS_METSHDR,
    TAG_METS_STRUCTMAP,
    IDENTIFIER_PRIORITY,
    TAG_MODS_IDENTIFIER,
    METS_XML_EMPTY,
)

TAG_METS_STRUCTLINK = '{%s}structLink' % NS['mets']
TAG_METS_SMLINK = '{%s}smLink' % NS['mets']




class OcrdAnybaseocrLayoutAnalyser(Processor):

    def __init__(self, *args, **kwargs):
        self.last_result = None 
        self.logID = 0
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-layout-analysis']
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrLayoutAnalyser, self).__init__(*args, **kwargs)

    def define_model(self, model = 'inception_v3', num_classes=34, input_size=(600, 500, 1)):
        input_dims = Input(shape=(input_size))
        if model == "inception_v3":
            model = inception_v3.InceptionV3(include_top=True, weights=None, classes=classes, input_tensor=input_dims)
        elif model == "resnet50":
            model = resnet50.ResNet50(include_top=True, weights=None, input_tensor=input_dims, classes=classes)
        else:
            print('wrong input')
            sys.exit(0)                      
                
    def create_model(self, path, model_name='inception_v3', def_weights=True, num_classes=34, input_size=(600, 500, 1)):
        ''' 
            num_classes : number of classes to predict for
            input_size  : size of input dimension of image 
        '''
        if def_weights:        
            model = load_model(path)
        else:
            model = self.define_model(model_name, num_classes, input_size)
            model.load_weights(path)                            
        return model

    def start_test(self, model, img_array, filename, labels):
        # shape should be 1,600,500 for keras
        pred = model.predict(img_array)
        pred_classes = np.argmax(pred, axis=1)

        # convert label
        # labels = train_generator.class_indices
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in pred_classes]            
        return predictions[0]

    def img_resize(self, image_path):
        size = 600, 500
        img = Image.open(image_path)
        return img.thumbnail(size, Image.ANTIALIAS)

    def write_to_mets(self, result, pageID):  
        if result != self.last_result and result != "page":
            #create div in logmap
            log_div = ET.SubElement(self.log_map, TAG_METS_DIV)
            log_div.set('TYPE', result)            
            log_div.set('ID', "LOG_"+str(self.logID))
            self.logID += 1
            self.last_result = result
        #add smlink
        smLink = ET.SubElement(self.link, TAG_METS_SMLINK)
        smLink.set('{'+NS['xlink']+'}'+'to', pageID)
        smLink.set('{'+NS['xlink']+'}'+'from', "LOG_"+str(self.logID))
    
    def create_logmap_smlink(self, workspace):
        el_root = self.workspace.mets._tree.getroot()
        log_map = el_root.find('mets:structMap[@TYPE="LOGICAL"]', NS)
        if log_map is None:
            log_map = ET.SubElement(el_root, TAG_METS_STRUCTMAP)
            log_map.set('TYPE', 'LOGICAL')
        link = el_root.find('mets:structLink', NS)
        if link is None:
            link = ET.SubElement(el_root, TAG_METS_STRUCTLINK)
        self.link = link
        self.log_map = log_map                        

    def process(self):
        if not tf.test.is_gpu_available():
            print("Your system has no CUDA installed. No GPU detected.")
            sys.exit(1)

        model_path = Path(self.parameter['model_path'])
        class_mapper_path = Path(self.parameter['class_mapping_path'])

        print('Loading model from file ', model_path)
        model = self.create_model(str(model_path))
        # load the mapping
        pickle_in = open(str(class_mapper_path), "rb")
        class_indices = pickle.load(pickle_in)

        
        if not Path(model_path).is_file():
            print("""\
                Layout Classfication model was not found at '%s'. Make sure the `model_path` parameter
                points to the local model path.

                model can be downloaded from http://url
                """ % model_path)
            sys.exit(1)
        else:
            print('Loading model from file ', model_path)
            model = self.create_model(str(model_path))
            # load the mapping
            pickle_in = open(str(class_mapper_path), "rb")
            class_indices = pickle.load(pickle_in)
        

        for (n, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            fname = pcgts.get_Page().imageFilename
            print(fname)
            size = 600, 500
            img = Image.open(fname)
            img_array = ocrolib.pil2array(img.resize((500, 600), Image.ANTIALIAS))
            img_array = img_array[np.newaxis, :, :, np.newaxis]            
            results = self.start_test(model, img_array, fname, class_indices)
            print(results)
            self.workspace.mets.set_physical_page_for_file("PHYS_000" + str(n) , input_file)
            self.create_logmap_smlink(pcgts)
            self.write_to_mets(results, "PHYS_000" + str(n))
            