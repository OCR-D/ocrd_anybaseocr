import sys
import pickle
import numpy as np 
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) 
from collections import defaultdict
from ..constants import OCRD_TOOL

import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml,
    MetadataItemType,
    LabelsType, LabelType
)
from ocrd_utils import getLogger
from ocrd_models import ocrd_mets

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

from ..tensorflow_importer import tf, keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

TAG_METS_STRUCTLINK = '{%s}structLink' % NS['mets']
TAG_METS_SMLINK = '{%s}smLink' % NS['mets']


TOOL = 'ocrd-anybaseocr-layout-analysis'
LOG = getLogger('OcrdAnybaseocrLayoutAnalyser')


class OcrdAnybaseocrLayoutAnalyser(Processor):

    def __init__(self, *args, **kwargs):
        
        self.last_result = [] 
        self.logID = 0 # counter for new key
        self.logIDs = defaultdict(int) # dict to keep track of previous keys for labels other then chapter or section
        self.log_id = 0 # var to keep the current ongoing key
        self.log_links = {}
        self.first = None
        
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrLayoutAnalyser, self).__init__(*args, **kwargs)
                
    def create_model(self, path ):#model_name='inception_v3', def_weights=True, num_classes=34, input_size=(600, 500, 1)):

        ''' 
            path: string containing path to model definition 
        '''
        model = load_model(path)
        return model

    def start_test(self, model, img_array, filename, labels):
    
        # shape should be 1,600,500 for keras
        pred = model.predict(img_array)
        pred = np.array(pred)
        # multi-label predictions
        if len(pred.shape)>2:        
            pred = np.squeeze(pred)
            pred = pred.T

        preds = (pred>=0.5)
        predictions = []
        for index, cls in enumerate(preds):
            if cls:
                predictions.append(labels[index])
        
        if len(predictions) == 0:
            # if no prediction get the maximum one
            predictions.append(labels[np.argmax(pred)])
            #predictions.append('page') # default label
     
        return predictions

    def img_resize(self, image_path):
        size = 600, 500
        img = Image.open(image_path)
        return img.thumbnail(size, Image.ANTIALIAS)    
    
    def write_to_mets(self, result, pageID):  
        
        for i in result:   
            create_new_logical = False
            # check if label is page skip 
            if i !="page":
            
            # if not page, chapter and section then its something old
                
                if i!="chapter" and i!="section":
        
                    if i in self.last_result:
                        self.log_id = self.logIDs[i]
                    else:
                        create_new_logical = True

                    if i =='binding':
                        parent_node = self.log_map
                    
                    if i=='cover' or i=='endsheet' or i=='paste_down':

                        # get the link for master node
                        parent_node = self.log_links['binding']

                    else:
                        
                        if self.first is not None and i!='title_page':
                            parent_node = self.log_links[self.first]
                        else:
                            parent_node = self.log_map
                            
                else:
                    create_new_logical = True
                    
                    if self.first is None:
                        self.first = i
                        parent_node = self.log_map
                            
                    else:
                        if self.first == i:
                            parent_node = self.log_map
                        else:
                            parent_node = self.log_links[self.first]
                            
                    
                if create_new_logical:
        
                    log_div = ET.SubElement(parent_node, TAG_METS_DIV)
                    log_div.set('TYPE', str(i))            
                    log_div.set('ID', "LOG_"+str(self.logID))
                      
                    self.log_links[i] = log_div # store the link 
                    #if i!='chapter' and i!='section':
                    self.logIDs[i] = self.logID
                    self.log_id = self.logID
                    self.logID += 1
                                        
            
            else:
                if self.logIDs['chapter'] > self.logIDs['section']:
                    self.log_id = self.logIDs['chapter']
                 
                if self.logIDs['section'] > self.logIDs['chapter']:
                    self.log_id = self.logIDs['section']
                    
                if self.logIDs['chapter']==0 and self.logIDs['section']==0:
                    
                    # if both chapter and section dont exist
                    if self.first is None:
                        self.first = 'chapter'
                        parent_node = self.log_map
                            
                    log_div = ET.SubElement(parent_node, TAG_METS_DIV)
                    log_div.set('TYPE', str(i))            
                    log_div.set('ID', "LOG_"+str(self.logID))
                      
                    self.log_links[i] = log_div # store the link 
                    #if i!='chapter' and i!='section':
                    self.logIDs[i] = self.logID
                    self.log_id = self.logID
                    self.logID += 1                    
                
                
            smLink = ET.SubElement(self.link, TAG_METS_SMLINK)
            smLink.set('{'+NS['xlink']+'}'+'to', pageID)
            smLink.set('{'+NS['xlink']+'}'+'from', "LOG_"+str(self.log_id))
        
        self.last_result = result
    
    def create_logmap_smlink(self, workspace):
        el_root = self.workspace.mets._tree.getroot()
        log_map = el_root.find('mets:structMap[@TYPE="LOGICAL"]', NS)
        if log_map is None:
            log_map = ET.SubElement(el_root, TAG_METS_STRUCTMAP)
            log_map.set('TYPE', 'LOGICAL')
        else:
            LOG.info('LOGICAL structMap already exists, adding to it')
        link = el_root.find('mets:structLink', NS)
        if link is None:
            link = ET.SubElement(el_root, TAG_METS_STRUCTLINK)
        self.link = link
        self.log_map = log_map                        

    def process(self):
        if not tf.test.is_gpu_available():
            LOG.error("Your system has no CUDA installed. No GPU detected.")
            # sys.exit(1)
        model_path = Path(self.parameter['model_path'])
        class_mapper_path = Path(self.parameter['class_mapping_path'])
        if not Path(model_path).is_file():
            LOG.error("""\
                Layout Classfication model was not found at '%s'. Make sure the `model_path` parameter
                points to the local model path.
                model can be downloaded from http://url
                """ % model_path)
            sys.exit(1)
        else:
            
            LOG.info('Loading model from file %s', model_path)
            model = self.create_model(str(model_path))
            # load the mapping
            pickle_in = open(str(class_mapper_path), "rb")
            class_indices = pickle.load(pickle_in)
            label_mapping = dict((v,k) for k,v in class_indices.items())    
        
            # print("INPUT FILE HERE",self.input_files)
        for (n, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            fname = pcgts.get_Page().imageFilename
            page_id = input_file.pageId or input_file.ID
            size = 600, 500
            
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                    MetadataItemType(type_="processingStep",
                                     name=self.ocrd_tool['steps'][0],
                                     value=TOOL,                                     
                                     Labels=[LabelsType(#externalRef="parameter",
                                                        Label=[LabelType(type_=name,
                                                                         value=self.parameter[name])
                                                               for name in self.parameter.keys()])]))

            page = pcgts.get_Page()
            LOG.info("INPUT FILE %s", input_file.pageId or input_file.ID)
            
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_selector='binarized')


            img_array = ocrolib.pil2array(page_image.resize((500, 600), Image.ANTIALIAS))
            img_array = img_array * 1./255.
            img_array = img_array[np.newaxis, :, :, np.newaxis]            
            results = self.start_test(model, img_array, fname, label_mapping)
            LOG.info(results)
            self.workspace.mets.set_physical_page_for_file("PHYS_000" + str(n) , input_file)
            self.create_logmap_smlink(pcgts)
            self.write_to_mets(results, "PHYS_000" + str(n))

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrLayoutAnalyser, *args, **kwargs)
